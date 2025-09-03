"""
chat_phase_text_v3.py

Conversational small-talk agent for an elderly-care robot (EN-only), powered by
Google Gemini via google.generativeai. This version implements:

1) Consecutive-question cap (max 2). After that, switch to
   backchannel + reflection + gentle invitation (no options UI, no lists).
2) Time-bucket awareness embedded in the system instruction (morning/afternoon/evening/night)
   to avoid time-incongruent topics unless the user brings them up.
3) Soft close policy (no fixed turn limit). Close is suggested only when engagement is low
   OR user indicates completion; never close if the user is asking a question or sharing at length.
4) Health talk allowed, but with medical-advice guard. Emergency symptom triggers switch to a
   safety template (no diagnosis or treatment suggestions).

Run (pick ONE):
  1) Hardcode: set DEFAULT_API_KEY = "YOUR_KEY_HERE" in this file, then
     python chat_phase_text_v3.py --model gemini-1.5-flash
  2) CLI:   python chat_phase_text_v3.py --model gemini-1.5-flash --api_key YOUR_KEY_HERE
  3) Env:   export GEMINI_API_KEY=your_key && python chat_phase_text_v3.py --model gemini-1.5-flash

Stop with Ctrl+C or typing "exit".
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any

import google.generativeai as genai

# --- Hardcoded key (NOT recommended for production). Paste your key here if you prefer. ---
DEFAULT_API_KEY = ""  # e.g., "AIza...". If left blank, the code falls back to --api_key or ENV.

# -----------------------------
# Config & templates
# -----------------------------

def time_bucket_from_local(now: datetime | None = None) -> str:
    now = now or datetime.now()
    h = now.hour
    if 5 <= h < 12:
        return "morning"
    if 12 <= h < 17:
        return "afternoon"
    if 17 <= h < 21:
        return "evening"
    return "night"

SAFETY_TEMPLATE = (
    "I’m concerned about your safety. I can’t assess emergencies or give medical advice. "
    "If you’re in danger or feel severe or sudden symptoms, please contact local emergency services "
    "or a trusted person right now. I can stay with you while you do."
)

SOFT_CLOSE_TEMPLATE = (
    "Let’s pause here for now—rest matters. I’m here whenever you want to talk again."
)

SYSTEM_POLICY = """
ROLE: You are a warm, simple, English-only small‑talk companion for an elderly‑care robot.
Current time bucket: {TIME_BUCKET}.
Tone: friendly, reassuring. Use 1–2 short sentences per turn. Use light backchannels (e.g., ‘I see’, ‘That sounds nice’).
Avoid time‑incongruent topics unless the user brings them up.

Capabilities & scope:
- Conversational‑only. You do NOT perform real‑world actions, control devices, play media, place calls, send messages, or operate apps.
- Never claim to start/stop/turn on/launch/do anything. Offer suggestions in plain language instead (e.g., “Music might help; I can suggest ideas,” not “I’ll play music now.”).

Question pacing:
- At most one NEW question per turn. Never exceed two consecutive new questions.
- If question_streak >= 2, do NOT ask another question. Produce: backchannel + short reflection/paraphrase + gentle invitation that does NOT end with a question mark.

Style:
- No menus or A/B options; keep it natural.
- Specificity: Prefer concrete, everyday questions grounded in the user’s last message or the current time bucket.
- Avoid generic prompts like "anything else" or "what's on your mind"; ask about one tangible detail instead.
- Health talk is allowed, but do not diagnose or suggest medication changes. If the user mentions emergency symptoms, reply ONLY with SAFETY_TEMPLATE.

Closing:
- No fixed turn limit. If engagement is low or the user signals completion, gently close after a brief one‑line summary using SOFT_CLOSE_TEMPLATE. If the user asks to continue or shares new content, keep going.
"""

# Emergency / danger heuristics (not exhaustive, but practical)
DANGER_PAT = re.compile(
    r"\b(chest pain|pressure in (my|the) chest|can'?t breathe|short(ness)? of breath|severe headache|"
    r"confusion|fainted|pass(ed)? out|unconscious|bleeding (a lot|heavily)|stroke|slurred speech|"
    r"suicidal|kill myself|want to die|jump off|overdose)\b",
    re.IGNORECASE,
)

# Light end/thanks intent
END_PAT = re.compile(
    r"(that'?s (all|it)|we'?re done|you can stop|you may stop|stop here|that will be all)",
    re.IGNORECASE,
)

# Low‑engagement markers
LOW_ACK = {"ok","okay","yeah","yep","fine","good","thanks","thank you","alright","mm","mmm","uh-huh","uh huh","no","nope","nah","nothing"}

# Simple question detector for assistant outputs
QUESTION_PAT = re.compile(
    r"\?\s*$|\b(what|when|where|who|why|how|could you|would you|can you|do you|are you|should we|shall we|did you)\b",
    re.IGNORECASE,
)


def looks_like_low_engagement(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return True
    if t in LOW_ACK:
        return True
    # very short and no punctuation
    if len(t) < 8 and t.isalpha():
        return True
    return False


def looks_like_question(text: str) -> bool:
    return bool(QUESTION_PAT.search((text or "").strip()))

# --- Specificity helpers -------------------------------------------------
GENERIC_Q_PAT = re.compile(
    r"(anything (?:else|you'd like to (?:share|chat about)|on your mind)|"
    r"what(?:'s| is) on your mind|what would you like to talk about|"
    r"how can i help|is there anything you(?:'d)? like to talk about)",
    re.IGNORECASE,
)

TOPIC_NUDGES = {
    "morning": [
        "Did you sleep okay last night?",
        "What did you have for breakfast?",
        "Did you get a bit of fresh air this morning?",
    ],
    "afternoon": [
        "How did your morning go?",
        "Did you have anything tasty for lunch?",
        "Was the weather comfortable when you stepped outside?",
    ],
    "evening": [
        "How was dinner tonight?",
        "Did anything small go well today?",
        "Is there a show or music you enjoy this evening?",
    ],
    "night": [
        "Are you getting comfortable for the night?",
        "Is there anything you want to get ready for tomorrow?",
        "Would a warm drink sound nice before bed?",
    ],
}

def pick_concrete_question(user_text: str, bucket: str) -> str:
    t = (user_text or "").lower()
    if "school" in t or "class" in t:
        return "Is there a class tomorrow you’re preparing for?"
    if "work" in t or "shift" in t:
        return "Was there a small win at work today?"
    if "banana" in t or "breakfast" in t:
        return "How did breakfast taste this morning?"
    if "walk" in t:
        return "Did you notice anything nice on your walk?"
    choices = TOPIC_NUDGES.get(bucket, TOPIC_NUDGES["evening"])
    idx = (sum(map(ord, t)) + len(t)) % len(choices)
    return choices[idx]


def opening_greeting_for_bucket(bucket: str) -> str:
    if bucket == "morning":
        return "Good morning. How's your morning going so far?"
    if bucket == "afternoon":
        return "Good afternoon. How has your day been?"
    if bucket == "evening":
        return "Good evening. How was your day?"
    return "Hi there. How are you feeling tonight?"


def looks_like_closing_text(text: str) -> bool:
    t = (text or "").lower()
    return (
        "let's pause here" in t or "let’s pause here" in t or
        "i’m here whenever you want to talk again" in t or "i'm here whenever you want to talk again" in t or
        "rest matters" in t
    )


@dataclass
class ConvState:
    question_streak: int = 0  # consecutive NEW questions asked by assistant
    time_bucket: str = field(default_factory=lambda: time_bucket_from_local())
    last_two_user_low: List[bool] = field(default_factory=list)  # track last two user low-engagement flags
    last_user_question: bool = False
    closing_suggested: bool = False

    def update_time_bucket(self):
        self.time_bucket = time_bucket_from_local()

    def push_user_engagement(self, is_low: bool):
        self.last_two_user_low.append(is_low)
        if len(self.last_two_user_low) > 2:
            self.last_two_user_low.pop(0)

    def should_soft_close(self, user_text: str, assistant_can_close: bool) -> bool:
        # Never close if user is asking or sharing a lot.
        if looks_like_question(user_text) or len(user_text.strip()) >= 80:
            return False
        # Soft close if low engagement pattern or explicit end pattern.
        if END_PAT.search(user_text):
            return True
        if assistant_can_close and all(self.last_two_user_low[-2:] or [False, False]):
            return True
        return False


class SmallTalkAgent:
    def __init__(self, model_name: str = "gemini-1.5-flash", api_key: str | None = None, debug: bool = False):
        self.debug = debug
        # Precedence: explicit arg > ENV > DEFAULT_API_KEY
        api_key = api_key or os.getenv("GEMINI_API_KEY") or DEFAULT_API_KEY
        if not api_key:
            raise RuntimeError("No API key found. Set DEFAULT_API_KEY, pass --api_key, or set GEMINI_API_KEY.")
        genai.configure(api_key=api_key)

        # Try to attach system_instruction (newer SDKs). Fallback to prompt-injection on older SDKs.
        self._inject_system_each_turn = False
        try:
            self.model = genai.GenerativeModel(
                model_name=model_name,
                system_instruction=SYSTEM_POLICY,
            )
        except TypeError:
            # Older SDKs may not support system_instruction
            self.model = genai.GenerativeModel(model_name=model_name)
            self._inject_system_each_turn = True
        except Exception as e:
            if self.debug:
                print(f"[ERROR] Model init failed: {e}", file=sys.stderr)
            # Try minimal init
            self.model = genai.GenerativeModel(model_name=model_name)
            self._inject_system_each_turn = True

        self.chat = self.model.start_chat(history=[])

    def _gen(self, prompt: str, time_bucket: str, max_tokens: int = 180) -> str:
        # Build a single message string (more compatible across SDK versions)
        system_text = SYSTEM_POLICY.replace("{TIME_BUCKET}", time_bucket)
        combined = f"""[SYSTEM]
{system_text}

[TURN]
{prompt}"""
        try:
            resp = self.chat.send_message(
                combined,
                generation_config={
                    "max_output_tokens": max_tokens,
                    "temperature": 0.6,
                    "top_p": 0.9,
                },
            )
            return (resp.text or "").strip()
        except Exception as e:
            if self.debug:
                print(f"[ERROR] send_message failed: {e}", file=sys.stderr)
            return "Sorry, I had trouble speaking just now. Could we try again?"

    def reply(self, user_text: str, state: ConvState) -> str:
        state.update_time_bucket()

        # Safety first.
        if DANGER_PAT.search(user_text or ""):
            # Reset streak; only safety template
            state.question_streak = 0
            return SAFETY_TEMPLATE

        # Engagement tracking
        low = looks_like_low_engagement(user_text)
        state.push_user_engagement(low)
        state.last_user_question = looks_like_question(user_text)

        # Decide mode based on question streak
        if state.question_streak >= 2:
            mode = "reflect_invite_no_question"
            pacing_note = (
                "You have already asked two consecutive questions. "
                "Now you must NOT ask a question. Instead, produce: backchannel + short reflection/paraphrase + gentle invitation WITHOUT a question mark."
            )
        else:
            mode = "open_chat"
            pacing_note = (
                "You may ask at most one new question this turn. Keep it natural and short."
            )

        # Soft close consideration (no hard cap)
        assistant_can_close = True  # allow agent discretion based on signals
        should_close = state.should_soft_close(user_text, assistant_can_close)
        soft_close_flag = "YES" if should_close else "NO"

        # Build turn prompt
        prompt = (
            f"[MODE] {mode}"
            f"[PACing] {pacing_note}"
            f"[USER] {user_text.strip()}"
            f"[SOFT_CLOSE] {soft_close_flag}"
            f"[GUIDANCE] If the user asks a question, you may answer directly even if question_streak >= 2, then follow with a gentle invitation WITHOUT a question."
            f"[CLOSING] If [SOFT_CLOSE] = YES: first give a one-line summary, THEN output exactly this closing line: {SOFT_CLOSE_TEMPLATE} End your message after the closing line. If NO: do not close."
            f"[SAFETY_TEMPLATE] {SAFETY_TEMPLATE}"
            "Respond in 1–2 short sentences. No bullet lists. No option menus."
        )

        draft = self._gen(prompt, state.time_bucket)
        # Note: system enforces conversational-only; no action claims.

        # Guard: if we are in no-question mode but the model asked a question, regenerate once.
        if mode == "reflect_invite_no_question" and looks_like_question(draft):
            prompt2 = (
                f"You must NOT ask a question in your next message. Rewrite the previous draft as backchannel + reflection + gentle invitation WITHOUT any question marks.\n"
                f"Previous draft: {draft}"
            )
            draft = self._gen(prompt2, state.time_bucket)

        # Sanitize action-claims → convert to suggestions (conversational-only)
        actiony = re.search(r"(I'?ll|I will|Let me|I can (?:start|turn|play|call|message|text|book|order)|I'll start|Starting|Turning on)", draft, re.IGNORECASE)
        if actiony:
            rewrite = (
                "Rewrite the following so it does NOT claim to perform any action. "
                "Offer a gentle suggestion in plain language instead. No promises to do things."
                f"Original: {draft}"
            )
            draft = self._gen(rewrite, state.time_bucket)

        # If soft close is appropriate but the draft lacks a closing, force a closing rewrite
        if should_close and not looks_like_closing_text(draft):
            force = (
                f"[SOFT_CLOSE] YES[USER] {user_text.strip()}"
                f"Produce a one-line summary and then exactly this closing line: {SOFT_CLOSE_TEMPLATE}"
                "End your message after the closing line."
            )
            draft = self._gen(force, state.time_bucket)

        # If we decided to soft close and we now have a closing, mark it so the outer loop can exit
        if should_close and looks_like_closing_text(draft):
            state.closing_suggested = True

        # Update question streak depending on assistant output
        if looks_like_question(draft):
            state.question_streak += 1
        else:
            state.question_streak = 0

        return draft


# -----------------------------
# CLI loop
# -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gemini-1.5-flash")
    parser.add_argument("--api_key", default=None)  # optional override
    parser.add_argument("--debug", action="store_true")  # print errors to stderr when API calls fail
    args = parser.parse_args()

    agent = SmallTalkAgent(model_name=args.model, api_key=args.api_key, debug=args.debug)
    state = ConvState()

    print("Small-talk agent ready. Type 'exit' to quit.")

    # Auto opening greeting
    greet = opening_greeting_for_bucket(state.time_bucket)
    print(f"Bot: {greet}")
    # Count this as one question to respect pacing rules
    state.question_streak = 1

    try:
        while True:
            user = input("You: ").strip()
            if not user:
                continue
            if user.lower() in {"exit", "quit"}:
                print("Bot: Take care. I’ll be here when you need me.")
                break

            bot = agent.reply(user, state)
            print(f"Bot: {bot}")
            if state.closing_suggested:
                break
    except KeyboardInterrupt:
        print("\nBot: Bye for now—rest well.")


if __name__ == "__main__":
    main()