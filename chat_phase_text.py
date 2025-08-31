# -*- coding: utf-8 -*-
"""
CHAT phase only (text I/O)
- Small talk for N turns
- Health-topic guard so it doesn't drift into ad-hoc health questions
- (Optional) Extracts enabled health signals from user utterances and prints a handoff JSON at the end
  -> Pipe this into the CHECK script as --prefill-json

Deps: pip install google-genai
"""

from google import genai
from google.genai import types, errors

import argparse, os, sys, time, json, random, re, unicodedata
from enum import Enum

# ========= Auth =========
EMBEDDED_API_KEY = os.environ.get("GEMINI_API_KEY", "")

# ========= Model =========
MODEL = "gemini-2.5-flash"

SYS_CORE = """You are a companion robot for older adults.
- Keep responses short, polite, and use simple wording; one fact per turn.
- Always end with a brief question to keep the conversation going.
- Do not diagnose or prescribe."""

POLICY_CHAT = """
# Phase: CHAT (small talk)
- Do NOT introduce new questions about the configured health topics (sleep, meds, etc.).
- Keep it to small talk only.
- 1–2 sentences, end with a light small-talk question.
"""

# ========= Topics / registry (used for guard & optional extraction) =========
TOPIC_REGISTRY = {
    "sleep_hours": {"labels": ["sleep", "slept", "hours", "insomnia", "rest"]},
    "meal_morning": {"labels": ["meal", "breakfast", "ate", "eating", "skip breakfast"]},
    "med_taken": {"labels": ["medicine", "meds", "pill", "took meds", "missed meds", "dose"]},
    "pain": {"labels": ["pain", "ache", "hurt", "headache", "stomachache", "back pain"]},
    "dizzy": {"labels": ["dizzy", "lightheaded", "light-headed", "vertigo"]},
    "mood_1to5": {"labels": ["mood", "energy", "feeling", "scale"]},
}

def time_greeting():
    hh = time.localtime().tm_hour
    if 5 <= hh < 11:  return "Good morning. How has your morning been?"
    if 11 <= hh < 17: return "Hello! How’s your day going?"
    return "Good evening. How was your day?"

def normalize_en(s: str) -> str:
    if not s: return ""
    s = unicodedata.normalize("NFKC", s)
    return s.strip().lower()

END_PAT = re.compile(r"(?:\b(end|stop|finish|bye|goodbye)\b|\bsee you\b|\bthat'?s it\b)", re.I)
def is_end_intent_en(text: str) -> bool:
    return bool(END_PAT.search((text or "").strip().lower()))

# ========= LLM config builders =========
def build_guard_cfg(enabled_keys: list[str]) -> types.GenerateContentConfig:
    vocab = []
    for k in enabled_keys:
        labels = TOPIC_REGISTRY[k]["labels"]
        vocab.append(f"- {k}: " + " / ".join(labels))
    vocab_text = "\n".join(vocab) if vocab else "- (none)"
    sys_text = (
        "Decide whether the following English text mentions any of the TARGET topics, "
        "or asks a question about those topics. Return JSON only.\n"
        "TARGET topics:\n" + vocab_text + "\n"
        'Output: {"is_topic": true|false, "is_question": true|false, "reason": "<short reason>"}'
    )
    return types.GenerateContentConfig(
        system_instruction=sys_text, temperature=0.0, response_mime_type="application/json", max_output_tokens=100
    )

def build_cfg_for_chat(state_json: str, strict: bool=False) -> types.GenerateContentConfig:
    policy = POLICY_CHAT + ("\n- Do not include any vocabulary related to the TARGET topics." if strict else "")
    sys_text = f"{SYS_CORE}\n{policy}\n\n# Session state (for reference; do NOT echo it):\n{state_json}"
    return types.GenerateContentConfig(
        system_instruction=sys_text, temperature=0.25, max_output_tokens=200,
        thinking_config=types.ThinkingConfig(thinking_budget=0),
    )

def build_extract_general_cfg(enabled_keys: list[str]) -> types.GenerateContentConfig:
    # Optional extractor to prefill signals for CHECK handoff
    parts = []
    for k in enabled_keys:
        if k in ("sleep_hours", "mood_1to5"):
            parts.append(f"\"{k}\": number|null")
        else:
            parts.append(f"\"{k}\": true|false|null")
    schema = "{ " + ", ".join(parts) + " }"
    sys_text = f"Extract health info from the English utterance. Return JSON only with this schema:\n{schema}"
    return types.GenerateContentConfig(
        system_instruction=sys_text, temperature=0.0, response_mime_type="application/json", max_output_tokens=120
    )

# ========= LLM helpers =========
def guard_flags(g_client, text: str, GUARD_CFG):
    try:
        r = g_client.models.generate_content(
            model=MODEL, contents=[types.Content(role="user", parts=[types.Part.from_text(text=text)])], config=GUARD_CFG
        )
        data = json.loads(r.text or "{}")
        return bool(data.get("is_topic", False)), bool(data.get("is_question", False))
    except Exception:
        return (False, False)

def gen_chat_with_guard(g_client, history, user_text, state_capsule, GUARD_CFG):
    def generate_once(strict=False):
        cfg = build_cfg_for_chat(state_capsule, strict=strict)
        resp = g_client.models.generate_content(
            model=MODEL, contents=history+[types.Content(role="user", parts=[types.Part.from_text(text=user_text)])],
            config=cfg
        )
        return (resp.text or "").strip()

    cand = generate_once(False)
    tries = 0
    topic, quest = guard_flags(g_client, cand, GUARD_CFG)
    while (topic or quest) and tries < 2:
        cand = generate_once(True)
        tries += 1
        topic, quest = guard_flags(g_client, cand, GUARD_CFG)
    if topic or quest or not cand:
        cand = random.choice([
            "Thanks. How has your day been?",
            "Got it. What are you up to next?",
            "Nice. Anything you enjoyed recently?"
        ])
    return cand

def llm_extract_general(g_client, utterance: str, EXTRACT_GENERAL_CFG):
    try:
        utterance = normalize_en(utterance)
        r = g_client.models.generate_content(
            model=MODEL, contents=[types.Content(role="user", parts=[types.Part.from_text(text=utterance)])],
            config=EXTRACT_GENERAL_CFG
        )
        data = json.loads(r.text or "{}")
        return {k:v for k,v in data.items() if v is not None}
    except Exception:
        return {}

# ========= Main =========
def main():
    ap = argparse.ArgumentParser(description="CHAT phase only (text)")
    ap.add_argument("--api-key", default=None, help="Gemini API key (overrides env)")
    ap.add_argument("--chat-turns", type=int, default=5, help="# of small-talk turns before you manually switch to CHECK")
    ap.add_argument("--guard-keys", default="sleep_hours,meal_morning,med_taken",
                    help="Comma-separated keys for health guard (e.g., sleep_hours,meal_morning,med_taken | all)")
    ap.add_argument("--extract-keys", default="sleep_hours,meal_morning,med_taken",
                    help="Keys to try extracting during CHAT for handoff JSON (same choices as guard-keys)")
    args = ap.parse_args()

    # enabled keys
    all_keys = list(TOPIC_REGISTRY.keys())
    def parse_keys(s):
        return all_keys if s.strip().lower()=="all" else [k.strip() for k in s.split(",") if k.strip() in TOPIC_REGISTRY]

    guard_keys = parse_keys(args.guard_keys)
    extract_keys = parse_keys(args.extract_keys)

    GUARD_CFG = build_guard_cfg(guard_keys)
    EXTRACT_GENERAL_CFG = build_extract_general_cfg(extract_keys) if extract_keys else None

    api_key = args.api_key or EMBEDDED_API_KEY
    if not api_key:
        print("Gemini API key not found. Pass --api-key or set GEMINI_API_KEY."); sys.exit(1)
    g_client = genai.Client(api_key=api_key)

    history: list[types.Content] = []
    signals = {k: None for k in extract_keys}

    first = time_greeting()
    print("\nRobot> " + first + "\n")
    history.append(types.Content(role="model", parts=[types.Part.from_text(text=first)]))
    turns = 1

    try:
        while turns <= max(1, int(args.chat_turns)):
            u = input("You> ").strip()
            if u.lower() in ("q","quit","exit"): print("Ending. Bye!"); break
            if not u: continue
            if is_end_intent_en(u):
                print("(That’s all for today. Thanks for chatting!)"); break

            # optional extraction for handoff
            if EXTRACT_GENERAL_CFG:
                ex = llm_extract_general(g_client, u, EXTRACT_GENERAL_CFG)
                for k,v in (ex or {}).items():
                    if k in signals and v is not None and signals[k] is None:
                        signals[k] = v

            state_capsule = f"<STATE>\nphase: CHAT\nturn: {turns}\nsignals: {json.dumps(signals)}\n</STATE>"
            try:
                text = gen_chat_with_guard(g_client, history, u, state_capsule, GUARD_CFG)
            except errors.APIError as e:
                print("API error:", e.message); text = "Thanks."
            if not text: text = "Thanks."
            if len(text) > 400: text = text[:380] + "... (let me know if you want more)"
            print("\nRobot> " + text + "\n")

            history.append(types.Content(role="user", parts=[types.Part.from_text(text=u)]))
            history.append(types.Content(role="model", parts=[types.Part.from_text(text=text)]))
            if len(history) > 24: history[:] = history[-24:]
            turns += 1

        # handoff JSON for CHECK
        if signals:
            handoff = {k:v for k,v in signals.items() if v is not None}
            print(f"(Handoff JSON) {json.dumps(handoff)}")

    except KeyboardInterrupt:
        print("\nEnd")

if __name__ == "__main__":
    main()
