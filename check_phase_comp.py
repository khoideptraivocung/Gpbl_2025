# -*- coding: utf-8 -*-
"""
CHECK phase (text I/O) — google.generativeai
- Time-bucketed health check (morning / afternoon / evening-night)
- Conversational confirmations (no robotic "Noted:") + next question in the same turn
- Clarify only when the answer is clearly ambiguous (e.g., "umm", empty)
- Danger keywords -> safety template and stop
- Morning also asks mood; noon may re-ask morning meds if unknown/false (making 6 Qs)
- Human-friendly summary first, then (Summary) JSON

Usage:
  pip install google-generativeai
  python check_phase_text_v3.py --model gemini-2.0-flash --api-key AIza... [--prefill-json '{...}'] [--bucket auto|morning|afternoon|evening]

Notes:
- English only.
- --prefill-json can carry prior signals (e.g., from morning run) to influence noon set.
"""
import os, sys, time, json, random, re, argparse, unicodedata
from typing import Dict, Any, List
import google.generativeai as genai

# ========= API KEY =========
GEMINI_API_KEY = ""  # or pass --api-key
DEFAULT_MODEL = "gemini-2.0-flash"

# ========= Helpers =========
def time_bucket_from_local() -> str:
    hh = time.localtime().tm_hour
    if 5 <= hh < 12: return "morning"
    if 12 <= hh < 17: return "afternoon"
    return "evening"

SAFETY_TEMPLATE = (
    "I’m concerned about your safety. I can’t assess emergencies or give medical advice. "
    "If you’re in danger or feel severe or sudden symptoms, contact local emergency services "
    "or a trusted person right now."
)

DANGER_PAT = re.compile(
    r"\b(chest pain|pressure in (?:my|the) chest|can'?t breathe|short(?:ness)? of breath|severe headache|"
    r"confusion|fainted|pass(?:ed)? out|unconscious|bleeding (?:a lot|heavily)|stroke|slurred speech|"
    r"suicidal|kill myself|want to die|jump off|overdose)\b",
    re.I,
)

END_PAT = re.compile(r"(?:\b(end|stop|finish|bye|goodbye)\b|\bsee you\b|\bthat'?s it\b)", re.I)

AMBIG_PAT = re.compile(r"^(?:\.*|…|uh+|um+|h+mm+|idk|not sure|\s*)$", re.I)

# ========= Model cfg =========
def cfg_model(system_instruction: str, model_name: str):
    return genai.GenerativeModel(model_name=model_name, system_instruction=system_instruction)

def gen_config(temperature=0.0, max_tokens=120, json_mode=False):
    cfg = {"temperature": float(temperature), "max_output_tokens": int(max_tokens)}
    if json_mode:
        cfg["response_mime_type"] = "application/json"
    return cfg

# ========= Registry =========
TOPIC_REGISTRY: Dict[str, Dict[str, Any]] = {
    # Numbers
    "sleep_hours": {"id":"sleep","prompt":"About how many hours did you sleep last night?","type":"number"},
    "mood_1to5": {"id":"mood","prompt":"On a 1–5 scale, how is your mood or energy right now?","type":"number"},
    # Booleans (time-specific)
    "med_morning": {"id":"med","prompt":"Did you take your morning medication today?","type":"bool"},
    "med_noon": {"id":"med","prompt":"Did you take your midday medication today?","type":"bool"},
    "med_evening": {"id":"med","prompt":"Did you take your evening medication today?","type":"bool"},
    "meal_breakfast": {"id":"meal","prompt":"Did you have breakfast today?","type":"bool"},
    "meal_lunch": {"id":"meal","prompt":"Did you have lunch today?","type":"bool"},
    "meal_dinner": {"id":"meal","prompt":"Did you have dinner today?","type":"bool"},
    "pain": {"id":"pain","prompt":"Are you having any pain?","type":"bool"},
    "dizzy": {"id":"dizzy","prompt":"Have you felt dizzy or lightheaded recently?","type":"bool"},
}

# ========= Heuristics =========
def normalize_en(s: str) -> str:
    if not s: return ""
    s = unicodedata.normalize("NFKC", s)
    return s.strip().lower()

NEG_PAT = re.compile(r"\b(?:no|nope|none|not really|don'?t|didn'?t|haven'?t|never|nothing|missed|skipped)\b", re.I)
POS_PAT = re.compile(r"\b(?:yes|yeah|yep|a bit|a little|some|hurts?|dizzy|light[- ]?headed|took|ate|had)\b", re.I)

def bool_heuristic(answer: str) -> bool | None:
    a = normalize_en(answer)
    if re.search(r"\bnot bad\b", a): return True
    if POS_PAT.search(a) and not re.search(r"\bnot\b", a): return True
    if NEG_PAT.search(a): return False
    return None

def card_specific_bool(card_key: str, answer: str) -> bool | None:
    a = normalize_en(answer)
    if card_key in {"med_morning","med_noon","med_evening"}:
        if re.search(r"\b(missed|forgot|skip+ed|didn'?t take|have(?: not)? taken)\b", a): return False
        if re.search(r"\b(took|have taken|did take|took my meds?|took my pill)\b", a): return True
    if card_key in {"meal_breakfast","meal_lunch","meal_dinner"}:
        if re.search(r"\b(skip+ed (?:breakfast|lunch|dinner|meal)|didn'?t eat|no (?:breakfast|lunch|dinner)|no meal)\b", a): return False
        if re.search(r"\b(ate|had (?:breakfast|lunch|dinner|a meal)|i had (?:breakfast|lunch|dinner))\b", a): return True
    return None

# ========= Extraction =========
def build_extract_focused_model(key: str, card_prompt: str, model_name: str):
    examples = (
        "EXAMPLES:\n"
        'Q: Are you having any pain? / A: no → {"pain": false}\n'
        'Q: Have you felt dizzy recently? / A: not really → {"dizzy": false}\n'
        'Q: Did you take your morning medication today? / A: yes → {"med_morning": true}\n'
        'Q: About how many hours did you sleep last night? / A: around 6 → {"sleep_hours": 6}\n'
        'Q: On a 1–5 scale, how is your mood? / A: maybe 4 → {"mood_1to5": 4}\n'
    )
    rules = (
        "RULES:\n"
        "- Negations (no, not really, didn’t, haven’t, missed, skipped) → false for booleans.\n"
        "- Affirmations (yes, a bit, some, hurts, dizzy, took, ate) → true for booleans.\n"
        "- If a number is present, return it (e.g., 6, 7.5) for numeric fields.\n"
        "- Prefer true/false over null when possible.\n"
    )
    sys_text = (
        f"From the following English Q/A, return ONLY the {key} value as JSON. "
        f'Use key name "{key}" exactly. Other keys must NOT be included.\n'
        f"{examples}{rules}\n- Question: {card_prompt}\n- Output example: {{\"{key}\": value}}\n"
    )
    return cfg_model(sys_text, model_name)


def llm_extract_focused(extractor_model, card: dict, answer: str) -> Dict[str, Any]:
    try:
        resp = extractor_model.generate_content(
            contents=[{"role":"user","parts":[answer]}],
            generation_config=gen_config(temperature=0.0, max_tokens=120, json_mode=True),
        )
        data = json.loads(resp.text or "{}")
        # normalize numeric strings
        if card["key"] == "sleep_hours" and isinstance(data.get("sleep_hours"), str):
            try: data["sleep_hours"] = float(re.sub(r"[^\d\.]", "", data["sleep_hours"]))
            except: data["sleep_hours"] = None
        if card["key"] == "mood_1to5" and isinstance(data.get("mood_1to5"), str):
            try: data["mood_1to5"] = int(re.sub(r"[^\d]", "", data["mood_1to5"]))
            except: data["mood_1to5"] = None
        # fallback to heuristics for booleans
        if card["type"] == "bool" and data.get(card["key"]) is None:
            h = card_specific_bool(card["key"], answer)
            if h is None: h = bool_heuristic(answer)
            if h is not None: data = {card["key"]: h}
        return {k:v for k,v in data.items() if v is not None}
    except Exception:
        # full heuristic fallback
        if card["type"] == "bool":
            h = card_specific_bool(card["key"], answer)
            if h is None: h = bool_heuristic(answer)
            if h is not None: return {card["key"]: h}
        if card["type"] == "number":
            m = re.search(r"(\d+(?:\.\d+)?)", answer)
            if m:
                val = float(m.group(1))
                if card["key"] == "mood_1to5":
                    val = min(5, max(1, int(round(val))))
                return {card["key"]: val}
        return {}

# ========= Question set per bucket =========
def question_keys_for_bucket(bucket: str, prefill: Dict[str, Any]) -> List[str]:
    b = bucket
    keys: List[str] = []
    if b == "morning":
        keys = ["sleep_hours","med_morning","meal_breakfast","pain","dizzy","mood_1to5"]
    elif b == "afternoon":
        # If we know med_morning was taken (True), ask only noon meds. If false/unknown, confirm morning first.
        if prefill.get("med_morning") is True:
            keys = ["med_noon","meal_lunch","pain","dizzy","mood_1to5"]
        else:
            keys = ["med_morning","med_noon","meal_lunch","pain","dizzy","mood_1to5"]
    else:  # evening/night
        keys = ["med_evening","meal_dinner","pain","dizzy","mood_1to5"]
    return keys

# ========= Conversational confirmations =========
ACK_NUM = [
    "Okay, {v} hours.",
    "Got it — {v} hours.",
    "Thanks — {v} hours then.",
]

ACK_BOOL = {
    # meals
    "meal_breakfast": {True: "Great — you had breakfast.", False: "Okay, no breakfast today."},
    "meal_lunch": {True: "Good — you had lunch.", False: "Okay, no lunch today."},
    "meal_dinner": {True: "Good — you had dinner.", False: "Okay, no dinner today."},
    # meds
    "med_morning": {True: "Good — you took your morning meds.", False: "Okay — morning meds not taken."},
    "med_noon": {True: "Thanks — midday meds taken.", False: "Okay — midday meds not taken."},
    "med_evening": {True: "Thanks — evening meds taken.", False: "Okay — evening meds not taken."},
    # symptoms
    "pain": {True: "I’m sorry you’re having pain.", False: "Okay — no pain."},
    "dizzy": {True: "Sorry to hear about the dizziness.", False: "Okay — no dizziness."},
}

ACK_MOOD = [
    "Alright — mood {v} out of 5.",
    "Noted — mood {v}/5.",
    "Okay — that’s a {v} out of 5.",
]

CLARIFY_TEMPLATES = {
    "number": "Just a number please — for example, 7.",
    "bool": "Please answer yes or no.",
}

def make_ack(card_key: str, card_type: str, value: Any) -> str:
    if value is None: return ""
    if card_key == "sleep_hours":
        v = float(value)
        v = int(v) if abs(v - int(v)) < 1e-6 else v
        return random.choice(ACK_NUM).format(v=v)
    if card_key == "mood_1to5":
        try:
            v = int(round(float(value)))
            v = min(5, max(1, v))
        except: v = value
        return random.choice(ACK_MOOD).format(v=v)
    if card_type == "bool":
        m = ACK_BOOL.get(card_key)
        if m and isinstance(value, bool):
            return m[value]
    return "Okay."

# ========= Summary rendering =========
def human_summary(lines: List[str]) -> str:
    return "Here’s today’s summary — " + "; ".join(lines) + "."

# ========= Main =========
def main():
    ap = argparse.ArgumentParser(description="CHECK phase (text) — generativeai")
    ap.add_argument("--api-key", default=GEMINI_API_KEY, help="Gemini API key (defaults to embedded)")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="Model name (e.g., gemini-2.0-flash)")
    ap.add_argument("--prefill-json", default="", help="JSON string with prefilled signals (e.g., from morning run)")
    ap.add_argument("--bucket", default="auto", help="auto|morning|afternoon|evening")
    args = ap.parse_args()

    if not args.api_key:
        print("Gemini API key not found. Put it at top or pass --api-key.")
        sys.exit(1)
    genai.configure(api_key=args.api_key)

    # Bucket
    bucket = args.bucket.lower().strip()
    if bucket == "auto":
        bucket = time_bucket_from_local()

    # Prefill
    prefill: Dict[str, Any] = {}
    if args.prefill_json.strip():
        try:
            prefill = json.loads(args.prefill_json)
        except Exception as e:
            print(f"(WARN) Failed to parse --prefill-json: {e}")
            prefill = {}

    # Build question list for this bucket
    keys = question_keys_for_bucket(bucket, prefill)
    cards = [
        {"key": k, "id": TOPIC_REGISTRY[k]["id"], "prompt": TOPIC_REGISTRY[k]["prompt"], "type": TOPIC_REGISTRY[k]["type"]}
        for k in keys
    ]

    # Opening line per bucket
    if bucket == "morning":
        opening = "Good morning—starting the morning health check now. Just a few brief questions."
    elif bucket == "afternoon":
        opening = "Good afternoon—starting the afternoon health check now. Just a few brief questions."
    else:
        opening = "Good evening—starting the evening health check now. Just a few brief questions."

    print(f"\nRobot> {opening}\n")

    signals: Dict[str, Any] = {k: None for k in keys}

    # Ask first unanswered
    idx = 0
    while idx < len(cards) and signals.get(cards[idx]["key"]) is not None:
        idx += 1
    if idx >= len(cards):
        # nothing to ask
        lines = []
        for k,v in signals.items():
            if v is None: continue
            if k == "sleep_hours":
                vv = float(v); vv = int(vv) if abs(vv-int(vv))<1e-6 else vv
                lines.append(f"sleep {vv}h")
            elif k.startswith("meal_"):
                meal = k.split("_")[1]
                lines.append(f"{meal} {'yes' if v else 'no'}")
            elif k.startswith("med_"):
                part = k.split("_")[1]
                lines.append(f"{part} meds {'yes' if v else 'no'}")
            elif k == "mood_1to5":
                lines.append(f"mood {int(v)}/5")
            elif k in ("pain","dizzy"):
                lines.append(f"{k} {'yes' if v else 'no'}")
        print("Robot> " + human_summary(lines))
        print("(Summary) " + json.dumps(signals))
        print("Robot> That’s all for now. Take care.")
        return

    print("Robot> " + cards[idx]["prompt"] + "\n")
    clarify_done = False

    try:
        while True:
            u = input("You> ").strip()
            if u.lower() in ("q","quit","exit") or END_PAT.search(u):
                # summarize whatever we have
                lines = []
                for k,v in signals.items():
                    if v is None: continue
                    if k == "sleep_hours":
                        vv = float(v); vv = int(vv) if abs(vv-int(vv))<1e-6 else vv
                        lines.append(f"sleep {vv}h")
                    elif k.startswith("meal_"):
                        meal = k.split("_")[1]
                        lines.append(f"{meal} {'yes' if v else 'no'}")
                    elif k.startswith("med_"):
                        part = k.split("_")[1]
                        lines.append(f"{part} meds {'yes' if v else 'no'}")
                    elif k == "mood_1to5":
                        lines.append(f"mood {int(v)}/5")
                    elif k in ("pain","dizzy"):
                        lines.append(f"{k} {'yes' if v else 'no'}")
                if lines:
                    print("Robot> " + human_summary(lines))
                print("(Summary) " + json.dumps(signals))
                print("Robot> That’s all for now. Take care.")
                break

            if DANGER_PAT.search(u):
                print("Robot> " + SAFETY_TEMPLATE)
                break

            # Extract
            extractor = build_extract_focused_model(cards[idx]["key"], cards[idx]["prompt"], args.model)
            ex = llm_extract_focused(extractor, cards[idx], u)
            for k,v in (ex or {}).items():
                if k in signals and v is not None:
                    if k == "mood_1to5":
                        try:
                            v = int(round(float(v))); v = min(5, max(1, v))
                        except: pass
                    signals[k] = v

            key = cards[idx]["key"]
            if (signals.get(key) is None) and (AMBIG_PAT.match(u)) and (not clarify_done):
                # Clarify once
                hint = CLARIFY_TEMPLATES[ cards[idx]["type"] ]
                print("\nRobot> " + hint + "\n")
                clarify_done = True
                continue

            # Move to next card
            if signals.get(key) is None and not clarify_done:
                # No clear value but not obviously ambiguous — ask a short restatement
                if cards[idx]["type"] == "bool":
                    print("\nRobot> Please answer yes or no.\n")
                elif cards[idx]["id"] == "sleep":
                    print("\nRobot> A number please — about how many hours did you sleep?\n")
                elif cards[idx]["id"] == "mood":
                    print("\nRobot> On a 1–5 scale, what number fits best?\n")
                clarify_done = True
                continue

            # We have a value or clarifications done — advance
            idx += 1
            clarify_done = False
            # skip already-filled (from prefill or previous answers)
            while idx < len(cards) and signals.get(cards[idx]["key"]) is not None:
                idx += 1

            if idx >= len(cards):
                # Final summary
                lines = []
                for k,v in signals.items():
                    if v is None: continue
                    if k == "sleep_hours":
                        vv = float(v); vv = int(vv) if abs(vv-int(vv))<1e-6 else vv
                        lines.append(f"sleep {vv}h")
                    elif k.startswith("meal_"):
                        meal = k.split("_")[1]
                        lines.append(f"{meal} {'yes' if v else 'no'}")
                    elif k.startswith("med_"):
                        part = k.split("_")[1]
                        lines.append(f"{part} meds {'yes' if v else 'no'}")
                    elif k == "mood_1to5":
                        lines.append(f"mood {int(v)}/5")
                    elif k in ("pain","dizzy"):
                        lines.append(f"{k} {'yes' if v else 'no'}")
                print("Robot> " + human_summary(lines))
                print("(Summary) " + json.dumps(signals))
                print("Robot> That’s all for now. Take care.")
                break

            # Ask next with conversational confirmation of the previous answer
            prev_key = key
            prev_val = signals.get(prev_key)
            ack = make_ack(prev_key, TOPIC_REGISTRY[prev_key]["type"], prev_val)
            nxt = cards[idx]["prompt"]
            print("\nRobot> " + ack + " " + nxt + "\n")

    except KeyboardInterrupt:
        print("\nEnd")

if __name__ == "__main__":
    main()