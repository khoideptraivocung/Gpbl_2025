# -*- coding: utf-8 -*-
"""
CHECK phase only (text I/O)
- Asks configured items once, with at most one clarification
- Can accept prefilled signals from CHAT via --prefill-json '{"sleep_hours":6,"meal_morning":true}'
- Prints a compact JSON summary at the end

Deps: pip install google-genai
"""

from google import genai
from google.genai import types, errors

import argparse, os, sys, time, json, random, re, unicodedata

# ========= Auth =========
EMBEDDED_API_KEY = os.environ.get("GEMINI_API_KEY", "")

# ========= Model =========
MODEL = "gemini-2.5-flash"

SYS_CORE = """You are a companion robot for older adults.
- Keep responses short, polite, and simple; one fact per turn.
- Do not diagnose or prescribe. Encourage contacting family or seeing a clinician if concerning answers appear.
"""

POLICY_CHECK = """
# Phase: CHECK (health check)
- Ask the fixed question once, then at most one clarification if needed.
- Do not add extra questions. Keep responses brief and supportive.
"""

# ========= Registry =========
TOPIC_REGISTRY = {
    "sleep_hours": {"id":"sleep","prompt":"About how many hours did you sleep last night?","type":"number"},
    "meal_morning": {"id":"meal","prompt":"Did you have breakfast today?","type":"bool"},
    "med_taken": {"id":"med","prompt":"Did you take your medicine today?","type":"bool"},
    "pain": {"id":"pain","prompt":"Are you having any pain?","type":"bool"},
    "dizzy": {"id":"dizzy","prompt":"Have you felt dizzy or lightheaded recently?","type":"bool"},
    "mood_1to5": {"id":"mood","prompt":"How is your energy or mood on a 1–5 scale?","type":"number"},
}

PRAISES = ["Thanks for sharing.", "That helps.", "Great, thank you.", "Got it, thanks."]

END_PAT = re.compile(r"(?:\b(end|stop|finish|bye|goodbye)\b|\bsee you\b|\bthat'?s it\b)", re.I)
def is_end_intent_en(text: str) -> bool:
    return bool(END_PAT.search((text or "").strip().lower()))

def normalize_en(s: str) -> str:
    import unicodedata
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
    if card_key == "med_taken":
        if re.search(r"\b(missed|forgot|skip+ed|didn'?t take|have(?: not)? taken)\b", a): return False
        if re.search(r"\b(took|have taken|did take|took my meds?|took my pill)\b", a): return True
    if card_key == "meal_morning":
        if re.search(r"\b(skip+ed (?:breakfast|meal)|didn'?t eat|no breakfast|no meal)\b", a): return False
        if re.search(r"\b(ate|had (?:breakfast|a meal)|i had breakfast)\b", a): return True
    return None

# ========= LLM CFGs =========
def build_extract_focused_cfg(key: str, card_prompt: str) -> types.GenerateContentConfig:
    examples = (
        "EXAMPLES:\n"
        'Q: Are you having any pain? / A: no → {"pain": false}\n'
        'Q: Have you felt dizzy recently? / A: not really → {"dizzy": false}\n'
        'Q: Did you take your medicine today? / A: yes → {"med_taken": true}\n'
        'Q: About how many hours did you sleep last night? / A: around 6 → {"sleep_hours": 6}\n'
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
    return types.GenerateContentConfig(
        system_instruction=sys_text, temperature=0.0,
        response_mime_type="application/json", max_output_tokens=120,
    )

def llm_extract_focused(g_client, card: dict, answer: str):
    try:
        cfg = build_extract_focused_cfg(card["key"], card["prompt"])
        r = g_client.models.generate_content(
            model=MODEL, contents=[types.Content(role="user", parts=[types.Part.from_text(text=answer)])], config=cfg
        )
        data = json.loads(r.text or "{}")
        # normalize
        if card["key"] == "sleep_hours" and isinstance(data.get("sleep_hours"), str):
            try: data["sleep_hours"] = float(re.sub(r"[^\d\.]", "", data["sleep_hours"]))
            except: data["sleep_hours"] = None
        if card["key"] == "mood_1to5" and isinstance(data.get("mood_1to5"), str):
            try: data["mood_1to5"] = int(re.sub(r"[^\d]", "", data["mood_1to5"]))
            except: data["mood_1to5"] = None
        if card["type"] == "bool" and data.get(card["key"]) is None:
            a = answer
            h = card_specific_bool(card["key"], a)
            if h is None: h = bool_heuristic(a)
            if h is not None: data = {card["key"]: h}
        return {k:v for k,v in data.items() if v is not None}
    except Exception:
        if card["type"] == "bool":
            a = answer
            h = card_specific_bool(card["key"], a)
            if h is None: h = bool_heuristic(a)
            if h is not None: return {card["key"]: h}
        # numeric fallback
        if card["type"] == "number":
            m = re.search(r"(\d+(?:\.\d+)?)", answer)
            if m:
                val = float(m.group(1))
                if card["key"] == "mood_1to5":
                    val = min(5, max(1, int(round(val))))
                return {card["key"]: val}
        return {}

def recommended_order_by_time(keys: list[str]) -> list[str]:
    hh = time.localtime().tm_hour
    base = ["sleep_hours","pain","dizzy","mood_1to5","med_taken","meal_morning"]
    if 5 <= hh < 11: base = ["meal_morning","sleep_hours","pain","dizzy","mood_1to5","med_taken"]
    if 11 <= hh < 17: base = ["sleep_hours","pain","dizzy","mood_1to5","med_taken","meal_morning"]
    return [k for k in base if k in keys]

def main():
    ap = argparse.ArgumentParser(description="CHECK phase only (text)")
    ap.add_argument("--api-key", default=None, help="Gemini API key (overrides env)")
    ap.add_argument("--check-keys", default="sleep_hours,meal_morning,med_taken",
                    help="Comma-separated keys to ask (e.g., sleep_hours,meal_morning,med_taken | all)")
    ap.add_argument("--prefill-json", default="", help="JSON string with prefilled signals from CHAT")
    args = ap.parse_args()

    all_keys = list(TOPIC_REGISTRY.keys())
    if args.check_keys.strip().lower() == "all":
        enabled_keys = all_keys
    else:
        enabled_keys = [k.strip() for k in args.check_keys.split(",") if k.strip() in TOPIC_REGISTRY]
        if not enabled_keys:
            print("No valid --check-keys provided. Choose from:", ", ".join(all_keys)); sys.exit(1)

    api_key = args.api_key or EMBEDDED_API_KEY
    if not api_key:
        print("Gemini API key not found. Pass --api-key or set GEMINI_API_KEY."); sys.exit(1)
    g_client = genai.Client(api_key=api_key)

    # signals, possibly prefilled
    signals = {k: None for k in enabled_keys}
    if args.prefill_json.strip():
        try:
            pre = json.loads(args.prefill_json)
            for k,v in (pre or {}).items():
                if k in signals and v is not None:
                    signals[k] = v
        except Exception as e:
            print(f"(WARN) Failed to parse --prefill-json: {e}")

    order = recommended_order_by_time(enabled_keys)
    cards = [{"key": k, "id": TOPIC_REGISTRY[k]["id"], "prompt": TOPIC_REGISTRY[k]["prompt"], "type": TOPIC_REGISTRY[k]["type"]} for k in order]

    # kickoff
    print("\nRobot> Let’s do a quick health check for today. Just a few questions, it’ll be quick.\n")

    idx = 0
    asked_once = False
    clarify_done = False
    awaiting = False

    # find first pending card
    while idx < len(cards) and signals.get(cards[idx]["key"]) is not None:
        idx += 1
    if idx >= len(cards):
        print(f"(Summary) {json.dumps(signals)}"); print("(That’s all for today. Thanks for chatting!)"); return

    # ask first
    print("Robot> " + random.choice(PRAISES) + " " + cards[idx]["prompt"] + "\n")
    awaiting = True; asked_once = True; clarify_done = False

    try:
        while True:
            u = input("You> ").strip()
            if u.lower() in ("q","quit","exit"):
                print("(That’s all for today. Thanks for chatting!)"); break
            if not u: continue
            if is_end_intent_en(u):
                print("(That’s all for today. Thanks for chatting!)"); break

            card = cards[idx]
            # extract
            ex = llm_extract_focused(g_client, card, u)
            for k,v in (ex or {}).items():
                if k in signals and v is not None:
                    signals[k] = v

            # advance logic
            if asked_once and awaiting:
                if signals.get(card["key"]) is None and not clarify_done:
                    # clarification
                    if card["type"] == "bool":
                        text = random.choice(PRAISES) + " Please answer in one or two words."
                    elif card["id"] == "sleep":
                        text = random.choice(PRAISES) + " A number please. About how many hours did you sleep?"
                    elif card["id"] == "mood":
                        text = random.choice(PRAISES) + " On a 1–5 scale, how is it?"
                    else:
                        text = random.choice(PRAISES) + " " + card["prompt"]
                    print("\nRobot> " + text + "\n")
                    clarify_done = True
                    continue
                elif signals.get(card["key"]) is None and clarify_done:
                    # give up this card; next
                    idx += 1; asked_once = False; clarify_done = False; awaiting = False
                else:
                    # got it; next
                    idx += 1; asked_once = False; clarify_done = False; awaiting = False

            # skip any prefilled cards
            while idx < len(cards) and signals.get(cards[idx]["key"]) is not None:
                idx += 1

            if idx >= len(cards):
                print(f"(Summary) {json.dumps(signals)}")
                print("(That’s all for today. Thanks for chatting!)")
                break

            # ask next card
            nxt = cards[idx]
            print("\nRobot> " + random.choice(PRAISES) + " " + nxt["prompt"] + "\n")
            asked_once = True; awaiting = True; clarify_done = False

    except KeyboardInterrupt:
        print("\nEnd")

if __name__ == "__main__":
    main()
