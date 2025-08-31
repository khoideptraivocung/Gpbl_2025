# -*- coding: utf-8 -*-
"""
CHAT phase only (text I/O) — google.generativeai 版
- Nターンの雑談
- 健康トピックに触れる出力を LLM でガード
- 任意で雑談中に拾えた信号を (Handoff JSON) として出す

必要:
  pip install google-generativeai
"""

import os, sys, time, json, random, re, argparse, unicodedata
import google.generativeai as genai

# ========= API KEY =========
GEMINI_API_KEY = ""
DEFAULT_MODEL = "gemini-2.0-flash"

def cfg_model(system_instruction: str, model_name: str):
    return genai.GenerativeModel(model_name=model_name, system_instruction=system_instruction)

def gen_config(temperature=0.25, max_tokens=200, json_mode=False):
    cfg = {"temperature": float(temperature), "max_output_tokens": int(max_tokens)}
    if json_mode:
        cfg["response_mime_type"] = "application/json"
    return cfg

# ========= Policies =========
SYS_CORE = """You are a companion robot for older adults.
- Keep responses short, polite, and use simple wording; one fact per turn.
- Always end with a brief question to keep the conversation going.
- Do not diagnose or prescribe."""

POLICY_CHAT = """
# Phase: CHAT (small talk)
- Do NOT introduce new questions about the configured health topics (sleep, meds, breakfast, pain, dizziness, mood).
- Keep it to small talk only.
- 1–2 sentences, end with a light small-talk question.
"""

TOPIC_REGISTRY = {
    "sleep_hours": {"labels": ["sleep", "slept", "hours", "insomnia", "rest"]},
    "meal_morning": {"labels": ["meal", "breakfast", "ate", "eating", "skip breakfast"]},
    "med_taken": {"labels": ["medicine", "meds", "pill", "took meds", "missed meds", "dose"]},
    "pain": {"labels": ["pain", "ache", "hurt", "headache", "stomachache", "back pain"]},
    "dizzy": {"labels": ["dizzy", "lightheaded", "light-headed", "vertigo"]},
    "mood_1to5": {"labels": ["mood", "energy", "feeling", "scale"]},
}

def normalize_en(s: str) -> str:
    if not s: return ""
    s = unicodedata.normalize("NFKC", s)
    return s.strip().lower()

END_PAT = re.compile(r"(?:\b(end|stop|finish|bye|goodbye)\b|\bsee you\b|\bthat'?s it\b)", re.I)
def is_end_intent_en(text: str) -> bool:
    return bool(END_PAT.search((text or "").strip().lower()))

# ========= Builders =========
def build_guard_model(enabled_keys, model_name):
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
    return cfg_model(sys_text, model_name)

def build_extract_model(enabled_keys, model_name):
    parts = []
    for k in enabled_keys:
        if k in ("sleep_hours", "mood_1to5"):
            parts.append(f"\"{k}\": number|null")
        else:
            parts.append(f"\"{k}\": true|false|null")
    schema = "{ " + ", ".join(parts) + " }"
    sys_text = f"Extract health info from the English utterance. Return JSON only with this schema:\n{schema}"
    return cfg_model(sys_text, model_name)

def build_chat_model(state_json: str, strict: bool, model_name: str):
    policy = POLICY_CHAT + ("\n- Do not include any vocabulary related to the TARGET topics." if strict else "")
    sys_text = f"{SYS_CORE}\n{policy}\n\n# Session state (for reference; do NOT echo it):\n{state_json}"
    return cfg_model(sys_text, model_name)

# ========= LLM helpers =========
def guard_flags(guard_model, text: str):
    try:
        resp = guard_model.generate_content(
            contents=[{"role":"user","parts":[text]}],
            generation_config=gen_config(temperature=0.0, max_tokens=100, json_mode=True),
        )
        data = json.loads(resp.text or "{}")
        return bool(data.get("is_topic", False)), bool(data.get("is_question", False))
    except Exception:
        return (False, False)

def llm_extract_general(extract_model, utterance: str):
    try:
        utterance = normalize_en(utterance)
        resp = extract_model.generate_content(
            contents=[{"role":"user","parts":[utterance]}],
            generation_config=gen_config(temperature=0.0, max_tokens=120, json_mode=True),
        )
        data = json.loads(resp.text or "{}")
        return {k:v for k,v in data.items() if v is not None}
    except Exception:
        return {}

def gen_chat_with_guard(model_name, history, user_text, state_capsule, guard_model):
    try:
        def generate_once(strict=False):
            chat_model = build_chat_model(state_capsule, strict, model_name)
            resp = chat_model.generate_content(
                contents=history + [{"role":"user","parts":[user_text]}],
                generation_config=gen_config(temperature=0.25, max_tokens=200),
            )
            return (resp.text or "").strip()

        cand = generate_once(False)
        tries = 0
        topic, quest = guard_flags(guard_model, cand)
        while (topic or quest) and tries < 2:
            cand = generate_once(True)
            tries += 1
            topic, quest = guard_flags(guard_model, cand)

        if topic or quest or not cand:
            return random.choice([
                "Thanks. How has your day been?",
                "Got it. What are you up to next?",
                "Nice. Anything you enjoyed recently?"
            ])
        return cand
    except Exception as e:
        print(f"(guard/generate error) {type(e).__name__}: {e}")
        return random.choice([
            "Thanks. How has your day been?",
            "Got it. What are you up to next?",
            "Nice. Anything you enjoyed recently?"
        ])

# ========= Greeting =========
def time_greeting():
    hh = time.localtime().tm_hour
    if 5 <= hh < 11:  return "Good morning. How has your morning been?"
    if 11 <= hh < 17: return "Hello! How’s your day going?"
    return "Good evening. How was your day?"

# ========= Main =========
def main():
    ap = argparse.ArgumentParser(description="CHAT phase only (text) — generativeai")
    ap.add_argument("--api-key", default=GEMINI_API_KEY, help="Gemini API key (defaults to embedded)")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="Model name (e.g., gemini-1.5-flash)")
    ap.add_argument("--chat-turns", type=int, default=5, help="# of small-talk turns")
    ap.add_argument("--guard-keys", default="sleep_hours,meal_morning,med_taken",
                    help="Comma-separated keys for health guard (or 'all')")
    ap.add_argument("--extract-keys", default="sleep_hours,meal_morning,med_taken",
                    help="Keys to extract during CHAT for handoff JSON (or 'all' / empty)")
    args = ap.parse_args()

    if not args.api_key:
        print("Gemini API key not found. Put it in GEMINI_API_KEY variable at top or pass --api-key.")
        sys.exit(1)
    genai.configure(api_key=args.api_key)

    all_keys = list(TOPIC_REGISTRY.keys())
    parse_keys = lambda s: (all_keys if s.strip().lower()=="all" else
                            [k.strip() for k in s.split(",") if k.strip() in TOPIC_REGISTRY])

    guard_keys = parse_keys(args.guard_keys)
    extract_keys = parse_keys(args.extract_keys) if args.extract_keys.strip() else []

    guard_model = build_guard_model(guard_keys, args.model)
    extract_model = build_extract_model(extract_keys, args.model) if extract_keys else None

    history = []
    signals = {k: None for k in extract_keys}

    first = time_greeting()
    print("\nRobot> " + first + "\n")
    history.append({"role":"model","parts":[first]})
    turns = 1

    try:
        while turns <= max(1, int(args.chat_turns)):
            u = input("You> ").strip()
            if u.lower() in ("q","quit","exit"): print("Ending. Bye!"); break
            if not u: continue
            if is_end_intent_en(u):
                print("(That’s all for today. Thanks for chatting!)"); break

            if extract_model:
                ex = llm_extract_general(extract_model, u)
                for k,v in (ex or {}).items():
                    if k in signals and v is not None and signals[k] is None:
                        signals[k] = v

            state_capsule = f"<STATE>\nphase: CHAT\nturn: {turns}\nsignals: {json.dumps(signals)}\n</STATE>"
            try:
                text = gen_chat_with_guard(args.model, history, u, state_capsule, guard_model)
            except Exception as e:
                print(f"(LLM error) {type(e).__name__}: {e}")
                text = random.choice([
                    "Got it. How has your day been?",
                    "I see. What are you up to next?",
                    "Nice. Anything you enjoyed recently?"
                ])

            if not text: text = "Thanks."
            if len(text) > 400: text = text[:380] + "... (let me know if you want more)"
            print("\nRobot> " + text + "\n")

            history.append({"role":"user","parts":[u]})
            history.append({"role":"model","parts":[text]})
            history[:] = history[-24:]
            turns += 1

        if signals:
            handoff = {k:v for k,v in signals.items() if v is not None}
            if handoff:
                print(f"(Handoff JSON) {json.dumps(handoff)}")

    except KeyboardInterrupt:
        print("\nEnd")

if __name__ == "__main__":
    main()