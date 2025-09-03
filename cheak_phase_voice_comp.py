# -*- coding: utf-8 -*-
"""
check_phase_voice_v3.py

Voice I/O wrapper for the CHECK phase (uses check_phase_text_v3.py logic).
- Cloud STT/TTS (Google Cloud)
- Same model/extraction & state-persistence behavior as text version
- PTT hold (Windows only) or Auto VAD
- Demo clean mode: print only "Bot:" / "You:" lines

Install:
  pip install google-generativeai google-cloud-speech google-cloud-texttospeech sounddevice numpy

Run examples:
  python check_phase_voice_v3.py --model gemini-2.0-flash --api-key AIza... --gcp_creds "C:\\path\\svc.json" --bucket morning
  python check_phase_voice_v3.py --api-key AIza... --gcp_creds "C:\\path\\svc.json" --ptt_hold --demo_clean

Notes:
- Requires check_phase_text_v3.py in the same directory.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import json
import queue
from typing import Dict, Any, List

import numpy as np
import sounddevice as sd

# Google Cloud clients
try:
    from google.cloud import speech as gspeech
    from google.cloud import texttospeech as gtts
except Exception:
    gspeech = None
    gtts = None

# LLM (Gemini)
import google.generativeai as genai

# Import text-phase helpers
import check_phase_comp as chk

# ============== Audio helpers ==============
class Recorder:
    def __init__(self, rate: int = 16000, channels: int = 1, block_ms: int = 30, silence_ms: int = 800, show_ui: bool = True):
        self.rate = rate
        self.channels = channels
        self.block_ms = block_ms
        self.silence_ms = silence_ms
        self.q: "queue.Queue[np.ndarray]" = queue.Queue()
        self.show_ui = show_ui

    def _callback(self, indata, frames, time_info, status):  # sd.InputStream
        if status and self.show_ui:
            print(status, file=sys.stderr)
        pcm = np.clip(indata[:, 0], -1.0, 1.0)
        self.q.put((pcm * 32767.0).astype(np.int16))

    @staticmethod
    def _is_voiced(chunk: np.ndarray, thresh: float = 0.01) -> bool:
        rms = float(np.sqrt(np.mean(chunk.astype(np.float32) ** 2)))
        return rms > thresh

    def record_autovad(self) -> bytes:
        silence_frames_needed = max(1, int(self.silence_ms / self.block_ms))
        buffers: List[np.ndarray] = []
        recent_voice = 0
        if self.show_ui:
            print("[Listening…]")
        with sd.InputStream(samplerate=self.rate, channels=self.channels, dtype='float32', callback=self._callback):
            while True:
                ch = self.q.get()
                if self._is_voiced(ch):
                    recent_voice += 1
                else:
                    recent_voice = max(0, recent_voice - 1)
                if recent_voice >= 3:
                    if self.show_ui:
                        print("[Recording]")
                    buffers.append(ch)
                    break
            silent_frames = 0
            while True:
                ch = self.q.get()
                buffers.append(ch)
                if self._is_voiced(ch):
                    silent_frames = 0
                else:
                    silent_frames += 1
                if silent_frames >= silence_frames_needed:
                    break
        if not buffers:
            return b""
        pcm16 = np.concatenate(buffers)
        return pcm16.tobytes()

    def record_ptt_hold(self) -> bytes:
        if os.name != 'nt':
            if self.show_ui:
                print("[warn] --ptt_hold is Windows-only. Falling back to Auto VAD.")
            return self.record_autovad()
        import ctypes
        user32 = ctypes.windll.user32
        VK_SPACE = 0x20
        def space_down() -> bool:
            return bool(user32.GetAsyncKeyState(VK_SPACE) & 0x8000)
        buffers: List[np.ndarray] = []
        if self.show_ui:
            print("[Listening…] Hold SPACE to talk…")
        with sd.InputStream(samplerate=self.rate, channels=self.channels, dtype='float32', callback=self._callback):
            while not space_down():
                time.sleep(0.01)
            if self.show_ui:
                print("[Recording]")
            while space_down():
                try:
                    ch = self.q.get(timeout=0.1)
                    buffers.append(ch)
                except queue.Empty:
                    pass
        if not buffers:
            return b""
        pcm16 = np.concatenate(buffers)
        return pcm16.tobytes()


class CloudSTT:
    def __init__(self, lang: str = "en-US", show_ui: bool = True):
        if gspeech is None:
            raise RuntimeError("google-cloud-speech not installed")
        self.lang = lang
        self.show_ui = show_ui
        self.client = gspeech.SpeechClient()

    def transcribe(self, audio_pcm16: bytes, rate: int) -> str:
        if self.show_ui:
            print("[Processing STT…]")
        config = gspeech.RecognitionConfig(
            encoding=gspeech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=rate,
            language_code=self.lang,
            enable_automatic_punctuation=True,
        )
        audio = gspeech.RecognitionAudio(content=audio_pcm16)
        try:
            resp = self.client.recognize(config=config, audio=audio)
            for r in resp.results:
                if r.alternatives:
                    return (r.alternatives[0].transcript or "").strip()
        except Exception as e:
            if self.show_ui:
                print(f"[STT error] {e}")
        return ""


class CloudTTS:
    def __init__(self, lang: str = "en-US", voice: str | None = None, show_ui: bool = True):
        if gtts is None:
            raise RuntimeError("google-cloud-texttospeech not installed")
        self.lang = lang
        self.voice = voice or "en-US-Neural2-A"
        self.show_ui = show_ui
        self.client = gtts.TextToSpeechClient()

    def speak(self, text: str):
        text = (text or "").strip()
        if not text:
            return
        if self.show_ui:
            print("[Speaking]")
        input_cfg = gtts.SynthesisInput(text=text)
        voice = gtts.VoiceSelectionParams(language_code=self.lang, name=self.voice)
        audio_cfg = gtts.AudioConfig(audio_encoding=gtts.AudioEncoding.LINEAR16, sample_rate_hertz=16000)
        try:
            resp = self.client.synthesize_speech(input=input_cfg, voice=voice, audio_config=audio_cfg)
            pcm = np.frombuffer(resp.audio_content, dtype=np.int16).astype(np.float32) / 32767.0
            sd.play(pcm, 16000)
            sd.wait()
        except Exception as e:
            if self.show_ui:
                print(f"[TTS error] {e}")


# ============== Flow ==============

def render_human_summary(signals: Dict[str, Any]) -> str:
    lines = []
    for k, v in signals.items():
        if v is None:
            continue
        if k == "sleep_hours":
            vv = float(v)
            vv = int(vv) if abs(vv - int(vv)) < 1e-6 else vv
            lines.append(f"sleep {vv}h")
        elif k.startswith("meal_"):
            meal = k.split("_")[1]
            lines.append(f"{meal} {'yes' if v else 'no'}")
        elif k.startswith("med_"):
            part = k.split("_")[1]
            lines.append(f"{part} meds {'yes' if v else 'no'}")
        elif k == "mood_1to5":
            lines.append(f"mood {int(v)}/5")
        elif k in ("pain", "dizzy"):
            lines.append(f"{k} {'yes' if v else 'no'}")
    return "Here’s today’s summary — " + "; ".join(lines) + "."


def persist_state(state_path: str, today_key: str, today_state: Dict[str, Any], signals: Dict[str, Any]):
    for kk, vv in signals.items():
        if vv is not None:
            today_state[kk] = vv
    try:
        os.makedirs(os.path.dirname(state_path), exist_ok=True)
        with open(state_path, "r", encoding="utf-8") as f:
            full = json.load(f) if os.path.getsize(state_path) > 0 else {}
    except Exception:
        full = {}
    full[today_key] = today_state
    try:
        with open(state_path, "w", encoding="utf-8") as f:
            json.dump(full, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"(WARN) Failed to save state: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-key", required=True, help="Gemini API key")
    ap.add_argument("--model", default=chk.DEFAULT_MODEL)
    ap.add_argument("--gcp_creds", default=None, help="Path to GCP service-account JSON")
    ap.add_argument("--lang", default="en-US")
    ap.add_argument("--bucket", default="auto", help="auto|morning|afternoon|evening")
    ap.add_argument("--state-file", default="", help="Path to JSON state store (persists by day)")
    ap.add_argument("--reset-today", action="store_true")
    ap.add_argument("--prefill-json", default="", help="Optional JSON prefill merged with saved state")
    ap.add_argument("--ptt_hold", action="store_true", help="Push-to-talk (hold SPACE) [Windows only]")
    ap.add_argument("--demo_clean", action="store_true", help="Only show 'Bot:' and 'You:' lines")
    args = ap.parse_args()

    # Configure Google Cloud creds
    if args.gcp_creds:
        if not os.path.isfile(args.gcp_creds):
            print(f"[error] JSON not found: {args.gcp_creds}")
            sys.exit(1)
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.abspath(args.gcp_creds)

    # Configure Gemini
    genai.configure(api_key=args.api_key)

    # UI mode
    show_ui = not args.demo_clean

    # Bucket
    bucket = args.bucket.lower().strip()
    if bucket == "auto":
        bucket = chk.time_bucket_from_local()

    # State persistence path (same default as text module)
    default_dir = r"C:\\Users\\kouki\\OneDrive\\Desktop\\gbbl_2025"
    state_path = args.state_file.strip() or os.path.join(default_dir, "health_state.json")

    # Load existing state
    try:
        os.makedirs(os.path.dirname(state_path), exist_ok=True)
        with open(state_path, "r", encoding="utf-8") as f:
            state = json.load(f) if os.path.getsize(state_path) > 0 else {}
    except Exception:
        state = {}

    today_key = time.strftime("%Y-%m-%d", time.localtime())
    today_state: Dict[str, Any] = state.get(today_key, {})

    if args.reset_today:
        today_state = {}

    prefill_cli: Dict[str, Any] = {}
    if args.prefill_json.strip():
        try:
            prefill_cli = json.loads(args.prefill_json)
        except Exception as e:
            if show_ui:
                print(f"(WARN) Failed to parse --prefill-json: {e}")
            prefill_cli = {}

    prefill: Dict[str, Any] = {**today_state, **prefill_cli}

    # Build question list & cards
    keys = chk.question_keys_for_bucket(bucket, prefill)
    cards = [
        {"key": k, "id": chk.TOPIC_REGISTRY[k]["id"], "prompt": chk.TOPIC_REGISTRY[k]["prompt"], "type": chk.TOPIC_REGISTRY[k]["type"]}
        for k in keys
    ]

    # Audio devices
    rec = Recorder(show_ui=show_ui)
    stt = CloudSTT(lang=args.lang, show_ui=show_ui)
    tts = CloudTTS(lang=args.lang, show_ui=show_ui)

    # Opening line
    if bucket == "morning":
        opening = "Good morning—starting the morning health check now. Just a few brief questions."
    elif bucket == "afternoon":
        opening = "Good afternoon—starting the afternoon health check now. Just a few brief questions."
    else:
        opening = "Good evening—starting the evening health check now. Just a few brief questions."

    print(f"Bot: {opening}")
    tts.speak(opening)

    # Build initial signals with prefill
    signals: Dict[str, Any] = {k: None for k in keys}
    for k in keys:
        if prefill.get(k) is not None:
            signals[k] = prefill[k]

    # Find first unanswered
    idx = 0
    while idx < len(cards) and signals.get(cards[idx]["key"]) is not None:
        idx += 1

    # If no questions remain -> summarize & exit
    if idx >= len(cards):
        summary = render_human_summary(signals)
        print(f"Bot: {summary}")
        print("(Summary) " + json.dumps(signals))
        tts.speak(summary)
        tts.speak("That’s all for now. Take care.")
        persist_state(state_path, today_key, today_state, signals)
        return

    # Ask first prompt
    print(f"Bot: {cards[idx]['prompt']}")
    tts.speak(cards[idx]["prompt"])

    clarify_done = False

    try:
        while True:
            # Record one utterance
            if args.ptt_hold and os.name == 'nt':
                if show_ui:
                    print("[PTT] Hold SPACE to talk. Release to finish.")
                audio = rec.record_ptt_hold()
            else:
                if show_ui:
                    print("[AutoVAD] Speak when ready…")
                audio = rec.record_autovad()

            text = stt.transcribe(audio, rec.rate)
            if text:
                print(f"You: {text}")
            else:
                if show_ui:
                    print("You: ")
                continue

            # End intent
            if chk.END_PAT.search(text) or text.strip().lower() in {"q","quit","exit"}:
                summary = render_human_summary(signals)
                if any(v is not None for v in signals.values()):
                    print(f"Bot: {summary}")
                    tts.speak(summary)
                print("(Summary) " + json.dumps(signals))
                tts.speak("That’s all for now. Take care.")
                persist_state(state_path, today_key, today_state, signals)
                break

            # Safety
            if chk.DANGER_PAT.search(text):
                print(f"Bot: {chk.SAFETY_TEMPLATE}")
                tts.speak(chk.SAFETY_TEMPLATE)
                persist_state(state_path, today_key, today_state, signals)
                break

            # Extract
            card = cards[idx]
            extractor = chk.build_extract_focused_model(card["key"], card["prompt"], args.model)
            ex = chk.llm_extract_focused(extractor, card, text)
            for k, v in (ex or {}).items():
                if k in signals and v is not None:
                    if k == "mood_1to5":
                        try:
                            v = int(round(float(v))); v = min(5, max(1, v))
                        except Exception:
                            pass
                    signals[k] = v

            key = card["key"]
            # Clarify on clear ambiguity only
            if (signals.get(key) is None) and (chk.AMBIG_PAT.match(text)) and (not clarify_done):
                hint = chk.CLARIFY_TEMPLATES[ card["type"] ]
                print(f"Bot: {hint}")
                tts.speak(hint)
                clarify_done = True
                continue

            # No clear value & not ambiguous -> one short restatement
            if signals.get(key) is None and not clarify_done:
                if card["type"] == "bool":
                    msg = "Please answer yes or no."
                elif card["id"] == "sleep":
                    msg = "A number please — about how many hours did you sleep?"
                elif card["id"] == "mood":
                    msg = "On a 1–5 scale, what number fits best?"
                else:
                    msg = card["prompt"]
                print(f"Bot: {msg}")
                tts.speak(msg)
                clarify_done = True
                continue

            # Advance
            idx += 1
            clarify_done = False
            while idx < len(cards) and signals.get(cards[idx]["key"]) is not None:
                idx += 1

            if idx >= len(cards):
                # Final summary
                summary = render_human_summary(signals)
                print(f"Bot: {summary}")
                print("(Summary) " + json.dumps(signals))
                tts.speak(summary)
                tts.speak("That’s all for now. Take care.")
                persist_state(state_path, today_key, today_state, signals)
                break

            # Ack + next question
            prev_key = key
            prev_val = signals.get(prev_key)
            ack = chk.make_ack(prev_key, chk.TOPIC_REGISTRY[prev_key]["type"], prev_val)
            nxt = cards[idx]["prompt"]
            say = (ack + " " + nxt).strip()
            print(f"Bot: {say}")
            tts.speak(say)

    except KeyboardInterrupt:
        persist_state(state_path, today_key, today_state, signals)
        if show_ui:
            print("\n[ctrl+c] bye")


if __name__ == "__main__":
    main()