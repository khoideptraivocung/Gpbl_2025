"""
chat_phase_voice_minimal_demo.py

Voice I/O wrapper for SmallTalkAgent (chat_phase_comp.py) with a demo-friendly log mode.
- Cloud-only STT/TTS (Google Cloud); pass service-account JSON path.
- Two input modes:
    1) Auto VAD (simple RMS, start on voice, end on silence)
    2) Push-to-Talk *hold* (Windows only; record while SPACE is held)
- Log modes:
    - FULL (default): shows [Listening]/[Recording]/[Processing STT]/[Speaking]
    - CLEAN (--demo_clean): prints only "Bot:" and "You:" lines (for demos)

Install deps:
  pip install google-generativeai google-cloud-speech google-cloud-texttospeech sounddevice numpy

Run examples:
  python chat_phase_voice_minimal_demo.py --model gemini-1.5-flash --gcp_creds "C:\\path\\to\\svc.json"
  python chat_phase_voice_minimal_demo.py --model gemini-1.5-flash --gcp_creds "C:\\path\\to\\svc.json" --ptt_hold --demo_clean
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import queue
from dataclasses import dataclass

import numpy as np
import sounddevice as sd

# Google Cloud clients (non-streaming v1 APIs)
try:
    from google.cloud import speech as gspeech
    from google.cloud import texttospeech as gtts
except Exception:
    gspeech = None
    gtts = None

# Import the text agent
from chat_phase_comp import SmallTalkAgent, ConvState, opening_greeting_for_bucket

@dataclass
class AudioConfig:
    rate: int = 16000
    channels: int = 1
    block_ms: int = 30   # 10/20/30ms typical
    silence_ms: int = 800  # Auto VAD: end when this much silence passes


class Recorder:
    def __init__(self, cfg: AudioConfig, show_ui: bool = True):
        self.cfg = cfg
        self.show_ui = show_ui
        self.q: "queue.Queue[np.ndarray]" = queue.Queue()

    def _callback(self, indata, frames, time_info, status):  # sd.InputStream
        if status and self.show_ui:
            print(status, file=sys.stderr)
        # mono float32 [-1,1] -> int16 PCM
        pcm = np.clip(indata[:, 0], -1.0, 1.0)
        self.q.put((pcm * 32767.0).astype(np.int16))

    @staticmethod
    def _is_voiced(chunk: np.ndarray, thresh: float = 0.01) -> bool:
        # Simple RMS VAD (quiet rooms)
        rms = float(np.sqrt(np.mean(chunk.astype(np.float32) ** 2)))
        return rms > thresh

    def record_autovad(self) -> bytes:
        cfg = self.cfg
        silence_frames_needed = max(1, int(cfg.silence_ms / cfg.block_ms))
        buffers: list[np.ndarray] = []
        recent_voice = 0
        if self.show_ui:
            print("[Listening…]")
        with sd.InputStream(samplerate=cfg.rate, channels=cfg.channels, dtype='float32', callback=self._callback):
            # wait for speech
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
            # capture until silence
            while True:
                ch = self.q.get()
                buffers.append(ch)
                if self._is_voiced(ch):
                    silent_frames = 0
                else:
                    silent_frames += 1
                if silent_frames >= silence_frames_needed:
                    break
        pcm16 = np.concatenate(buffers)
        return pcm16.tobytes()

    def record_ptt_hold(self) -> bytes:
        """PTT (record while SPACE is held). Windows only."""
        if os.name != 'nt':
            if self.show_ui:
                print("[warn] --ptt_hold is Windows-only. Falling back to Auto VAD.")
            return self.record_autovad()
        import ctypes
        user32 = ctypes.windll.user32
        VK_SPACE = 0x20
        def space_down() -> bool:
            return bool(user32.GetAsyncKeyState(VK_SPACE) & 0x8000)

        cfg = self.cfg
        buffers: list[np.ndarray] = []
        if self.show_ui:
            print("[Listening…] Hold SPACE to talk…")
        with sd.InputStream(samplerate=cfg.rate, channels=cfg.channels, dtype='float32', callback=self._callback):
            # wait for key down
            while not space_down():
                time.sleep(0.01)
            if self.show_ui:
                print("[Recording]")
            # collect while held
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemini-1.5-flash")
    ap.add_argument("--api_key", default=None)
    ap.add_argument("--gcp_creds", default=None, help="Path to GCP service-account JSON")
    ap.add_argument("--lang", default="en-US")
    ap.add_argument("--ptt_hold", action="store_true", help="Push-to-talk (record while SPACE is held) [Windows only]")
    ap.add_argument("--demo_clean", action="store_true", help="Hide status lines; print only 'Bot:' and 'You:'")
    args = ap.parse_args()

    # Accept GCP creds as a path
    if args.gcp_creds:
        if not os.path.isfile(args.gcp_creds):
            print(f"[error] JSON not found: {args.gcp_creds}")
            sys.exit(1)
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.abspath(args.gcp_creds)

    # Text agent
    agent = SmallTalkAgent(model_name=args.model, api_key=args.api_key, debug=False)
    state = ConvState()

    # UI mode
    show_ui = not args.demo_clean

    # Audio I/O
    cfg = AudioConfig()
    rec = Recorder(cfg, show_ui=show_ui)
    stt = CloudSTT(lang=args.lang, show_ui=show_ui)
    tts = CloudTTS(lang=args.lang, show_ui=show_ui)

    if show_ui:
        print("Voice small-talk ready. Ctrl+C to exit.\n")

    # Opening greeting (spoken)
    greet = opening_greeting_for_bucket(state.time_bucket)
    print(f"Bot: {greet}")
    tts.speak(greet)
    state.question_streak = 1  # opening counts as a question

    try:
        while True:
            if args.ptt_hold and os.name == 'nt':
                if show_ui:
                    print("[PTT] Hold SPACE to talk. Release to finish.")
                audio = rec.record_ptt_hold()
            else:
                if show_ui:
                    print("[AutoVAD] Speak when ready…")
                audio = rec.record_autovad()

            text = stt.transcribe(audio, cfg.rate)
            if text:
                print(f"You: {text}")
            else:
                # If nothing recognized, continue silently in CLEAN mode
                if show_ui:
                    print("You: ")
                continue

            if text.strip().lower() in {"exit", "quit"}:
                farewell = "Take care. I’ll be here when you need me."
                print(f"Bot: {farewell}")
                tts.speak(farewell)
                break

            bot = agent.reply(text, state)
            print(f"Bot: {bot}")
            tts.speak(bot)

            if state.closing_suggested:
                break

    except KeyboardInterrupt:
        if show_ui:
            print("\n[ctrl+c] bye")


if __name__ == "__main__":
    main()