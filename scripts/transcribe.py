"""Interactive speech-to-text using any pipeline variant.

    # file mode
    uv run python scripts/transcribe.py --model STT --variant C_dyn --file audio.wav

    # mic mode (push-to-talk: Enter to start, Enter to stop, Ctrl+C to quit)
    uv run python scripts/transcribe.py --model STT --variant A --mic

Needs the ``dev`` group on machines that have a microphone (mac).
"""
from __future__ import annotations

import argparse, sys, threading
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (task_dir, load_variants, load_tokenizer, GreedyPipeline,
                    CANONICAL_SR, banner)  # noqa: E402

import numpy as np
import soundfile as sf


def transcribe_file(variant: str, path: str, model: str, max_tokens: int):
    data, sr = sf.read(path, dtype="float32")
    if len(data.shape) > 1:
        data = data.mean(axis=1)
    if sr != CANONICAL_SR:
        import soxr
        data = soxr.resample(data.astype(np.float32), sr, CANONICAL_SR)
    data = data.astype(np.float32)
    dur = len(data) / CANONICAL_SR
    print(f"loaded {path}: {dur:.1f}s, {CANONICAL_SR}Hz mono")
    _run_and_print(model, variant, data, max_tokens)


def transcribe_mic(model: str, variant: str, max_tokens: int):
    import sounddevice as sd

    class Recorder:
        def __init__(self):
            self.recording = False
            self.chunks: list[np.ndarray] = []

        def callback(self, indata, _frames, _time, _status):
            if self.recording:
                self.chunks.append(indata.copy())

    recorder = Recorder()
    stream = sd.InputStream(samplerate=CANONICAL_SR, channels=1,
                            dtype="float32", callback=recorder.callback)
    stream.start()

    def _clean():
        count = len(recorder.chunks)
        recorder.chunks.clear()
        return count

    print("Press Enter to record, Enter again to stop, Ctrl+C to quit\n")
    try:
        while True:
            input("> ")
            recorder.recording = True
            _clean()
            print("🔴 Recording... (press Enter to stop)")
            input("> ")
            recorder.recording = False
            stream.stop()
            if not recorder.chunks:
                print("(no audio captured)")
                stream.start()
                continue
            audio = np.concatenate(recorder.chunks, axis=0).flatten()
            dur = len(audio) / CANONICAL_SR
            print(f"recorded {dur:.1f}s, transcribing...")
            _run_and_print(model, variant, audio, max_tokens)
            stream.start()
    except KeyboardInterrupt:
        print("\ndone.")
    finally:
        recorder.recording = False
        stream.stop()
        stream.close()


def _run_and_print(model: str, variant: str, audio: np.ndarray, max_tokens: int):
    regs = load_variants(model)
    vpath = task_dir(model) / regs[variant]["path"]
    pipe = GreedyPipeline(model, vpath, opt_level="extended")
    decode = load_tokenizer(model)
    ids, tim = pipe.run(audio, max_tokens)
    text = decode(ids)
    print(f"  -> {text}")
    n_tok = tim.get("n_tokens", 0)
    print(f"  (enc {tim['enc_ms']:.0f}ms, dec {tim['dec_total_ms']:.0f}ms, "
          f"{n_tok} tokens, {tim['total_ms']:.0f}ms total)\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="STT")
    ap.add_argument("--variant", default="A")
    ap.add_argument("--file", default=None, help="path to an audio file")
    ap.add_argument("--mic", action="store_true", help="interactive microphone mode")
    ap.add_argument("--max-new-tokens", type=int, default=40)
    args = ap.parse_args()
    if not args.file and not args.mic:
        ap.error("specify --file <path> or --mic")
    if args.file:
        transcribe_file(args.variant, args.file, args.model, args.max_new_tokens)
    else:
        transcribe_mic(args.model, args.variant, args.max_new_tokens)


if __name__ == "__main__":
    main()