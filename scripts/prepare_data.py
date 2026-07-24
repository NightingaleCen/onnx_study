"""Prepare calibration + eval data for moonshine STT.

DATA OWNERSHIP: YOU choose the dataset. Two ways to supply audio:

  (1) Drop files into ``data/raw/`` (recursively; .wav .flac .mp3 .ogg supported) and run:
        uv run python scripts/prepare_data.py --model STT

  (2) Pull directly from a HuggingFace dataset (needs the `dev` group: `datasets`):
        uv run python scripts/prepare_data.py --model STT \
            --hf-dataset <repo> --hf-split <split> [--audio-column audio]

Contract of the produced files (consumed by stage2 calibration reader & bench.py):
  - 16 kHz, mono, 16-bit PCM WAV
  - filenames calib_00000.wav ... / eval_00000.wav ...
  - split is DETERMINISTIC (fixed seed) -> identical on every machine given the
    same inputs (so both your mac and the Pi regenerate byte-identical sets; no
    model/audio sync is ever needed across machines).

Note: Pi runs base deps only (no datasets/librosa). Re-run prepare_data on the Pi
with the same source files (copy data/raw/ over once) or re-issue the HF pull.

    uv run python scripts/prepare_data.py --model STT --calib-n 50 --eval-n 30
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA_DIR, CANONICAL_SR, banner  # noqa: E402

import numpy as np
import soundfile as sf
import soxr

RAW_DIR = DATA_DIR / "raw"
CAL_DIR = DATA_DIR / "calibration"
EVAL_DIR = DATA_DIR / "eval"
SUPPORTED = (".wav", ".flac", ".mp3", ".ogg")


def resample_mono(data: np.ndarray, sr: int) -> np.ndarray:
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != CANONICAL_SR:
        data = soxr.resample(data.astype(np.float32), sr, CANONICAL_SR)
    return data.astype(np.float32)


def load_file(path: Path) -> np.ndarray:
    data, sr = sf.read(str(path), dtype="float32")
    return resample_mono(data, sr)


def iter_manual(transcripts_path=None):
    tmap = {}
    if transcripts_path:
        p = Path(transcripts_path)
        if p.exists():
            with p.open() as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split("\t" if "\t" in line else ",")
                    if len(parts) >= 2:
                        tmap[parts[0].strip()] = parts[1].strip()
    files = sorted(p for p in RAW_DIR.rglob("*") if p.suffix.lower() in SUPPORTED and p.is_file())
    for p in files:
        try:
            text = tmap.get(p.stem)
            yield p.stem, load_file(p), text
        except Exception as e:
            yield p.stem, None, None
            print(f"  skip {p.name}: {e}")


def iter_hf(repo, split, audio_col, text_col=None):
    from datasets import load_dataset
    ds = load_dataset(repo, split=split)
    for i, row in enumerate(ds):
        a = row[audio_col]
        arr = resample_mono(np.asarray(a["array"], dtype=np.float32), a["sampling_rate"])
        text = row[text_col] if text_col else None
        yield f"{repo.split('/')[-1]}_{i:05d}", arr, text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="STT")
    ap.add_argument("--calib-n", type=int, default=50)
    ap.add_argument("--eval-n", type=int, default=30)
    ap.add_argument("--min-dur", type=float, default=1.0)
    ap.add_argument("--max-dur", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--hf-dataset", default=None, help="HuggingFace dataset repo id")
    ap.add_argument("--hf-split", default="train")
    ap.add_argument("--audio-column", default="audio")
    ap.add_argument("--text-column", default=None,
                    help="column name in the HF dataset carrying the transcript")
    ap.add_argument("--transcripts", default=None,
                    help="CSV/TSV mapping filename->reference text (for manual data/raw/ mode)")
    args = ap.parse_args()

    if args.hf_dataset:
        source = iter_hf(args.hf_dataset, args.hf_split, args.audio_column, args.text_column)
        banner(f"Loading HF dataset {args.hf_dataset}[{args.hf_split}]")
    else:
        source = iter_manual(args.transcripts)
        banner(f"Scanning {RAW_DIR} for raw audio")

    items = []  # (name, waveform, text_or_None)
    rejected = {"short": 0, "long": 0, "error": 0}
    for name, wav, text in source:
        if wav is None:
            rejected["error"] += 1
            continue
        dur = len(wav) / CANONICAL_SR
        if dur < args.min_dur:
            rejected["short"] += 1
            continue
        if dur > args.max_dur:
            rejected["long"] += 1
            continue
        items.append((name, wav, text))
    print(f"accepted={len(items)} rejected={rejected}")

    if not items:
        print("No audio available. Drop files in data/raw/ or pass --hf-dataset. Exiting.")
        return

    # deterministic disjoint split
    names = sorted(n for n, _, _ in items)
    order = {n: i for i, n in enumerate(names)}
    idx = list(range(len(items)))
    rng = np.random.default_rng(args.seed)
    rng.shuffle(idx)
    n_eval = min(args.eval_n, len(items))
    n_calib = min(args.calib_n, len(items) - n_eval)
    eval_idx = set(idx[:n_eval])
    calib_idx = set(idx[n_eval:n_eval + n_calib])

    CAL_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    by_name = {n: (w, t) for n, w, t in items}
    total_dur = 0.0
    for split_name, sel, outdir in (("calibration", calib_idx, CAL_DIR),
                                    ("eval", eval_idx, EVAL_DIR)):
        sel_names = sorted(names[i] for i in sel)
        dur_sum = 0.0; has_text = False
        text_lines = []
        for k, n in enumerate(sel_names):
            w, t = by_name[n]
            sf.write(str(outdir / f"{split_name[:4]}_{k:05d}.wav"), w, CANONICAL_SR, subtype="PCM_16")
            dur_sum += len(w) / CANONICAL_SR
            if t is not None:
                has_text = True
                text_lines.append((f"{split_name[:4]}_{k:05d}.wav", t))
        total_dur += dur_sum
        if has_text:
            with (outdir / "transcripts.csv").open("w") as f:
                for fname, txt in text_lines:
                    f.write(f"{fname}\t{txt}\n")
            print(f"  {split_name}: wrote {len(sel_names)} clips, {dur_sum:.1f}s + transcripts"
                  f"(requested {args.eval_n if split_name=='eval' else args.calib_n})")
        else:
            print(f"  {split_name}: wrote {len(sel_names)} clips, {dur_sum:.1f}s "
                  f"(requested {args.eval_n if split_name=='eval' else args.calib_n})")

    missing = (args.calib_n - n_calib) + (args.eval_n - n_eval)
    if missing > 0:
        print(f"  WARNING: {missing} fewer clips than requested. Add more to data/raw/ "
              "or enlarge the HF split; rerun (deterministic, idempotent).")
    banner(f"prepare_data done | {len(items)} clips, {total_dur:.1f}s total")


if __name__ == "__main__":
    main()