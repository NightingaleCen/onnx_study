"""Fetch a small AISHELL-1 subset for moonshine STT calibration + eval.

Downloads 3 speaker archives (~1000 clips total), extracts wavs, parses
transcripts, and evenly samples 80 clips into data/raw/ + data/raw/transcripts.tsv.
Then run prepare_data.py to produce the deterministic calib/eval split.

    uv run python scripts/fetch_aishell.py
    uv run python scripts/prepare_data.py --model STT --calib-n 50 --eval-n 30 \
        --transcripts data/raw/transcripts.tsv
"""
from __future__ import annotations

import random
import sys
import tarfile
import tempfile
from io import StringIO
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA_DIR, CANONICAL_SR, banner  # noqa: E402

from huggingface_hub import hf_hub_download
import numpy as np
import soundfile as sf

REPO = "AISHELL/AISHELL-1"
SPEAKERS = ["S0002", "S0020", "S0040"]   # 3 speakers, ~360 clips each
TARGET = 80
RAW = DATA_DIR / "raw"
SEED = 1337


def _path_in_repo(speaker: str) -> str:
    return f"data_aishell/wav/{speaker}.tar.gz"


def _transcript_path() -> str:
    return "data_aishell/transcript/aishell_transcript_v0.8.txt"


def main():
    banner(f"Downloading {len(SPEAKERS)} speaker archives + transcript from {REPO}")
    RAW.mkdir(parents=True, exist_ok=True)

    # 1. download transcript (plain text, ~5MB)
    tmpl = hf_hub_download(REPO, _transcript_path(), repo_type="dataset")
    transcript_map: dict[str, str] = {}
    with open(tmpl) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # format: "BAC009S0002W0122 今天天气很好" (space between id and text)
            parts = line.split(None, 1)
            if len(parts) == 2:
                transcript_map[parts[0]] = parts[1]
    print(f"transcript map: {len(transcript_map)} entries")

    # 2. download 3 tar.gz archives & extract wavs
    rng = np.random.default_rng(SEED)
    clips: list[tuple[str, Path]] = []  # (stem, temp_wav_path)
    with tempfile.TemporaryDirectory() as tmp:
        for spk in SPEAKERS:
            arc = hf_hub_download(REPO, _path_in_repo(spk), repo_type="dataset")
            print(f"extracting {spk}.tar.gz ...")
            with tarfile.open(arc, "r:gz") as tf:
                tf.extractall(tmp)
            for wav_p in Path(tmp).rglob("*.wav"):
                try:
                    info = sf.info(str(wav_p))
                    dur = info.duration
                    if 1.0 <= dur <= 30.0:
                        clips.append((wav_p.stem, wav_p))
                except Exception:
                    pass
            print(f"  {spk}: accumulated {len(clips)} acceptable clips so far")

        print(f"total acceptable clips: {len(clips)}")

        if len(clips) < TARGET:
            print(f"only {len(clips)} clips found (need {TARGET}); aborting")
            return

        # 3. stratified even-sample: group by speaker prefix, spread evenly
        by_spk: dict[str, list[tuple[str, Path]]] = {}
        for stem, p in clips:
            parts = stem.split("S")
            spk_id = "S" + parts[-1][:4] if len(parts) > 1 else "unknown"
            by_spk.setdefault(spk_id, []).append((stem, p))

        n = min(TARGET, len(clips))
        per_spk = {spk: max(1, round(n * len(vv) / len(clips))) for spk, vv in by_spk.items()}
        while sum(per_spk.values()) > n:
            k = max(per_spk, key=per_spk.get)
            if per_spk[k] > 1:
                per_spk[k] -= 1
        while sum(per_spk.values()) < n:
            k = min(per_spk, key=per_spk.get)
            per_spk[k] += 1

        selected: dict[str, Path] = {}
        for spk, vv in sorted(by_spk.items()):
            rng.shuffle(vv)
            for stem, p in vv[: per_spk[spk]]:
                selected[stem] = p
        print(f"sampled {len(selected)} clips across {len(by_spk)} speakers")

        # 4. copy wavs to data/raw/ and write transcripts.tsv
        tsv_lines = []
        for stem in sorted(selected):
            src = selected[stem]
            dst = RAW / f"{stem}.wav"
            data, sr = sf.read(str(src), dtype="float32")
            if len(data.shape) > 1:
                data = data[:, 0] if data.shape[1] == 1 else data.mean(axis=1)
            if sr != CANONICAL_SR:
                import soxr
                data = soxr.resample(data, sr, CANONICAL_SR)
            sf.write(str(dst), data, CANONICAL_SR, subtype="PCM_16")
            text = transcript_map.get(stem, "")
            tsv_lines.append(f"{stem}.wav\t{text}")

        tsv_path = RAW / "transcripts.tsv"
        tsv_path.write_text("\n".join(tsv_lines) + "\n", encoding="utf-8")
        print(f"wrote {len(selected)} wavs + {tsv_path}")
        banner("fetch_aishell done — now run prepare_data.py")


if __name__ == "__main__":
    main()