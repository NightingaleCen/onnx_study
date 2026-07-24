"""Stage 0a: download HuggingFace model snapshot.

Run on BOTH machines independently (no model-sync between them):

    uv run python scripts/stage0_download.py --model STT

Drop your own calibration/eval dataset under ``data/raw/`` separately
(see ``scripts/prepare_data.py``). This script only fetches the model.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import hf_dir, banner  # noqa: E402

from huggingface_hub import snapshot_download  # base dep


def tree_listing(d: Path, prefix: str = "") -> list[tuple[str, int]]:
    rows: list[tuple[str, int]] = []
    for p in sorted(d.iterdir(), key=lambda x: (x.is_file(), x.name)):
        if p.is_dir():
            rows.append((f"{prefix}{p.name}/", -1))
            rows.extend(tree_listing(p, prefix + "  "))
        else:
            rows.append((f"{prefix}{p.name}", p.stat().st_size))
    return rows


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def show(d: Path, title: str) -> None:
    print(f"\n### {title}  ->  {d}")
    if not d.exists() or not any(d.rglob("*")):
        print("  (empty)")
        return
    rows = tree_listing(d)
    total = sum(s for _, s in rows if s >= 0)
    for name, s in rows:
        print(f"  {name:<48} {human(s) if s >= 0 else ''}")
    print(f"  {'TOTAL':<48} {human(total)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="STT", help="task name -> models/<model>/ folder")
    ap.add_argument("--repo", default="UsefulSensors/moonshine-tiny-zh")
    args = ap.parse_args()

    banner(f"Downloading HF snapshot: {args.repo}")
    out = snapshot_download(
        repo_id=args.repo, local_dir=hf_dir(args.model), revision="main",
    )
    print("done:", out)
    show(hf_dir(args.model), "HF snapshot (config/tokenizer/preprocessor/weights)")
    print("\nStage 0 download complete. Inspect the file list above to confirm "
          "which tokenizer/config/preprocessor files are shipped with the model.")


if __name__ == "__main__":
    main()