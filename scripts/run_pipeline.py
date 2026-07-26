"""run_pipeline.py -- chain the stage scripts in order.

Convenience orchestrator: invokes each stage as `uv run python scripts/<script>`
with a shared --model. Stage 0-download and 0-export need the dev group (mac).
prepare_data.py is NOT included -- data sourcing is the user's job (see its docstring).

    uv run python scripts/run_pipeline.py --model STT --from 0-download --to 3-runtime
    uv run python scripts/run_pipeline.py --model STT --stages 1-simplify,1b-preopt,2-quantize
"""
from __future__ import annotations

import argparse, subprocess, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import banner  # noqa: E402

STAGES = [
    ("0-download",   "dataset/download_models.py",  []),
    ("0-export",     "pipeline/export_onnx.py",    []),
    ("1-simplify",   "pipeline/simplify.py",  ["--in", "A", "--out", "A_sim"]),
    ("1b-preopt",    "pipeline/preopt.py",   ["--in", "A_sim", "--out", "B"]),
    ("2-quantize",   "pipeline/quantize.py",  ["--in", "B", "--out", "C"]),
    ("2-quantize-raw", "pipeline/quantize.py", ["--in", "A", "--out", "C_raw"]),
    ("3-runtime",    "pipeline/runtime_opt.py", ["--in", "C"]),
]
ORDER = [s for s, _, _ in STAGES]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="STT")
    ap.add_argument("--from", dest="frm", default=ORDER[0])
    ap.add_argument("--to", default=ORDER[-1])
    ap.add_argument("--stages", default="", help="explicit comma list (overrides --from/--to)")
    args = ap.parse_args()
    if args.stages:
        sel = [s.strip() for s in args.stages.split(",") if s.strip()]
    else:
        i, j = ORDER.index(args.frm), ORDER.index(args.to)
        sel = ORDER[i:j + 1]
    for s in sel:
        script, extra = next((sc, ex) for nm, sc, ex in STAGES if nm == s)
        banner(f"STAGE {s}")
        cmd = [sys.executable, "scripts/" + script, "--model", args.model, *extra]
        print(" ".join(cmd))
        r = subprocess.run(cmd)
        if r.returncode != 0:
            print(f"stage {s} failed (exit {r.returncode}); aborting pipeline")
            sys.exit(r.returncode)
    print("pipeline complete.")


if __name__ == "__main__":
    main()