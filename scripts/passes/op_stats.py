"""Pass 1: operator statistics

CONTRACT
    run(model) -> dict
      model  : onnx.ModelProto (in memory) | str | pathlib.Path  (an .onnx file)
      return : {
          "model_path": str,
          "total_nodes": int,
          "ops": { op_type(str): count(int), ... },   # every node in graph.node
      }

PURPOSE (outline Pass 1)
    Run on the pre-optimized FP32 graph (variant B) to see the op distribution and
    predict where quantization Fusion / runtime bottlenecks will sit.
"""

from __future__ import annotations
from pathlib import Path
import argparse, sys
from tabulate import tabulate
from collections import Counter

import onnx
from tqdm import tqdm
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import stage_dir, list_onnx, MODELS_DIR

def run(model):
    if isinstance(model, (str, Path)):
        model = onnx.load(str(model))
    ops = Counter(n.op_type for n in model.graph.node)
    return {
        "model_path": str(model),
        "total_nodes": len(model.graph.node),
        "ops": dict(ops)
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="STT", help="Task type (STT, TTS)")
    ap.add_argument("--stage", default="all", help="Stages (A/B/C/D...) or 'all'")
    args = ap.parse_args()

    if args.stage == "all":
        stages = sorted({p.name for p in (MODELS_DIR / args.task).glob("*/")})  # all stages for this task
    else:
        stages = [args.stage]

    rows = []
    print(f"Analyzing {args.task}...")
    for stage in tqdm(stages):
        d = stage_dir(args.task, stage)
        onnx_files = sorted(list_onnx(d))
        if not onnx_files:
            print(f"no .onnx in {d}")
        
        for file in onnx_files:
            res = run(file)
            ops = res.get("ops", {})
            ops = ", ".join(f"{k}:{v}" for k, v in sorted(ops.items(), key=lambda x: -x[1]))
            rows.append(["_".join([stage, file.name]), res.get("total_nodes", "?"), ops])

    print(tabulate(rows, headers=["Stage", "File", "Total Nodes", "Ops"]))


if __name__ == "__main__":
    main()