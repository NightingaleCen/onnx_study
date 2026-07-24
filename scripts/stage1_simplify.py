"""Stage 1a: A -> A_sim  (onnxsim simplification).

Runs onnxsim on every .onnx in the input stage dir. onnxsim is told to keep
dynamic input shapes (the encoder's num_samples / decoder's enc_len) symbolic.

    uv run python scripts/stage1_simplify.py --model STT --in A --out A_sim
"""
from __future__ import annotations

import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import stage_dir, list_onnx, write_manifest, upsert_variant, banner  # noqa: E402

import onnx
import onnxsim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="STT")
    ap.add_argument("--in", dest="inp", default="A")
    ap.add_argument("--out", dest="out", default="A_sim")
    args = ap.parse_args()
    in_dir = stage_dir(args.model, args.inp)
    out_dir = stage_dir(args.model, args.out)
    banner(f"onnxsim  {in_dir} -> {out_dir}")
    for p in list_onnx(in_dir):
        m = onnx.load(str(p))
        try:
            simp, ok = onnxsim.simplify(m)
        except Exception as e:
            print(f"  {p.name}: onnxsim failed ({e}); copying as-is")
            simp = m; ok = False
        onnx.checker.check_model(simp)
        op = out_dir / p.name
        onnx.save(simp, str(op))
        print(f"  {p.name}: nodes {len(m.graph.node)} -> {len(simp.graph.node)} | check={ok} | {op.stat().st_size//1024}KB")
    write_manifest(args.model, args.out, created_by="stage1_simplify.py",
                   args={"from": args.inp}, extra={"tool": "onnxsim", "dynamic_input_shape": True})
    upsert_variant(args.model, args.out, path=args.out, dtype="fp32",
                   note="onnxsim simplified", created_by="stage1_simplify.py")
    print("done.")


if __name__ == "__main__":
    main()