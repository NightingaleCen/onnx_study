"""Stage 1b: A' -> B  (quant pre-processing / symbolic shape inference).

Runs onnxruntime.quantization.shape_inference.quant_pre_process before static
quantization. It performs symbolic shape inference and folds/prepares the graph so
the QDQ quantizer has accurate tensor type info (critical for dynamic shapes).

    uv run python scripts/pipeline/preopt.py --model STT --in A_sim --out B
"""
from __future__ import annotations

import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import stage_dir, list_onnx, write_manifest, upsert_variant, banner  # noqa: E402

from onnxruntime.quantization.shape_inference import quant_pre_process


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="STT")
    ap.add_argument("--in", dest="inp", default="A_sim")
    ap.add_argument("--out", dest="out", default="B")
    args = ap.parse_args()
    in_dir = stage_dir(args.model, args.inp)
    out_dir = stage_dir(args.model, args.out)
    banner(f"quant_pre_process  {in_dir} -> {out_dir}")
    for p in list_onnx(in_dir):
        op = out_dir / p.name
        try:
            quant_pre_process(str(p), str(op))
        except Exception as e:
            # symbolic shape inference can fail on dynamic dims (moonshine encoder
            # num_samples / decoder enc_len). Fall back to the input graph so the
            # pipeline keeps flowing; quantize_static may still succeed on A_sim.
            print(f"  {p.name}: quant_pre_process failed ({e}); copying input as-is to B")
            import shutil; shutil.copy(str(p), str(op))
        import onnx
        m = onnx.load(str(op)); onnx.checker.check_model(m)
        print(f"  {p.name}: nodes={len(m.graph.node)} -> {op}")
    write_manifest(args.model, args.out, created_by="stage1b_preopt.py",
                   args={"from": args.inp}, extra={"tool": "quant_pre_process"})
    upsert_variant(args.model, args.out, path=args.out, dtype="fp32",
                   note="quant_pre_process", created_by="stage1b_preopt.py")
    print("done.")


if __name__ == "__main__":
    main()