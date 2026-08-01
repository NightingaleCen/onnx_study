"""Stage 1b: A_sim -> B  (quant pre-processing).

Runs onnxruntime.quantization.shape_inference.quant_pre_process to prepare
the graph for quantization. Falls back to copying the input as-is on failure.

Symbolic shape inference is skipped (skip_symbolic_shape=True): it fails on
both models of this pipeline — the decoder (Expand broadcast of the fixed
dec_len=128 against symbolic enc_len) and the encoder (Range with symbolic
num_samples-derived length). Plain onnx C++ shape inference alone succeeds
and annotates the full value_info.

Model optimization is off by default (skip_optimization=True): ORT_ENABLE_BASIC
rewrites the graph structure, which changes which MatMuls quantize_dynamic
treats as "const B", measurably shifting int8 accuracy (cf. onnxsim path in
bench reports). B then stays structurally identical to its input, with only
value_info + pre-process metadata added. Pass --enable-optimize to opt in.

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
    ap.add_argument("--enable-optimize", action="store_true",
                    help="run ORT model optimization (graph rewrite) during pre-processing; "
                         "default off to keep the graph structure identical for quantize_dynamic")
    args = ap.parse_args()
    in_dir = stage_dir(args.model, args.inp)
    out_dir = stage_dir(args.model, args.out)
    banner(f"quant_pre_process  {in_dir} -> {out_dir}")
    for p in list_onnx(in_dir):
        op = out_dir / p.name
        try:
            quant_pre_process(str(p), str(op),
                              skip_symbolic_shape=True,
                              skip_optimization=not args.enable_optimize)
        except Exception as e:
            # best-effort: quant_pre_process can still fail on dynamic dims
            # (moonshine encoder num_samples / decoder enc_len). Fall back to
            # the input graph so the pipeline keeps flowing; quantize_dynamic
            # may still succeed on the copy.
            print(f"  {p.name}: quant_pre_process failed ({e}); copying input as-is to B")
            import shutil; shutil.copy(str(p), str(op))
        import onnx
        m = onnx.load(str(op)); onnx.checker.check_model(m)
        print(f"  {p.name}: nodes={len(m.graph.node)} -> {op}")
    write_manifest(args.model, args.out, created_by="stage1b_preopt.py",
                   args={"from": args.inp,
                         "skip_symbolic_shape": True,
                         "skip_optimization": not args.enable_optimize},
                   extra={"tool": "quant_pre_process"})
    upsert_variant(args.model, args.out, path=args.out, dtype="fp32",
                   note="quant_pre_process", created_by="stage1b_preopt.py")
    print("done.")


if __name__ == "__main__":
    main()