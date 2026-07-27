"""Stage 2: B -> C  (dynamic INT8 weight quantization).

Applies onnxruntime.quantization.quantize_dynamic with QInt8 weights.
Produces a QDQ model — no calibration data required; activation scales
are computed at inference time.

    uv run python scripts/pipeline/quantize.py --model STT               # B -> C
    uv run python scripts/pipeline/quantize.py --model STT --in A --out C_raw
"""
from __future__ import annotations

import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import stage_dir, list_onnx, write_manifest, upsert_variant, banner  # noqa: E402

import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="STT")
    ap.add_argument("--in", dest="inp", default="B")
    ap.add_argument("--out", dest="out", default="C")
    ap.add_argument("--per-channel", action="store_true", default=False,
                    help="per-channel weight quantization (may trip some EP kernels)")
    args = ap.parse_args()
    in_dir = stage_dir(args.model, args.inp)
    out_dir = stage_dir(args.model, args.out)
    banner(f"quantize_dynamic  {in_dir} -> {out_dir}  (QInt8 weights)")

    for p in list_onnx(in_dir):
        op = out_dir / p.name
        quantize_dynamic(str(p), str(op), weight_type=QuantType.QInt8,
                         per_channel=args.per_channel)
        onnx.checker.check_model(str(op))
        print(f"  {p.name}: quantized -> {op.stat().st_size//1024}KB")
    write_manifest(args.model, args.out, created_by="stage2_quantize.py",
                   args={"from": args.inp, "mode": "dynamic",
                         "weight": "int8", "per_channel": args.per_channel},
                   extra={"dtype": "int8"})
    upsert_variant(args.model, args.out, path=args.out, dtype="int8",
                   note=f"QDQ dynamic from {args.inp}", created_by="stage2_quantize.py")
    print("done.")


if __name__ == "__main__":
    main()