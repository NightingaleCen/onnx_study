"""Stage 2: B -> C  (QDQ static quantization, INT8 activations + INT8 weights).

Quantizes every .onnx in the input stage dir separately:
  * encoder_model.onnx  -> calibrated with audio from data/calibration/*.wav
  * decoder_model.onnx  -> calibrated with precomputed A-encoder outputs on those
                           wavs (cross-attention representative) + typical
                           bos/padded input_ids.

To reproduce the outline's "INT8 of the un-simplified model" groups, run with
``--in A --out C_raw`` (quantize the raw export directly).

    uv run python scripts/stage2_quantize.py --model STT --in B --out C
    uv run python scripts/stage2_quantize.py --model STT --in A --out C_raw
"""
from __future__ import annotations

import argparse, sys, glob
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (stage_dir, list_onnx, write_manifest, upsert_variant,  # noqa: E402
                    preprocess_audio, list_onnx as _lo, banner)

import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import (quantize_static, QuantFormat, QuantType,
                                      CalibrationMethod)


class _Reader:
    """Minimal CalibrationDataReader: yields dicts of inputs once, cycle on demand."""
    def __init__(self, input_names, batches):
        self.input_names = input_names
        self.batches = batches  # list of list of arrays (aligned to input_names order)
        self.i = 0
    def get_next(self):
        if self.i >= len(self.batches):
            return None
        b = self.batches[self.i]; self.i += 1
        return {n: a for n, a in zip(self.input_names, b)}
    def rewind_initial(self):
        self.i = 0


def calib_wavs():
    return sorted(glob.glob("data/calibration/*.wav"))


def encoder_batches(wavs):
    return [[preprocess_audio(Path(w))[None, :].astype(np.float32)] for w in wavs]


def decoder_batches(wavs, a_encoder_path, dec_fix_len, bos, pad):
    es = ort.SessionOptions(); es.enable_mem_pattern = False
    enc = ort.InferenceSession(str(a_encoder_path), sess_options=es)
    enc_outs = []
    for w in wavs:
        iv = preprocess_audio(Path(w))[None, :].astype(np.float32)
        enc_outs.append(enc.run(None, {"input_values": iv})[0])
    rng = np.random.default_rng(7)
    id_patterns = [
        [bos] + [pad] * (dec_fix_len - 1),
        [bos, 30672, 29871, 232] + [pad] * (dec_fix_len - 4),
        [bos] + [int(x) for x in rng.integers(3, 32768, dec_fix_len - 1)],
    ]
    batches = []
    for eo in enc_outs:
        for ids in id_patterns:
            batches.append([np.asarray([ids], dtype=np.int64), eo])
    return batches


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="STT")
    ap.add_argument("--in", dest="inp", default="B")
    ap.add_argument("--out", dest="out", default="C")
    ap.add_argument("--per-channel", action="store_true", default=False,
                    help="per-channel weight quantization (better acc, may trip some EP kernels)")
    args = ap.parse_args()
    in_dir = stage_dir(args.model, args.inp)
    out_dir = stage_dir(args.model, args.out)
    wavs = calib_wavs()
    assert wavs, "no data/calibration/*.wav -- run prepare_data.py (drop audio in data/raw/)"
    meta = __import__("common").load_gen_meta(args.model)
    a_enc = stage_dir(args.model, "A") / "encoder_model.onnx"
    banner(f"quantize_static  {in_dir} -> {out_dir}  ({len(wavs)} calib wavs, QDQ S8S8)")

    for p in list_onnx(in_dir):
        model = onnx.load(str(p)) if False else None  # noqa
        import onnx
        names = [i.name for i in onnx.load(str(p)).graph.input]
        if "input_values" in names:
            batches = encoder_batches(wavs); kind = "encoder"
        elif "input_ids" in names:
            batches = decoder_batches(wavs, a_enc, meta["dec_fix_len"],
                                      meta["bos"], meta["pad"]); kind = "decoder"
        else:
            print(f"  {p.name}: unknown inputs {names}, skipping"); continue
        reader = _Reader(names, batches)
        op = out_dir / p.name
        quantize_static(str(p), str(op), reader,
                        quant_format=QuantFormat.QDQ,
                        activation_type=QuantType.QInt8,
                        weight_type=QuantType.QInt8,
                        calibrate_method=CalibrationMethod.MinMax,
                        per_channel=args.per_channel)
        onnx.checker.check_model(str(op))
        print(f"  {p.name} [{kind}]: quantized -> {op.stat().st_size//1024}KB")
    write_manifest(args.model, args.out, created_by="stage2_quantize.py",
                   args={"from": args.inp, "quant_format": "QDQ", "activation": "int8",
                         "weight": "int8", "per_channel": args.per_channel,
                         "calib_wavs": len(wavs)},
                   extra={"dtype": "int8"})
    upsert_variant(args.model, args.out, path=args.out, dtype="int8",
                   note=f"QDQ static S8S8 from {args.inp}", created_by="stage2_quantize.py")
    print("done.")


if __name__ == "__main__":
    main()