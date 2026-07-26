"""Stage 3: C -> D_xnn / D_cpu  (runtime QDQ fusion via optimized_model_filepath).

Creates an ORT InferenceSession at ORT_ENABLE_EXTENDED with the EP you request and
serializes the post-optimization graph (optimized_model_filepath). On the mac arm64
ORT wheel XNNPACK is NOT compiled in -> only D_cpu is produced there; the Pi's
aarch64 wheel includes XNNPACK -> D_xnn is produced there. bench.py records the EP
that actually ran.

    uv run python scripts/pipeline/runtime_opt.py --model STT --in C
"""
from __future__ import annotations

import argparse, sys, glob
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import (stage_dir, list_onnx, write_manifest, upsert_variant,  # noqa: E402
                    preprocess_audio, resolve_providers, banner)

import numpy as np
import onnxruntime as ort


def _feed(model_path, calib_wav, meta, enc_out=None):
    """Build a representative input dict for one run (triggers optimization dump)."""
    import onnx
    names = [i.name for i in onnx.load(str(model_path)).graph.input]
    if "input_values" in names:
        return {"input_values": preprocess_audio(Path(calib_wav))[None, :].astype(np.float32)}, None
    ids = [meta["bos"]] + [meta["pad"]] * (meta["dec_fix_len"] - 1)
    return ({"input_ids": np.asarray([ids], dtype=np.int64), "encoder_hidden_states": enc_out},
            None)


def dump_one(in_stage, out_stage, prov_key, model, calib_wav, meta):
    out_dir = stage_dir(model, out_stage)
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    so.enable_mem_pattern = False
    used, missing = resolve_providers(prov_key)
    if missing:
        print(f"  [{out_stage}] {prov_key} unavailable ({missing}); skipping")
        return False
    enc_out = None
    files = list_onnx(stage_dir(model, in_stage))
    # process encoder FIRST so its output feeds the decoder dump (sorted order
    # would put decoder_model before encoder_model alphabetically).
    files.sort(key=lambda p: 0 if p.name.startswith("encoder") else 1)
    for p in files:
        op = out_dir / p.name
        so.optimized_model_filepath = str(op)
        sess = ort.InferenceSession(str(p), so, providers=used)
        feed, _ = _feed(p, calib_wav, meta, enc_out)
        sess.run(None, feed)
        if "input_values" in feed:  # cache encoder output for the decoder dump
            enc_out = sess.run(None, feed)[0]
        print(f"  [{out_stage}/{prov_key}] {p.name} -> {op.stat().st_size//1024}KB (EP={used})")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="STT")
    ap.add_argument("--in", dest="inp", default="C")
    args = ap.parse_args()
    wavs = sorted(glob.glob("data/calibration/*.wav"))
    assert wavs, "need data/calibration/*.wav (run prepare_data.py)"
    meta = __import__("common").load_gen_meta(args.model)
    banner(f"stage3 runtime opt  {args.inp} -> D_xnn / D_cpu")
    made_xnn = dump_one(args.inp, "D_xnn", "xnnpack", args.model, wavs[0], meta)
    made_cpu = dump_one(args.inp, "D_cpu", "cpu", args.model, wavs[0], meta)
    for name, made in (("D_xnn", made_xnn), ("D_cpu", made_cpu)):
        if made:
            write_manifest(args.model, name, created_by="stage3_runtime_opt.py",
                           args={"from": args.inp, "opt_level": "ORT_ENABLE_EXTENDED", "ep": name},
                           extra={"dtype": "int8"})
            upsert_variant(args.model, name, path=name, dtype="int8",
                           note=f"runtime-optimized {args.inp} ({name})", created_by="stage3_runtime_opt.py")
    if not made_xnn:
        print("NOTE: XNNPACK not available on this build (mac arm64 ORT wheel). "
              "Run this same script on the Pi to produce D_xnn.")
    print("done.")


if __name__ == "__main__":
    main()