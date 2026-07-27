"""Stage 0: Export PyTorch moonshine model to ONNX (variant A).

Produces two ONNX files under models/<model>/A/:
  - encoder_model.onnx  (symbolic audio length)
  - decoder_model.onnx  (fixed token length, stateless — no KV cache)

Requires the ``dev`` dependency group (torch/transformers). Mac-only.

    uv run python scripts/pipeline/export_onnx.py --model STT
    uv run python scripts/pipeline/export_onnx.py --model STT --dec-fix-len 128
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import (  # noqa: E402
    hf_dir, stage_dir, task_dir, write_manifest, upsert_variant, banner,
    OPT_LEVELS,
)

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
from transformers import AutoProcessor, MoonshineForConditionalGeneration

OPSET = 18  # dynamo exporter min supported; outline requires opset>=17


class EncWrapper(nn.Module):
    def __init__(self, model: MoonshineForConditionalGeneration):
        super().__init__()
        self.encoder = model.model.encoder

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        out = self.encoder(input_values=input_values, attention_mask=None)
        return out.last_hidden_state


class DecWrapper(nn.Module):
    def __init__(self, model: MoonshineForConditionalGeneration):
        super().__init__()
        self.decoder = model.model.decoder
        self.proj_out = model.proj_out

    def forward(self, input_ids: torch.Tensor, encoder_hidden_states: torch.Tensor) -> torch.Tensor:
        out = self.decoder(
            input_ids=input_ids,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=None,
            attention_mask=None,
            position_ids=None,
            past_key_values=None,
            use_cache=False,
        )
        return self.proj_out(out.last_hidden_state)


def synth_waveform(duration_s=3.0, sr=16000, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(int(duration_s * sr)) / sr
    # mix of tones + noise -> speech-ish energy distribution
    wav = (0.3 * np.sin(2 * np.pi * 220 * t)
           + 0.2 * np.sin(2 * np.pi * 660 * t)
           + 0.1 * rng.standard_normal(len(t)))
    wav = wav.astype(np.float32)
    wav = wav / max(1.0, np.max(np.abs(wav)))
    return wav


def export(enc, dec, sample_values, sample_ids, sample_enc_hs, out_dir, opset):
    enc_path = out_dir / "encoder_model.onnx"
    dec_path = out_dir / "decoder_model.onnx"
    # dynamo=True: symbolic-shape-aware exporter.
    # encoder fully dynamic in num_samples (audio length). batch concrete=1.
    torch.onnx.export(
        enc, (sample_values,), str(enc_path),
        input_names=["input_values"], output_names=["last_hidden_state"],
        dynamic_axes={"input_values": {1: "num_samples"},
                      "last_hidden_state": {1: "enc_len"}},
        opset_version=opset, do_constant_folding=True, dynamo=True,
    )
    # decoder: dec_len is FIXED (the transformers causal-mask CumSum bakes the
    # traced dec_len into the graph under torch.export, so we keep it static and
    # pad input_ids to that fixed length at runtime; causal masking means padding
    # tokens never affect real positions). encoder seq (enc_len) stays dynamic.
    torch.onnx.export(
        dec, (sample_ids, sample_enc_hs), str(dec_path),
        input_names=["input_ids", "encoder_hidden_states"], output_names=["logits"],
        # empty-dict keys anchor the name mapping so 'enc_len' lands on the
        # correct tensor (dec_len stays concrete, enc_len stays symbolic).
        dynamic_axes={"input_ids": {}, "encoder_hidden_states": {1: "enc_len"}, "logits": {}},
        opset_version=opset, do_constant_folding=True, dynamo=True,
    )
    # flatten external-data (.onnx.data) into self-contained single-file protobufs
    # to simplify downstream onnxsim / quantize_static / netron (decoder 78MB < 2GB).
    _flatten(enc_path); _flatten(dec_path)
    return enc_path, dec_path


def _flatten(path: Path):
    m = onnx.load(str(path))
    onnx.save_model(m, str(path), save_as_external_data=False)
    data = Path(str(path) + ".data")
    if data.exists():
        data.unlink()


def ort_greedy(enc_path, dec_path, input_values_np, bos, eos, pad,
               max_new_tokens, dec_fix_len, opt_level="extended"):
    # dec_len is static in the exported graph; always feed input_ids of length
    # dec_fix_len (real tokens left-aligned, right-padded with `pad`).
    so = ort.SessionOptions()
    so.graph_optimization_level = OPT_LEVELS[opt_level]
    so.enable_mem_pattern = False
    enc_sess = ort.InferenceSession(str(enc_path), sess_options=so)
    dec_sess = ort.InferenceSession(str(dec_path), sess_options=so)
    enc_out = enc_sess.run(None, {"input_values": input_values_np[None, :]})[0]  # [1, T', 288]
    ids = [bos]
    for _ in range(max_new_tokens):
        padded = ids + [pad] * (dec_fix_len - len(ids))
        logits = dec_sess.run(None, {"input_ids": np.asarray([padded], dtype=np.int64),
                                     "encoder_hidden_states": enc_out})[0]
        nxt = int(np.argmax(logits[0, len(ids) - 1]))
        ids.append(nxt)
        if nxt == eos:
            break
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="STT")
    ap.add_argument("--opset", type=int, default=OPSET)
    ap.add_argument("--max-new-tokens", type=int, default=24)
    ap.add_argument("--dec-fix-len", type=int, default=128,
                    help="fixed decoder token length baked into graph + runtime padding target")
    ap.add_argument("--out", dest="out", default="A",
                    help="output variant dir under models/<model>/")
    args = ap.parse_args()
    assert args.max_new_tokens + 1 <= args.dec_fix_len, "dec-fix-len must exceed max-new-tokens"

    hf = hf_dir(args.model)
    banner(f"Loading transformers model from {hf}")
    processor = AutoProcessor.from_pretrained(str(hf))
    # eager attention avoids SDPA's `is_causal` SymBool problem under torch.export
    model = MoonshineForConditionalGeneration.from_pretrained(str(hf), attn_implementation="eager")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    cfg = model.config
    bos, eos, pad = cfg.bos_token_id, cfg.eos_token_id, cfg.pad_token_id

    enc, dec = EncWrapper(model), DecWrapper(model)
    enc.eval(); dec.eval()

    wv = synth_waveform(3.0)
    iv = torch.from_numpy(wv).unsqueeze(0)  # [1, T]
    N = args.dec_fix_len
    with torch.no_grad():
        ref_enc = enc(iv)               # [1, T', 288]
        ref_hs = ref_enc
        idsN = torch.tensor([[bos] * N])          # trace/eval input at fixed dec_len
        ref_logits = dec(idsN, ref_hs)            # [1, N, V]

    out_dir = stage_dir(args.model, args.out)
    banner(f"Exporting to {out_dir} (opset {args.opset}, dec_fix_len={N})")
    with torch.no_grad():
        enc_path, dec_path = export(enc, dec, iv, idsN, ref_hs, out_dir, args.opset)

    for p in (enc_path, dec_path):
        onnx.checker.check_model(str(p))

    # ORT numerical check vs torch (decoder at fixed dec_len=N)
    so = ort.SessionOptions(); so.enable_mem_pattern = False
    enc_sess = ort.InferenceSession(str(enc_path), sess_options=so)
    dec_sess = ort.InferenceSession(str(dec_path), sess_options=so)
    ort_enc = enc_sess.run(None, {"input_values": iv.numpy()})[0]
    ok_enc = np.allclose(ort_enc, ref_enc.numpy(), atol=1e-4)
    ort_logits = dec_sess.run(None, {"input_ids": idsN.numpy(),
                                     "encoder_hidden_states": ref_hs.numpy()})[0]
    ok_dec = np.allclose(ort_logits, ref_logits.numpy(), atol=1e-4)
    print(f"encoder ORT vs torch allclose(atol=1e-4): {ok_enc}")
    print(f"decoder ORT vs torch allclose(atol=1e-4): {ok_dec}")
    assert ok_enc and ok_dec, "numerical mismatch"

    # Greedy decode sanity: my ORT pipeline vs transformers.generate
    banner("Greedy decode sanity (ORT pipeline vs transformers.generate)")
    ort_ids = ort_greedy(enc_path, dec_path, wv, bos, eos, pad,
                         args.max_new_tokens, N)

    # transformers reference pipeline (uses KV cache internally but same greedy math)
    proc_inputs = processor(wv, return_tensors="pt", sampling_rate=16000)
    with torch.no_grad():
        gen_ids = model.generate(**proc_inputs, max_new_tokens=args.max_new_tokens,
                                 do_sample=False, num_beams=1, use_cache=True)
    gen_ids = gen_ids[0].tolist()

    print("ORT ids :", ort_ids)
    print("HF  ids :", gen_ids)
    match = ort_ids == gen_ids or ort_ids == gen_ids[1:]  # generation may prepend bos
    print("token-sequence match:", match)
    if processor is not None:
        print("text (ORT pipeline):", processor.tokenizer.decode(ort_ids, skip_special_tokens=True))
        print("text (transformers):", processor.batch_decode([gen_ids], skip_special_tokens=True)[0])
    assert match, "greedy decode mismatch between ORT pipeline and transformers"

    write_manifest(args.model, args.out, created_by="stage0_export.py",
                   args={"opset": args.opset, "stateless_decoder": True,
                         "dec_fix_len": N,
                         "attn_impl": getattr(cfg, "_attn_implementation", "default")},
                   extra={"bos": bos, "eos": eos, "pad": pad,
                          "max_length": cfg.max_position_embeddings,
                          "sanity_tokens_match": match})
    upsert_variant(args.model, args.out, path=args.out, dtype="fp32",
                   note="raw PyTorch export (encoder + stateless decoder)", created_by="stage0_export.py")
    # gen_meta.json: minimal inference metadata for the base-only (Pi) greedy path
    import json
    (task_dir(args.model) / "gen_meta.json").write_text(json.dumps({
        "task": args.model, "opset": args.opset, "dec_fix_len": N,
        "bos": bos, "eos": eos, "pad": pad, "max_length": cfg.max_position_embeddings,
        "encoder": f"{args.out}/encoder_model.onnx", "decoder": f"{args.out}/decoder_model.onnx",
        "feature_sampling_rate": 16000, "feature_do_normalize": False,
        "max_tokens_per_second": 13.0,
    }, indent=2))
    banner("Stage 0 export done -> models/%s/%s" % (args.model, args.out))


if __name__ == "__main__":
    main()