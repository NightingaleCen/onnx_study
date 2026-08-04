"""Provider attribution audit: which EP actually executes each node.

bench.py's ``act_provider`` column only reflects the *requested* EP list, not
which EP ran the nodes. XNNPACK is a partial EP (MatMul 2D / Gemm / Softmax
axis=last / 2D Conv ...), so unsupported nodes silently fall back to CPU EP.
This script enables ORT node-level profiling, runs one representative encoder
+ decoder inference per variant, and aggregates the Chrome-trace by
``args["provider"]``: node counts + total time per EP, plus the ops XNNPACK
actually claims.

    uv run python scripts/analysis/provider_audit.py --model STT --variants D,D_CPU
    uv run python scripts/analysis/provider_audit.py --model STT --variants D --no-cpu
    uv run python scripts/analysis/provider_audit.py --model STT --provider cpu
"""
from __future__ import annotations

import argparse, glob, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import (task_dir, load_variants, resolve_providers, ep_options,  # noqa: E402
                    preprocess_audio, load_gen_meta, hostname, REPORTS_DIR, banner)

import numpy as np
import onnxruntime as ort


def _parse_profile(path) -> dict[str, dict[str, tuple[int, int]]]:
    """Trace JSON -> {provider: {op_name: (calls, total_us)}}."""
    with open(path) as f:
        events = json.load(f)
    prov_op: dict[str, dict[str, tuple[int, int]]] = {}
    for ev in events:
        if ev.get("cat") != "Node" or "dur" not in ev:
            continue
        args = ev.get("args") or {}
        provider = args.get("provider")
        if provider is None:
            continue
        op = args.get("op_name") or ev.get("name")
        calls, us = prov_op.setdefault(provider, {}).get(op, (0, 0))
        prov_op[provider][op] = (calls + 1, us + ev["dur"])
    return prov_op


def _summarize(prov_op: dict[str, dict[str, tuple[int, int]]]) -> dict:
    """{provider: {nodes, total_us, ops:{op:{calls,us}}}}."""
    out = {}
    for provider, ops in prov_op.items():
        nodes = sum(c for c, _ in ops.values())
        us = sum(u for _, u in ops.values())
        out[provider] = {"nodes": nodes, "total_us": us,
                         "ops": {op: {"calls": c, "us": u} for op, (c, u) in ops.items()}}
    return out


def _make_session(path: Path, prov_key: str) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    so.enable_mem_pattern = False
    used, missing = resolve_providers(prov_key)
    if missing:
        print(f"    note: requested {missing} unavailable; using {used}")
    return ort.InferenceSession(str(path), so, providers=used,
                                provider_options=ep_options(used))


def _audit(model: str, vpath: Path, prov_key: str, wav: Path, meta: dict,
           dec_steps: int) -> dict:
    enc_path = vpath / "encoder_model.onnx"
    dec_path = vpath / "decoder_model.onnx"
    if not enc_path.exists() or not dec_path.exists():
        print(f"    skip: missing encoder_model.onnx/decoder_model.onnx in {vpath}")
        return {}

    enc = _make_session(enc_path, prov_key)
    feed = {"input_values": preprocess_audio(wav)[None, :].astype(np.float32)}
    enc.run(None, feed)  # warmup (not profiled)
    enc.enable_profiling()
    enc_out = enc.run(None, feed)[0]
    enc_trace = enc.end_profiling()
    enc_attr = _summarize(_parse_profile(enc_trace))
    Path(enc_trace).unlink(missing_ok=True)

    dec = _make_session(dec_path, prov_key)
    ids = [meta["bos"]] + [meta["pad"]] * (meta["dec_fix_len"] - 1)
    dec_feed = {"input_ids": np.asarray([ids], dtype=np.int64),
                "encoder_hidden_states": enc_out}
    dec.run(None, dec_feed)  # warmup
    dec.enable_profiling()
    for _ in range(dec_steps):
        dec.run(None, dec_feed)
    dec_trace = dec.end_profiling()
    dec_attr = _summarize(_parse_profile(dec_trace))
    Path(dec_trace).unlink(missing_ok=True)

    return {"encoder_model.onnx": enc_attr, "decoder_model.onnx": dec_attr}


def _fmt_table(name: str, attr: dict) -> list[str]:
    lines = [f"  {name}:"]
    rows = []
    for provider, a in attr.items():
        ms = a["total_us"] / 1e3
        rows.append((provider, a["nodes"], ms))
    tot = sum(a["total_us"] for a in attr.values()) or 1
    if not rows:
        lines.append("    (no profiled nodes)")
        return lines
    lines.append("    provider                     nodes   total_ms      pct")
    for provider, nodes, ms in sorted(rows, key=lambda r: -r[2]):
        pct = attr[provider]["total_us"] / tot * 100
        lines.append(f"    {provider:28} {nodes:5d} {ms:9.1f} {pct:7.1f}%")
    for provider in ("XnnpackExecutionProvider",):
        if provider not in attr:
            continue
        ops = sorted(attr[provider]["ops"].items(), key=lambda kv: -kv[1]["us"])
        if ops:
            desc = " | ".join(f"{op} x{c['calls']} ({c['us'] / 1e3:.1f}ms)"
                              for op, c in ops[:12])
            lines.append(f"    {provider} ops: {desc}")
            if len(ops) > 12:
                lines.append(f"    ... and {len(ops) - 12} more ops")
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="STT")
    ap.add_argument("--variants", default="D,D_CPU",
                    help="comma list; empty=auto(all existing registered)")
    ap.add_argument("--provider", default="xnnpack", choices=["cpu", "xnnpack"])
    ap.add_argument("--also-cpu", action="store_true", default=True,
                    help="when provider is xnnpack, also audit a CPU-only session")
    ap.add_argument("--no-cpu", dest="also_cpu", action="store_false")
    ap.add_argument("--dec-steps", type=int, default=1,
                    help="decoder steps to profile (default 1; more averages timings)")
    args = ap.parse_args()

    meta = load_gen_meta(args.model)
    wavs = sorted(glob.glob("data/calibration/*.wav")) or sorted(glob.glob("data/eval/*.wav"))
    assert wavs, "need data/calibration/*.wav or data/eval/*.wav (run prepare_data.py)"
    regs = load_variants(args.model)
    avail = [v for v, info in regs.items()
             if (task_dir(args.model) / info["path"]).exists()
             and list((task_dir(args.model) / info["path"]).glob("*.onnx"))]
    variants = [v for v in args.variants.split(",") if v] if args.variants else avail
    prov_keys = [args.provider] + (["cpu"] if (args.also_cpu and args.provider != "cpu") else [])

    banner(f"provider audit | host={hostname()} ort={ort.__version__} | variants={variants} providers={prov_keys}")

    report: dict = {"host": hostname(), "ort": ort.__version__, "variants": {}}
    for v in variants:
        vpath = task_dir(args.model) / regs[v]["path"]
        print(f"[{v}] {vpath}")
        report["variants"][v] = {}
        for prov in prov_keys:
            attr = _audit(args.model, vpath, prov, Path(wavs[0]), meta, args.dec_steps)
            if not attr:
                continue
            report["variants"][v][prov] = attr
            print(f"  provider={prov}")
            for name in ("encoder_model.onnx", "decoder_model.onnx"):
                lines = _fmt_table(name, attr.get(name, {}))
                print("\n".join(lines))

    out_dir = REPORTS_DIR / "audit" / hostname()
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{Path(__file__).stem}_{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
