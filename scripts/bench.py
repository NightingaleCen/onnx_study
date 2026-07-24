"""Stage 4 benchmark: variant x opt_level matrix -> latency + memory, CSV.

Default matrix reproduces the outline's 8 groups (+D_xnn/D_cpu/D_manual if present):
  variants {A, A_sim, C_raw, C, D_xnn, D_cpu, D_manual}  (only those that exist)
  x opt_level {ORT_DISABLE_ALL, ORT_ENABLE_EXTENDED}

For opt_level=DISABLE_ALL the provider is CPU; for EXTENDED we REQUEST XNNPACK
(falling back to CPU where -- like the mac ORT wheel -- XNNPACK is not compiled in);
the ACTUAL provider used is recorded in the CSV so cross-machine comparison is honest.

Each "measurement" is one full greedy decode of one eval wav (fixed
max_new_tokens so decode length is identical across variants). Warmup runs first.

    uv run python scripts/bench.py --model STT --measurements 30 --warmup 5
    uv run python scripts/bench.py --model STT --variants A,C,D_manual
"""
from __future__ import annotations

import argparse, csv, glob, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (task_dir, load_variants, REPORTS_DIR,  # noqa: E402
                    hostname, git_commit, peak_rss_mb, resolve_providers,
                    preprocess_audio, banner)
from common import GreedyPipeline  # noqa: E402

import numpy as np
import onnxruntime as ort
import psutil


def eval_wavs():
    wavs = sorted(glob.glob("data/eval/*.wav"))
    assert wavs, "need data/eval/*.wav (run prepare_data.py)"
    return wavs


def percentile(v, q):
    return float(np.percentile(np.asarray(v, dtype=float), q)) if v else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="STT")
    ap.add_argument("--variants", default="", help="comma list; empty=auto(all existing registered)")
    ap.add_argument("--opt-levels", default="disable_all,extended")
    ap.add_argument("--measurements", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--max-new-tokens", type=int, default=24)
    ap.add_argument("--threads", type=int, default=None)
    args = ap.parse_args()

    try:
        meta = __import__("common").load_gen_meta(args.model)
        mnt = min(args.max_new_tokens, meta["dec_fix_len"] - 1)
    except Exception:
        mnt = args.max_new_tokens
    wavs = eval_wavs()
    regs = load_variants(args.model)
    avail = [v for v, info in regs.items()
             if (task_dir(args.model) / info["path"]).exists()
             and list((task_dir(args.model) / info["path"]).glob("*.onnx"))]
    variants = [v for v in args.variants.split(",") if v] if args.variants else avail
    opt_levels = [o for o in args.opt_levels.split(",") if o]
    baseline_rss = psutil.Process().memory_info().rss / 1e6
    banner(f"bench | host={hostname()} ort={ort.__version__} | variants={variants} opts={opt_levels} measures={args.measurements}")

    rows = []
    for v in variants:
        vpath = task_dir(args.model) / regs[v]["path"]
        dtype = regs[v].get("dtype", "?")
        for opt in opt_levels:
            prov_key = "cpu" if opt == "disable_all" else "xnnpack"
            used, missing = resolve_providers(prov_key)
            pipe = GreedyPipeline(args.model, vpath, opt_level=opt,
                                  providers=prov_key, intra_op_threads=args.threads)
            w0 = preprocess_audio(Path(wavs[0]))
            for _ in range(args.warmup):
                pipe.run(w0, mnt)
            times, enct, dect = [], [], []
            i = 0
            for _ in range(args.measurements):
                w = preprocess_audio(Path(wavs[i % len(wavs)])); i += 1
                _, t = pipe.run(w, mnt)
                times.append(t["total_ms"]); enct.append(t["enc_ms"]); dect.append(t["dec_total_ms"])
            peak_now = psutil.Process().memory_info().rss / 1e6 - baseline_rss
            row = dict(host=hostname(), git=git_commit(), ort=ort.__version__,
                       model=args.model, variant=v, dtype=dtype, opt_level=opt,
                       req_provider=prov_key, act_provider=used[0], threads=args.threads or "def",
                       measurements=args.measurements, warmup=args.warmup, max_new_tokens=mnt,
                       p50=percentile(times, 50), p95=percentile(times, 95), p99=percentile(times, 99),
                       enc_p50=percentile(enct, 50), dec_p50=percentile(dect, 50),
                       model_rss_mb=peak_now, ts=time.strftime("%Y%m%d-%H%M%S"))
            rows.append(row)
            print(f"  [{v:8}|{opt:8}|{row['act_provider']:22}] p50={row['p50']:.1f} p95={row['p95']:.1f} "
                  f"p99={row['p99']:.1f}ms  enc_p50={row['enc_p50']:.1f} dec_p50={row['dec_p50']:.1f}  rss={row['model_rss_mb']:.0f}MB")

    out_dir = REPORTS_DIR / "bench" / hostname()
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{time.strftime('%Y%m%d-%H%M%S')}.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"\nprocess peak rss (total): {peak_rss_mb():.0f}MB\nwrote {out}")


if __name__ == "__main__":
    main()