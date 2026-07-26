"""Stage 4 benchmark: variant x opt_level matrix -> latency + memory, CSV.

Default matrix reproduces the outline's 8 groups (+D_xnn/D_cpu/D_manual if present):
  variants {A, A_sim, C_raw, C, D_xnn, D_cpu, D_manual}  (only those that exist)
  x opt_level {ORT_DISABLE_ALL, ORT_ENABLE_EXTENDED}

For opt_level=DISABLE_ALL the provider is CPU; for EXTENDED we REQUEST XNNPACK
(falling back to CPU where -- like the mac ORT wheel -- XNNPACK is not compiled in);
the ACTUAL provider used is recorded in the CSV so cross-machine comparison is honest.

Each "measurement" is one full greedy decode of one eval wav (fixed
max_new_tokens so decode length is identical across variants). Warmup runs first.

With --accuracy, after latency measurements bench.py also computes WER/CER on the
eval set (one decode per clip; reference text from data/eval/transcripts.csv, which
prepare_data.py produces if --text-column / --transcripts is supplied).

    uv run python scripts/analysis/bench.py --model STT --measurements 30 --warmup 5
    uv run python scripts/analysis/bench.py --model STT --variants A,C,D_manual
    uv run python scripts/analysis/bench.py --model STT --accuracy          # + WER/CER
"""
from __future__ import annotations

import argparse, csv, glob, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import (task_dir, load_variants, REPORTS_DIR,  # noqa: E402
                    hostname, git_commit, peak_rss_mb, resolve_providers,
                    preprocess_audio, load_tokenizer, load_eval_transcripts, banner)
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


def accuracy_scores(variant_pipe, wavs, transcripts, decode_fn, max_new_tokens):
    """Run one greedy decode per eval wav and compute mean WER/CER vs reference."""
    import jiwer
    wers, cers, n_ok = [], [], 0
    for wpath in wavs:
        ref = transcripts.get(Path(wpath).name)
        if ref is None:
            continue
        wav = preprocess_audio(Path(wpath))
        ids, _ = variant_pipe.run(wav, max_new_tokens)
        hyp = decode_fn(ids)
        ref_clean = ref.replace(" ", "")
        hyp_clean = hyp.replace(" ", "")
        wers.append(jiwer.wer(ref_clean, hyp_clean))
        cers.append(jiwer.cer(ref_clean, hyp_clean))
        n_ok += 1
    if not wers:
        return float("nan"), float("nan"), 0
    return float(np.mean(wers)), float(np.mean(cers)), n_ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="STT")
    ap.add_argument("--variants", default="", help="comma list; empty=auto(all existing registered)")
    ap.add_argument("--opt-levels", default="disable_all,extended")
    ap.add_argument("--measurements", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--max-new-tokens", type=int, default=0,
                    help="max tokens per decode (0 = auto from audio duration × 13 tok/s)")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--accuracy", action="store_true",
                    help="measure WER/CER on eval set (needs data/eval/transcripts.csv)")
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

    transcripts = load_eval_transcripts() if args.accuracy else None
    decode_fn = None
    if transcripts:
        decode_fn = load_tokenizer(args.model)

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

            wer, cer, n_acc = float("nan"), float("nan"), 0
            if transcripts and decode_fn:
                wer, cer, n_acc = accuracy_scores(pipe, wavs, transcripts, decode_fn, mnt)

            row = dict(host=hostname(), git=git_commit(), ort=ort.__version__,
                       model=args.model, variant=v, dtype=dtype, opt_level=opt,
                       req_provider=prov_key, act_provider=used[0], threads=args.threads or "def",
                       measurements=args.measurements, warmup=args.warmup, max_new_tokens=mnt,
                       p50=percentile(times, 50), p95=percentile(times, 95), p99=percentile(times, 99),
                       enc_p50=percentile(enct, 50), dec_p50=percentile(dect, 50),
                       model_rss_mb=peak_now, wer=wer, cer=cer, acc_n=n_acc,
                       ts=time.strftime("%Y%m%d-%H%M%S"))
            rows.append(row)
            acc_str = f"  wer={row['wer']:.3f} cer={row['cer']:.3f} (n={n_acc})" if n_acc else ""
            print(f"  [{v:8}|{opt:8}|{row['act_provider']:22}] p50={row['p50']:.1f} p95={row['p95']:.1f} "
                  f"p99={row['p99']:.1f}ms  enc_p50={row['enc_p50']:.1f} dec_p50={row['dec_p50']:.1f}  rss={row['model_rss_mb']:.0f}MB{acc_str}")

    out_dir = REPORTS_DIR / "bench" / hostname()
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{time.strftime('%Y%m%d-%H%M%S')}.csv"
    fieldnames = list(rows[0].keys())
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames); w.writeheader(); w.writerows(rows)
    print(f"\nprocess peak rss (total): {peak_rss_mb():.0f}MB\nwrote {out}")


if __name__ == "__main__":
    main()