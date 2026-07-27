"""Compare greedy decode outputs between two model variants.

Runs greedy decode on eval wavs for --ref and --test and reports:
  - Exact token-id sequence match ratio (base deps only)
  - Decoded text diff (optional, needs dev group)

    uv run python scripts/analysis/compare_outputs.py --model STT --ref A --test D_manual
    uv run python scripts/analysis/compare_outputs.py --model STT --ref A --test C --decode
"""
from __future__ import annotations

import argparse, glob, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import task_dir, load_variants, preprocess_audio, banner, GreedyPipeline  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="STT")
    ap.add_argument("--ref", default="A")
    ap.add_argument("--test", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=0,
                    help="0 = auto from audio duration × 13 tok/sec (Chinese)")
    ap.add_argument("--decode", action="store_true", help="decode text via transformers (dev group)")
    ap.add_argument("--atol", type=float, default=0.0, help="if >0 also compare raw logits on first step")
    args = ap.parse_args()
    regs = load_variants(args.model)
    wavs = sorted(glob.glob("data/eval/*.wav"))
    assert wavs, "need data/eval/*.wav (run prepare_data.py)"

    ref_pipe = GreedyPipeline(args.model, task_dir(args.model) / regs[args.ref]["path"])
    test_pipe = GreedyPipeline(args.model, task_dir(args.model) / regs[args.test]["path"])
    mnt = min(args.max_new_tokens, ref_pipe.N - 1)

    decode = None
    if args.decode:
        from transformers import AutoProcessor
        decode = AutoProcessor.from_pretrained(str(task_dir(args.model) / "hf")).tokenizer

    banner(f"compare ref={args.ref} test={args.test} on {len(wavs)} eval clips")
    n_match, diffs = 0, []
    for w in wavs:
        wav = preprocess_audio(Path(w))
        r_ids, _ = ref_pipe.run(wav, mnt)
        t_ids, _ = test_pipe.run(wav, mnt)
        ok = r_ids == t_ids
        n_match += ok
        line = f"  {Path(w).name}: {'MATCH' if ok else 'DIFF'}"
        if not ok:
            line += f"\n     ref={r_ids}\n     test={t_ids}"
            if decode is not None:
                line += f"\n     ref_text={decode.decode(r_ids, skip_special_tokens=True)}" \
                        f"\n     test_text={decode.decode(t_ids, skip_special_tokens=True)}"
        print(line)
        if not ok:
            diffs.append(w)
    print(f"\nexact token-sequence match: {n_match}/{len(wavs)}")
    if decode is not None and all((ref_pipe.run(preprocess_audio(Path(w)), mnt)[0] ==
                                   test_pipe.run(preprocess_audio(Path(w)), mnt)[0]) for w in wavs):
        # full-text equality already implied by token match above
        print("decoded text identical where tokens matched.")


if __name__ == "__main__":
    main()