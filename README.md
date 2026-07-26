# onnx_study

A hands-on study of ONNX export, onnxsim, static QDQ quantization, and ONNX Runtime
graph-optimization/EP-fusion for deploying an ASR model on a Raspberry Pi 3B.

The pipeline is task-named (e.g. `STT` under `models/STT/`).

## Pipeline

```
0 download ── 0 export ─ 1 simplify ─ 1b preopt ─ 2 quantize ─ 3 runtime
   hf/            A/        A_sim/       B/         C/        D_xnn D_cpu
```
Each stage is an independently-runnable script. The orchestrator runs them chained:

    uv run python scripts/run_pipeline.py --model STT --from 0-download --to 3-runtime

### Quick start

    uv run python scripts/download_models.py --model STT         # HF snapshot
    uv run python scripts/prepare_data.py --model STT
    uv run python scripts/export_onnx.py --model STT           # needs dev (torch/transformers)
    uv run python scripts/simplify.py --model STT
    uv run python scripts/preopt.py --model STT
    uv run python scripts/quantize.py --model STT              # -> C (dynamic QDQ, QInt8 weights)
    uv run python scripts/runtime_opt.py --model STT            # -> D_cpu (mac); D_xnn on Pi
    uv run python scripts/analyze.py --model STT --stage C      # runs the 3 analysis passes
    uv run python scripts/bench.py --model STT                  # 8-group matrix -> CSV
    uv run python scripts/compare_outputs.py --model STT --ref A --test C

### Benchmark

`bench.py` runs all variants {A, A_sim, C_raw, C, +D_xnn/D_cpu if present, etc.} across
opt levels {DISABLE_ALL, EXTENDED}. EXTENDED requests XNNPACK (falls back to CPU if not available; the EP actually used is recorded).

### Data

Drop audio into `data/raw/` (.wav .flac .mp3 .ogg, recursively) OR pass
`--hf-dataset <repo> --hf-split <split>`. `prepare_data.py` converts to 16 kHz mono
16-bit PCM, adds optional transcripts via `--text-column` / `--transcripts`, and does
a split into `data/calibration/` + `data/eval/`.

For convenience, `fetch_aishell.py` downloads a small subset of
[AISHELL-1](https://huggingface.co/datasets/AISHELL/AISHELL-1) (3 speakers, ~80
clips with transcripts) directly into `data/raw/` — ready to feed into
`prepare_data.py`:

    uv run python scripts/fetch_aishell.py
    uv run python scripts/prepare_data.py --model STT --transcripts data/raw/transcripts.tsv