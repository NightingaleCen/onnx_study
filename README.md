# onnx_study

A hands-on study of ONNX export, onnxsim, static QDQ quantization, and ONNX Runtime
graph-optimization/EP-fusion for deploying an ASR model on a Raspberry Pi 3B.

The pipeline is task-named (e.g. `STT` under `models/STT/`).

## Pipeline

```
0 download ── 0 export ─ 1 simplify ─ 1b preopt ─ 2 quantize ─ 3 runtime
   hf/            A/        A_sim/       B/         C/        D_xnn D_cpu
```

### `build.py` — declarative pipeline (recommended)

`build.py` reads `pipeline.toml` (which defines the DAG of variants and their
dependencies) and runs the stage scripts in topological order. It skips variants
whose outputs already exist (use `--force` to rebuild). Each completed stage
auto-registers itself for downstream benchmarking and analysis.

```
# Build all variants in dependency order:
uv run python scripts/build.py

# Build a single variant (+ its dependency chain):
uv run python scripts/build.py --target C_skip_sim

# Force rebuild even if outputs exist:
uv run python scripts/build.py --force
```

`pipeline.toml` format:
```toml
[model]
name = "STT"

[model.variants.A]
stage = "0_export"

[model.variants.A_sim]
stage = "1_simplify"
source = "A"

[model.variants.C]
stage = "2_quantize"
source = "B"
```

`stage` keys: `0_export`, `1_simplify`, `1b_preopt`, `2_quantize`.

### `run_pipeline.py` — explicit chain

The orchestrator runs stages sequentially:

    uv run python scripts/run_pipeline.py --model STT --from 0-download --to 3-runtime

### Quick start

    uv run python scripts/dataset/download_models.py --model STT         # HF snapshot
    uv run python scripts/dataset/prepare_data.py --model STT
    uv run python scripts/pipeline/export_onnx.py --model STT           # needs dev (torch/transformers)
    uv run python scripts/pipeline/simplify.py --model STT
    uv run python scripts/pipeline/preopt.py --model STT
    uv run python scripts/pipeline/quantize.py --model STT              # -> C (dynamic QDQ, QInt8 weights)
    uv run python scripts/pipeline/runtime_opt.py --model STT            # -> D_cpu (mac); D_xnn on Pi
    uv run python scripts/analysis/analyze.py --model STT --stage C      # runs the 3 analysis passes
    uv run python scripts/analysis/bench.py --model STT                  # 8-group matrix -> CSV
    uv run python scripts/analysis/compare_outputs.py --model STT --ref A --test C

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

    uv run python scripts/dataset/fetch_aishell.py
    uv run python scripts/dataset/prepare_data.py --model STT --transcripts data/raw/transcripts.tsv