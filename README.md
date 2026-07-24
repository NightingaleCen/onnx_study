# onnx_study

A hands-on study of ONNX export, onnxsim, static QDQ quantization, and ONNX Runtime
graph-optimization/EP-fusion for deploying a tiny ASR model on a Raspberry Pi 3B.

Model: **UsefulSensors/moonshine-tiny-zh** (Mandarin Moonshine, encoder-decoder,
27M params). The whole pipeline is task-named `STT` under `models/STT/`; a future
TTS model slots in identically as `models/TTS/`.

## Pipeline (stages)

```
0 download ── 0 export ─ 1 simplify ─ 1b preopt ─ 2 quantize ─ 3 runtime ─ 5 manual (opt)
   hf/            A/        A_sim/       B/         C/        D_xnn D_cpu     D_manual/
```
Each stage is an independently-runnable script. See `AGENTS.md` for the stage→script
map and the critical gotchas. The orchestrator runs them chained:

    uv run python scripts/run_pipeline.py --model STT --from 0-download --to 3-runtime

### Quick start (mac, dev)

    export https_proxy=http://127.0.0.1:7890 http_proxy=http://127.0.0.1:7890 all_proxy=http://127.0.0.1:7890  # only for downloads
    uv sync
    uv run python scripts/stage0_download.py --model STT         # HF snapshot + reference ONNX
    # drop your audio into data/raw/ (see "Data" below), then:
    uv run python scripts/prepare_data.py --model STT
    uv run python scripts/stage0_export.py --model STT           # needs dev (torch/transformers)
    uv run python scripts/stage1_simplify.py --model STT
    uv run python scripts/stage1b_preopt.py --model STT
    uv run python scripts/stage2_quantize.py --model STT         # -> C (per-tensor QDQ S8S8)
    uv run python scripts/stage3_runtime_opt.py --model STT       # -> D_cpu (mac); D_xnn on Pi
    uv run python scripts/analyze.py --model STT --stage C        # runs the 3 analysis passes
    uv run python scripts/bench.py --model STT                   # 8-group matrix -> CSV
    uv run python scripts/compare_outputs.py --model STT --ref A --test C

### Benchmark matrix (8 groups x 2 devices = 16)

`bench.py` runs variants {A, A_sim, C_raw, C, +D_xnn/D_cpu/D_manual if present} across
opt levels {DISABLE_ALL, EXTENDED}. EXTENDED requests XNNPACK (falls back to CPU on the
mac wheel; the EP actually used is recorded). Each Pi run writes
`reports/bench/<hostname>/*.csv` — push it back via git, then on the mac:

    uv run python scripts/make_report.py              # merged markdown table
    uv run python scripts/make_report.py --compare D_cpu D_manual   # manual-optim payoff

### Data (you supply)

Drop audio into `data/raw/` (.wav .flac .mp3 .ogg, recursively) OR pass
`--hf-dataset <repo> --hf-split <split>`. `prepare_data.py` converts to 16 kHz mono
16-bit PCM and does a **deterministic** (seed 1337) split into `data/calibration/` +
`data/eval/` — identical on both machines, so no audio sync needed. `data/` is
gitignored.

### Pi

See `docs/pi_setup.md`.

## Layout

```
scripts/        common.py, stage0_*.py, stage1*.py, stage2*.py, stage3*.py, stage5*.py,
                passes/ (3 stubs), analyze.py, bench.py, compare_outputs.py,
                make_report.py, run_pipeline.py, prepare_data.py
models/STT/     hf/, reference-onnx/, A/, A_sim/, B/, C/, C_raw/, D_xnn/, D_cpu/,
                D_manual/, variants.json (registry), manifest.json (per dir), gen_meta.json
data/           raw/ (drop zone), calibration/, eval/
reports/        analysis/, bench/<host>/*.csv, benchmark_report.md
```
`models/` and `data/` are gitignored — every machine builds its own (download +
pipeline). `reports/bench/*.csv` are committed for cross-machine merging.