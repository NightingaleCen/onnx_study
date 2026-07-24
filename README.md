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

### 树莓派 (Pi 3B, 64-bit OS)

```bash
git clone <your-repo> && cd onnx_study
uv sync                         # full sync if running stage0; --no-group dev for stages 1-5 only
uv run python -c "import onnxruntime as ort; print(ort.get_available_providers())"
# must show XNNPACKExecutionProvider -- this is the Pi's edge over the mac wheel.
```

**1GB RAM 注意**：stage0 (torch+transformers) 可能 OOM。提前加 2G swap：
```bash
sudo apt install -y dphys-swapfile
sudo sed -i 's/^CONF_SWAPSIZE=.*/CONF_SWAPSIZE=2048/' /etc/dphys-swapfile
sudo systemctl restart dphys-swapfile
```
stages 1–5 + bench 是 base-only，512MB 内存即够。

流水线命令和 mac 完全相同（同上 Quick start），跑完后推回 bench CSV：
```bash
uv run python scripts/bench.py --model STT --measurements 100 --warmup 20
git add reports/bench && git commit -m "bench: pi3b results" && git push
```
mac 端 `git pull` 后运行 `make_report.py` 得到 16 组对比总表。

Pi 上看 Netron：`uv run netron models/STT/D_xnn/encoder_model.onnx --port 8080` 然后笔记本上 `ssh -L 8080:localhost:8080 pi@<pi-host>`，浏览器打开 `localhost:8080`。

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