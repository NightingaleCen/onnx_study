# Pi setup (Raspberry Pi 3B, 64-bit Raspberry Pi OS Bookworm)

The Pi runs the **same repo** (pulled via git) and the **same scripts**, building its
own copy of every artifact from the HF download onward — models are NOT synced from the
mac. ORT's official `manylinux_2_28_aarch64` wheel works on Bookworm (glibc 2.36).

## 1. Install uv

    curl -LsSf https://astral.sh/uv/install.sh | sh
    # reopen shell / `source $HOME/.local/bin/env` so `uv` is on PATH

## 2. Clone & sync

    git clone <your-repo> onnx_study && cd onnx_study
    uv sync --no-group dev          # base only: enough for stages 1..end + bench
    # if you want to RUN stage0_export on the Pi too (you said you would run every
    # stage on each machine), use the full sync instead:
    uv sync                          # installs torch/transformers (~1.5GB disk)

Verify ONNX Runtime + XNNPACK:

    uv run python -c "import onnxruntime as ort; print(ort.get_available_providers())"
    # must list  'XNNPACKExecutionProvider'   (mac does NOT -- this is the Pi's edge)

## 3. 1GB RAM caveats (Pi 3B)

- `uv sync` (full) downloading torch is fine; **importing** torch + transformers +
  exporting a 27M-param model can exceed 1GB and OOM-kill. Add swap **before** running
  stage0 on the Pi:

      sudo apt install -y dphys-swapfile
      sudo sed -i 's/^CONF_SWAPSIZE=.*/CONF_SWAPSIZE=2048/' /etc/dphys-swapfile
      sudo systemctl restart dphys-swapfile
    (or use zram: `sudo apt install zram-tools` with a 2G config.)

- bench/stages 1-5 are light (base deps) and run comfortably in 512MB.
- If stage0 export keeps OOMing on the Pi, you have a design choice: run stage0 on the
  mac and physically copy `models/STT/A/` (and onward) onto the Pi by SD card / scp.
  The repo's "no-sync" convention was your call — relax it only if needed.

## 4. Run the pipeline on the Pi

    export https_proxy=http://127.0.0.1:7890 http_proxy=http://127.0.0.1:7890 all_proxy=http://127.0.0.1:7890  # only stage0_download / hf-dataset
    uv run python scripts/stage0_download.py --model STT
    # put audio in data/raw/ the SAME as on the mac, then:
    uv run python scripts/prepare_data.py --model STT        # deterministic -> identical split
    uv run python scripts/stage0_export.py --model STT       # heavy (swap on)
    uv run python scripts/stage1_simplify.py --model STT
    uv run python scripts/stage1b_preopt.py --model STT
    uv run python scripts/stage2_quantize.py --model STT
    uv run python scripts/stage3_runtime_opt.py --model STT  # NOW D_xnn is produced (XNNPACK present)

## 5. Benchmark + send results back

    uv run python scripts/bench.py --model STT --measurements 100 --warmup 20
    # -> reports/bench/<this-pi-hostname>/*.csv
    git add reports/bench && git commit -m "bench: pi3b results" && git push

On the mac, `git pull`, then:

    uv run python scripts/make_report.py
    # -> reports/benchmark_report.md  (mac vs Pi side-by-side, 16-group table)

## 6. Watch in Netron (optional on Pi)

    uv run netron models/STT/D_xnn/encoder_model.onnx --port 8080
    # then from your laptop:  ssh -L 8080:localhost:8080 pi@<pi-host>  and open localhost:8080
    # (or just copy the .onnx back and open in Netron on the mac)