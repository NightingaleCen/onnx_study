# AGENTS.md — conventions for working in this repo

## Environment
- Manager: **uv** (Python 3.12 managed by uv, identical on mac & Pi via `.python-version`).
- Dependency groups: `base` (onnx/onnxruntime/onnxsim/numpy/soundfile/psutil/netron/tabulate/huggingface_hub/soxr) and `dev` (torch/transformers/datasets/onnxscript/librosa-excluded).
- Mac (dev + bench): `uv sync` (installs base+dev).
- Pi (bench only): `uv sync --no-group dev` (base only). To RUN stage0 export on the Pi you need the full `uv sync` (torch wheel exists for aarch64; add 2G swap for the 1GB-RAM Pi 3B).
- Proxy (only for downloads): `export https_proxy=http://127.0.0.1:7890 http_proxy=http://127.0.0.1:7890 all_proxy=http://127.0.0.1:7890` — enable before `dataset/download_models.py` / `dataset/prepare_data.py --hf-dataset` / `uv add`.

## Stage -> script map (all run via `uv run python scripts/<x>.py --model STT`)
| stage | script | produces (under models/STT/) | needs group |
|---|---|---|---|
| 0 download | dataset/download_models.py | hf/ | base |
| 0 export   | pipeline/export_onnx.py     | A/ (encoder_model.onnx, decoder_model.onnx) + gen_meta.json | dev |
| 1 simplify | pipeline/simplify.py | A_sim/ | base |
| 1b preopt  | pipeline/preopt.py   | B/ | base |
| 2 quantize | pipeline/quantize.py | C/ (and C_raw/ with `--in A`) | base |
| 3 runtime  | pipeline/runtime_opt.py | D/ | base |
| analysis  | analysis/analyze.py         | reports/analysis/*_*.json | base |
| bench     | analysis/bench.py           | reports/bench/<host>/*.csv | base |
| compare   | analysis/compare_outputs.py | stdout | base (+dev for --decode) |
| data      | dataset/prepare_data.py    | data/calibration/*, data/eval/* | base (+dev for --hf-dataset) |

Stage 0-export needs the `dev` group and is a **prerequisite** — run it once to
produce `A/`, then `build.py` (which only covers stages 1–3) uses it as an
`external` input. The Pi skips stage 0 entirely (models are synced from the Mac).

## Analysis passes (scripts/passes/)
Three stubs: `op_stats.py` (Pass1 operator counts), `quant_coverage.py` (Pass2 QDQ coverage), `redundant_qdq.py` (Pass3 redundant Q→DQ). Each has a documented I/O contract in its docstring and `run()` raising NotImplementedError. `analyze.py` tolerates NotImplementedError/error gracefully so you can implement one at a time.

## Critical gotchas (learned while building)
1. **Decoder export (stage0)**: `torch.onnx.export(dynamo=True)` + `attn_implementation="eager"` (SDPA's `is_causal` fails to symbolize under torch.export). The transformers causal-mask `CumSum` bakes the traced `dec_len` into the graph -> **`dec_len` is kept FIXED** (`--dec-fix-len`, default 128) and runtime greedy pads `input_ids` to N with pad_token (causal mask means padding never affects real positions). `enc_len` (decoder cross-attn) and `num_samples` (encoder input) stay SYMBOLIC. Validate any export change with: greedy token ids == `model.generate(...)` ids on a fixed waveform.
2. **dynamic_axes naming under dynamo**: when only ONE input has a symbolic dim, dynamo mis-applies the dim_param to the wrong tensor. Always pass empty-dict anchors for the concrete inputs too (see the decoder `dynamic_axes={"input_ids":{}, "encoder_hidden_states":{1:"enc_len"}, "logits":{}}` in export_onnx). Opset is 18 (dynamo min; outline wanted >=17).
3. **ORT `enable_mem_pattern` must be OFF** for any session run on variable `enc_len`/audio (else "Shape mismatch attempting to re-use buffer"). GreedyPipeline and stage3 do this.
4. **mac arm64 ORT wheel has NO XNNPACKExecutionProvider** (CoreML is the mac EP). D_xnn is produced only on the Pi (aarch64 ORT wheel includes XNNPACK). bench.py records `act_provider` so cross-machine comparison is honest.
5. **quant_pre_process fails** on moonshine encoder (incomplete symbolic shape inference on conv strides / num_samples). stage1b is best-effort: it copies the encoder through unchanged and only pre-processes the decoder. With dynamic quantization this is harmless — no calibration or shape inference is needed.
6. **`per_channel=True` trips CPU EP's QLinearMatMul** ("input zero point must be scalar"). Default is per-tensor (outline's S8S8 standard). `--per-channel` is opt-in for experimenting with XNNPACK per-channel on the Pi.
7. **stage3 file order**: process encoder before decoder (alphabetical would do decoder first, which needs encoder output as cross-attn input for the dump run).
8. **External data**: stage0 flattens `.onnx.data` into single-file protobufs so downstream onnxsim/quantize/netron are simple.

## Data ownership
YOU supply the dataset: drop audio in `data/raw/` (or `--hf-dataset`), run `dataset/prepare_data.py`. Split is deterministic (seed 1337) -> identical on both machines -> no audio sync needed. 16 kHz mono 16-bit PCM is the contract; `bench.py`/`stage2` read via `common.preprocess_audio` (soundfile-only, no transformers, Pi-safe).

## Models/outputs are gitignored and regenerable per machine; code + small reports/CSVs are in git.