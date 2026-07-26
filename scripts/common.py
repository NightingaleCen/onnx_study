"""Shared utilities for the onnx_study pipeline.

Conventions
-----------
- A "task" (e.g. "STT", "TTS") maps to a top-level folder under ``models/``.
- Each stage of the pipeline writes its ONNX products into a subfolder of the
  task dir (``A``, ``A_sim``, ``B``, ``C``, ``C_raw``, ``D_xnn``, ``D_cpu``,
  ``D_manual``, ...). A stage folder may contain **several** .onnx files
  (encoder-decoder architectures produce more than one).
- Every stage folder carries a ``manifest.json`` describing its provenance.
- The task dir carries a ``variants.json`` registry: name -> metadata, which
  ``bench.py`` iterates. Manually-created variants are added the same way.

This module is import-safe (no heavy deps imported at module top-level besides
onnxruntime/numpy which are in the base dependency group).
"""
from __future__ import annotations

import json
import os
import platform
import resource
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"

CANONICAL_SR = 16000  # moonshine expects 16 kHz raw waveform input
MAX_TOKENS_PER_SECOND = 13.0  # moonshine README: non-Latin languages (including Chinese)


# --------------------------------------------------------------------------- paths
def task_dir(task: str) -> Path:
    return MODELS_DIR / task


def stage_dir(task: str, stage: str) -> Path:
    d = task_dir(task) / stage
    d.mkdir(parents=True, exist_ok=True)
    return d


def hf_dir(task: str) -> Path:
    d = task_dir(task) / "hf"
    d.mkdir(parents=True, exist_ok=True)
    return d


def variants_path(task: str) -> Path:
    return task_dir(task) / "variants.json"


def list_onnx(d: Path) -> list[Path]:
    return sorted(p for p in d.iterdir() if p.suffix == ".onnx")


# --------------------------------------------------------------------------- git
def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT, capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


def hostname() -> str:
    return platform.node().split(".")[0]


# --------------------------------------------------------------------------- variant registry
DEFAULT_VARIANTS: dict[str, dict] = {
    "A":       {"path": "A",       "dtype": "fp32", "note": "raw PyTorch export"},
    "A_sim":   {"path": "A_sim",   "dtype": "fp32", "note": "onnxsim simplified"},
    "B":       {"path": "B",       "dtype": "fp32", "note": "quant_pre_process"},
    "C":       {"path": "C",       "dtype": "int8", "note": "QDQ static quant from B"},
    "C_raw":   {"path": "C_raw",   "dtype": "int8", "note": "QDQ static quant from A (no simplify)"},
    "D_xnn":   {"path": "D_xnn",   "dtype": "int8", "note": "runtime-optimized, XNNPACK EP"},
    "D_cpu":   {"path": "D_cpu",   "dtype": "int8", "note": "runtime-optimized, CPU EP"},
    "D_manual": {"path": "D_manual", "dtype": "int8", "note": "user hand-edited"},
}


def load_variants(task: str) -> dict[str, dict]:
    p = variants_path(task)
    if p.exists():
        return json.loads(p.read_text())
    return {}


def save_variants(task: str, variants: dict[str, dict]) -> None:
    variants_path(task).write_text(json.dumps(variants, indent=2, ensure_ascii=False))


def upsert_variant(task: str, name: str, **meta) -> None:
    v = load_variants(task)
    entry = dict(DEFAULT_VARIANTS.get(name, {"path": name}))
    entry["path"] = meta.get("path", entry["path"])
    if "dtype" in meta:
        entry["dtype"] = meta["dtype"]
    if "note" in meta:
        entry["note"] = meta["note"]
    for k in ("base", "created_by", "timestamp", "git_commit"):
        if k in meta:
            entry[k] = meta[k]
    v[name] = entry
    save_variants(task, v)


# --------------------------------------------------------------------------- manifest
def onnx_meta(path: Path) -> dict:
    import onnx
    m = onnx.load(str(path))
    nodes = len(m.graph.node)
    opsets = {d.domain or "ai.onnx": d.version for d in m.opset_import}
    ir = m.ir_version
    inputs = [(i.name, [d.dim_value or d.dim_param or "?" for d in i.type.tensor_type.shape.dim])
              for i in m.graph.input]
    outputs = [(o.name, [d.dim_value or d.dim_param or "?" for d in o.type.tensor_type.shape.dim])
               for o in m.graph.output]
    return {
        "name": path.name,
        "size_bytes": path.stat().st_size,
        "ir": ir,
        "opsets": opsets,
        "nodes": nodes,
        "inputs": inputs,
        "outputs": outputs,
    }


def write_manifest(task: str, stage: str, *, created_by: str,
                   args: dict | None = None, extra: dict | None = None) -> Path:
    d = stage_dir(task, stage)
    files = [onnx_meta(p) for p in list_onnx(d)]
    manifest = {
        "task": task,
        "stage": stage,
        "created_by": created_by,
        "git_commit": git_commit(),
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "args": args or {},
        "files": files,
        "extra": extra or {},
    }
    mp = d / "manifest.json"
    mp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    return mp


def read_manifest(task: str, stage: str) -> dict | None:
    mp = stage_dir(task, stage) / "manifest.json"
    if mp.exists():
        return json.loads(mp.read_text())
    return None


# --------------------------------------------------------------------------- ORT sessions
OPT_LEVELS = {
    "disable_all": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
    "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
    "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
    "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
}

AVAILABLE_PROVIDERS = set(ort.get_available_providers())


def resolve_providers(requested: str | list[str]) -> tuple[list[str], list[str]]:
    """Return (providers_actually_used, providers_requested_but_unavailable).

    ``requested`` may be 'xnnpack', 'cpu', 'coreml' or an explicit EP list.
    XNNPACK is requested as 'xnnpack'/'XNNPACKExecutionProvider'.
    """
    name_map = {"xnnpack": "XNNPACKExecutionProvider", "cpu": "CPUExecutionProvider",
                "coreml": "CoreMLExecutionProvider"}
    if isinstance(requested, str):
        requested = [requested]
    want = [name_map.get(r.lower(), r) for r in requested]
    have, missing = [], []
    for p in want:
        (have if p in AVAILABLE_PROVIDERS else missing).append(p)
    if not have:
        have = ["CPUExecutionProvider"]
    return have, missing


def make_session(model_path: Path | str, *, opt_level: str = "extended",
                 providers: str | list[str] = "cpu", intra_op_threads: int | None = None,
                 optimized_model_filepath: str | None = None) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.graph_optimization_level = OPT_LEVELS[opt_level]
    if intra_op_threads is not None:
        so.intra_op_num_threads = intra_op_threads
    if optimized_model_filepath:
        so.optimized_model_filepath = optimized_model_filepath
    used, _ = resolve_providers(providers)
    return ort.InferenceSession(str(model_path), so, providers=used)


# --------------------------------------------------------------------------- timing & memory
def percentiles(values: list[float], qs=(50, 95, 99)) -> dict:
    if not values:
        return {f"p{q}": float("nan") for q in qs}
    arr = np.asarray(values, dtype=np.float64)
    return {f"p{q}": float(np.percentile(arr, q)) for q in qs}


def peak_rss_mb() -> float:
    """Peak RSS in MB. Cross-platform: macOS reports bytes, Linux reports KB."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() == "Darwin":
        return rss / (1024 * 1024)
    return rss / 1024  # Linux: KB -> MB


class Timer:
    def __enter__(self):
        self.t = time.perf_counter()
        return self
    def __exit__(self, *exc):
        self.elapsed = time.perf_counter() - self.t


# --------------------------------------------------------------------------- small helpers
def ensure_empty_dir(d: Path) -> None:
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True, exist_ok=True)


def ndjson_append(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def banner(msg: str) -> None:
    bar = "=" * len(msg)
    print(f"\n{bar}\n{msg}\n{bar}")


# --------------------------------------------------------------------------- gen_meta + audio
def load_gen_meta(task: str) -> dict:
    p = task_dir(task) / "gen_meta.json"
    return json.loads(p.read_text())


def preprocess_audio(path: Path, sr: int = CANONICAL_SR) -> np.ndarray:
    """Read an audio file into a float32 1-D waveform at the expected sample rate.

    The pipeline's prepare_data.py guarantees 16 kHz mono WAV, so this only does
    a fast soundfile.read + dtype cast + mono mixdown. No resampling dependency
    (keeps the Pi runtime free of transformers/librosa).
    """
    import soundfile as sf
    data, file_sr = sf.read(str(path), dtype="float32", always_2d=True)
    data = data.mean(axis=1) if data.shape[1] > 1 else data[:, 0]
    if file_sr != sr:
        raise ValueError(f"{path.name}: sr={file_sr} but model expects {sr}. "
                          "Run scripts/prepare_data.py to produce 16 kHz mono WAV.")
    return data.astype(np.float32)


# --------------------------------------------------------------------------- greedy pipeline
class GreedyPipeline:
    """ORT-only encoder+decoder greedy decode (no transformers -> runs on Pi).

    The decoder is stateless with a FIXED dec_len (padded at runtime, see
    stage0_export.py). enc_len varies per audio; mem_pattern is disabled so one
    session accepts many audio lengths without buffer-reuse conflicts.
    """

    def __init__(self, task: str, variant_path: Path, opt_level: str = "extended",
                 providers: str | list[str] = "cpu", intra_op_threads: int | None = None):
        self.meta = load_gen_meta(task)
        self.bos = self.meta["bos"]
        self.eos = self.meta["eos"]
        self.pad = self.meta["pad"]
        self.N = self.meta["dec_fix_len"]
        so = ort.SessionOptions()
        so.graph_optimization_level = OPT_LEVELS[opt_level]
        so.enable_mem_pattern = False
        if intra_op_threads is not None:
            so.intra_op_num_threads = intra_op_threads
        used, missing = resolve_providers(providers)
        if missing:
            print(f"[GreedyPipeline] requested {missing} unavailable on this build; using {used}")
        self.enc_sess = ort.InferenceSession(str(variant_path / "encoder_model.onnx"), so, providers=used)
        self.dec_sess = ort.InferenceSession(str(variant_path / "decoder_model.onnx"), so, providers=used)

    def run(self, input_values: np.ndarray, max_new_tokens: int = 0) -> tuple[list[int], dict]:
        import math, time
        if max_new_tokens <= 0:
            dur = len(input_values) / CANONICAL_SR
            max_new_tokens = min(math.ceil(dur * MAX_TOKENS_PER_SECOND), self.N - 1)
        assert max_new_tokens + 1 <= self.N, f"max_new_tokens={max_new_tokens} exceeds dec_fix_len={self.N}"
        t0 = time.perf_counter()
        enc_out = self.enc_sess.run(None, {"input_values": input_values[None, :].astype(np.float32)})[0]
        enc_ms = (time.perf_counter() - t0) * 1e3
        ids = [self.bos]
        step_ms = []
        for _ in range(max_new_tokens):
            padded = ids + [self.pad] * (self.N - len(ids))
            t1 = time.perf_counter()
            logits = self.dec_sess.run(None, {"input_ids": np.asarray([padded], dtype=np.int64),
                                             "encoder_hidden_states": enc_out})[0]
            step_ms.append((time.perf_counter() - t1) * 1e3)
            nxt = int(np.argmax(logits[0, len(ids) - 1]))
            ids.append(nxt)
            if nxt == self.eos:
                break
        return ids, {"enc_ms": enc_ms, "dec_steps_ms": step_ms, "dec_total_ms": float(sum(step_ms)),
                     "total_ms": enc_ms + sum(step_ms), "n_tokens": len(ids) - 1}


# --------------------------------------------------------------------------- tokenizer (base-safe, no transformers on Pi)
def load_tokenizer(task: str):
    """Return a callable: token_ids(list[int]) -> text(str).

    Prefers transformers AutoProcessor (mac, dev group); falls back to the
    ``tokenizers`` library (base group, works on Pi -- reads tokenizer.json).
    """
    tk_path = hf_dir(task) / "tokenizer.json"
    try:
        from transformers import AutoProcessor
        proc = AutoProcessor.from_pretrained(str(hf_dir(task)))
        def _decode(ids):
            return proc.tokenizer.decode(ids, skip_special_tokens=True)
        return _decode
    except Exception:
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(str(tk_path))
        def _decode(ids):
            return tok.decode(ids, skip_special_tokens=True)
        return _decode


def load_eval_transcripts() -> dict[str, str] | None:
    """Read ``data/eval/transcripts.csv`` (two cols: filename, text)."""
    p = Path("data/eval/transcripts.csv")
    if not p.exists():
        return None
    mapping = {}
    with p.open() as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t" if "\t" in line else ",")
            if len(parts) >= 2:
                mapping[parts[0].strip()] = parts[1].strip()
    return mapping