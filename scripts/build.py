"""Build pipeline variants declaratively from pipeline.toml.

Resolves the DAG: each variant depends on its ``source``. Variants already
built in the same run (via shared dependencies) are skipped; nothing is
skipped based on disk state — every invocation rebuilds from scratch.

Variants marked ``external`` are not built by this script; they must already
exist on disk (e.g. A produced by export_onnx.py).

Usage:
    uv run python scripts/build.py                    # build all variants
    uv run python scripts/build.py --target C_skip_sim  # build a single variant + its chain
"""
from __future__ import annotations

import subprocess, sys, tomllib
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import banner, task_dir, list_onnx, DEFAULT_VARIANTS, save_variants  # noqa: E402

SCRIPT_ROOT = Path(__file__).resolve().parent / "pipeline"

STAGE_SPEC = {
    "1_simplify":    {"script": "simplify.py",  "has_source": True},
    "1b_preopt":     {"script": "preopt.py",   "has_source": True, "extra_args": ["enable_optimize"]},
    "2_quantize":    {"script": "quantize.py",  "has_source": True},
    "3_runtime":     {"script": "runtime_opt.py", "has_source": True, "extra_args": ["provider"]},
}


def build(target: str, model: str, variants: dict, built: set):
    if target in built:
        return
    v = variants[target]

    if external := v.get("external"):
        ext_dir = task_dir(model) / str(external)
        if not ext_dir.exists() or not list_onnx(ext_dir):
            raise SystemExit(f"variant '{target}' is external but {ext_dir} has no .onnx "
                             "files — run export_onnx.py first")
        print(f"  [{target}] external ({external}) -> skip")
        built.add(target)
        return

    if "stage" not in v:
        raise SystemExit(f"variant '{target}' has neither 'stage' nor 'external'")

    source = v.get("source")
    if source:
        if source not in variants:
            raise SystemExit(f"variant '{target}' depends on '{source}' which is not defined in pipeline.toml")
        build(source, model, variants, built)

    spec = STAGE_SPEC[v["stage"]]
    script = SCRIPT_ROOT / spec["script"]
    cmd = [sys.executable, str(script), "--model", model]
    if spec["has_source"]:
        src_v = variants[source]
        in_dir = str(src_v.get("external", source))
        cmd += ["--in", in_dir, "--out", target]
    for key in spec.get("extra_args", []):
        if key in v:
            if isinstance(v[key], bool):
                # boolean flags: only append the flag when true, so a literal
                # `key = false` in pipeline.toml does not enable it
                if v[key]:
                    cmd += [f"--{key}"]
            else:
                cmd += [f"--{key}", str(v[key])]
    print(f"  [{target}] build -> {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise SystemExit(f"build of {target} failed (exit {result.returncode})")
    built.add(target)


def rebuild_variants(model: str):
    """Rebuild variants.json from actual on-disk .onnx directories."""
    import json
    td = task_dir(model)
    variants: dict[str, dict] = {}
    for child in sorted(td.iterdir()):
        if not child.is_dir():
            continue
        onnx_files = list_onnx(child)
        if not onnx_files:
            continue
        name = child.name
        default = dict(DEFAULT_VARIANTS.get(name, {"path": name, "dtype": "?"}))
        entry = {"path": name, "dtype": default.get("dtype", "?"),
                 "note": default.get("note", "")}
        mf = child / "manifest.json"
        if mf.exists():
            manifest = json.loads(mf.read_text())
            if "created_by" in manifest:
                entry["created_by"] = manifest["created_by"]
            if "timestamp" in manifest:
                entry["timestamp"] = manifest["timestamp"]
        variants[name] = entry
    save_variants(model, variants)
    print(f"variants registry updated: {sorted(variants.keys())}")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default=None, help="single variant to build (default: all)")
    args = ap.parse_args()

    cfg = tomllib.loads((Path("pipeline.toml").read_text()))
    model: str = cfg["model"]["name"]
    variants: dict = cfg["model"]["variants"]

    in_degree: dict[str, int] = defaultdict(int)
    deps: dict[str, list[str]] = defaultdict(list)
    for name, v in variants.items():
        src = v.get("source")
        if src:
            in_degree[name] += 1
            deps[src].append(name)

    order = []
    q = [n for n in variants if in_degree[n] == 0]
    while q:
        n = q.pop(0)
        order.append(n)
        for child in deps[n]:
            in_degree[child] -= 1
            if in_degree[child] == 0:
                q.append(child)
    if len(order) != len(variants):
        raise SystemExit("pipeline.toml contains a circular dependency")

    targets = [args.target] if args.target else order
    banner(f"build pipeline ({model}) -> {targets}")
    built: set[str] = set()
    for t in targets:
        build(t, model, variants, built)
    print("done.")
    rebuild_variants(model)


if __name__ == "__main__":
    main()