"""Build pipeline variants declaratively from pipeline.toml.

Resolves the DAG: each variant depends on its ``source``. Variants already
built in the same run (via shared dependencies) are skipped; nothing is
skipped based on disk state — every invocation rebuilds from scratch.

Usage:
    uv run python scripts/build.py                    # build all variants
    uv run python scripts/build.py --target C_skip_sim  # build a single variant + its chain
"""
from __future__ import annotations

import subprocess, sys, tomllib
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import banner  # noqa: E402

SCRIPT_ROOT = Path(__file__).resolve().parent / "pipeline"

STAGE_SPEC = {
    "0_export":      {"script": "export_onnx.py",   "has_source": False},
    "1_simplify":    {"script": "simplify.py",  "has_source": True},
    "1b_preopt":     {"script": "preopt.py",   "has_source": True},
    "2_quantize":    {"script": "quantize.py",  "has_source": True},
}


def build(target: str, model: str, variants: dict, built: set):
    if target in built:
        return
    v = variants[target]
    source = v.get("source")
    if source:
        if source not in variants:
            raise SystemExit(f"variant '{target}' depends on '{source}' which is not defined in pipeline.toml")
        build(source, model, variants, built)

    spec = STAGE_SPEC[v["stage"]]
    script = SCRIPT_ROOT / spec["script"]
    cmd = [sys.executable, str(script), "--model", model]
    if spec["has_source"]:
        cmd += ["--in", source, "--out", target]
    if target == "A":
        cmd += ["--dec-fix-len", "128"]
    print(f"  [{target}] build -> {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise SystemExit(f"build of {target} failed (exit {result.returncode})")
    built.add(target)


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


if __name__ == "__main__":
    main()