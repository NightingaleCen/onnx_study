"""analyze.py -- run the three analysis passes on a stage dir and report.

Passes live in scripts/passes/ as STUBS. You implement them; this driver tolerates
NotImplementedError (prints 'not implemented' and continues) so you can implement
passes one at a time and still get partial output.

    uv run python scripts/analyze.py --model STT --stage B
    uv run python scripts/analyze.py --model STT --stage C       (after stage3 also D_xnn, D_cpu)
    uv run python scripts/analyze.py --model STT --stage A --pass op_stats

Writes reports/analysis/<model>_<stage>.json and prints a console table.
"""
from __future__ import annotations

import argparse, importlib, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "passes"))
from common import stage_dir, list_onnx, REPORTS_DIR, banner  # noqa: E402

import onnx

PASSES = {
    "op_stats": "op_stats",
    "quant_coverage": "quant_coverage",
    "redundant_qdq": "redundant_qdq",
}


def run_pass(name, model_path):
    mod = importlib.import_module(PASSES[name])
    try:
        return mod.run(str(model_path))
    except NotImplementedError as e:
        return {"_status": "not_implemented", "msg": str(e)}
    except Exception as e:
        return {"_status": "error", "msg": f"{type(e).__name__}: {e}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="STT")
    ap.add_argument("--stage", default="B")
    ap.add_argument("--pass", dest="only", choices=list(PASSES), default=None)
    args = ap.parse_args()
    d = stage_dir(args.model, args.stage)
    if not list_onnx(d):
        print(f"no .onnx in {d}"); return
    banner(f"analyze {d}  (passes: {list(PASSES)})")
    passes = [args.only] if args.only else list(PASSES)
    agg = {"model": args.model, "stage": args.stage, "files": {}}
    from tabulate import tabulate
    for p in list_onnx(d):
        per = {}
        for pname in passes:
            per[pname] = run_pass(pname, p)
        agg["files"][p.name] = per
        rows = []
        for pname, res in per.items():
            if res.get("_status"):
                rows.append([pname, res["_status"], res.get("msg", "")])
            elif pname == "op_stats":
                ops = res.get("ops", {})
                top = ", ".join(f"{k}:{v}" for k, v in sorted(ops.items(), key=lambda x: -x[1])[:8])
                rows.append([pname, f"{res.get('total_nodes','?')} nodes", top])
            elif pname == "quant_coverage":
                rows.append([pname, f"{res.get('coverage_pct','-')}%",
                             f"cov={res.get('quantized')}/{res.get('total')} notq={len(res.get('not_quantized',[]))}"])
            elif pname == "redundant_qdq":
                rows.append([pname, f"{res.get('redundant_pairs',0)} pairs", ""])
        print(f"\n## {p.name}")
        print(tabulate(rows, headers=["pass", "summary", "detail"]))
    out = REPORTS_DIR / "analysis" / f"{args.model}_{args.stage}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(agg, indent=2, ensure_ascii=False, default=str))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()