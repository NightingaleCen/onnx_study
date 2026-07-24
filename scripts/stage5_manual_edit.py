"""Stage 5: D -> D_manual  (optional hand-editing of redundant Q/DQ pairs).

A small rewire helper. Given explicit QuantizeLinear/DequantizeLinear node-name
pairs (locate them in Netron, or from Pass 3 once you implement it), this removes
each pair and reconnects downstream consumers of the DQ's output to the Q's input
(valid because identical scale/zp means DQ(Q(x)) == x). onnx.checker validates.

You can also edit graphs by hand in Netron/another tool and just register the
result in variants.json -- bench.py will pick it up either way.

    uv run python scripts/stage5_manual_edit.py --model STT --in C \
        --pairs "QuantizeLinear_x,DequantizeLinear_x;QuantizeLinear_y,DequantizeLinear_y"
"""
from __future__ import annotations

import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import stage_dir, list_onnx, write_manifest, upsert_variant, ensure_empty_dir, banner  # noqa: E402

import onnx


def remove_qdq_pair(model, q_name, dq_name):
    # map node name -> node, output name -> producing node
    by_name = {n.name: n for n in model.graph.node}
    q = by_name.get(q_name); dq = by_name.get(dq_name)
    if q is None or dq is None or q.op_type != "QuantizeLinear" or dq.op_type != "DequantizeLinear":
        return False
    q_in = q.input[0]            # the original tensor that was quantized
    dq_out = dq.output[0]        # what consumers currently read
    # rewire all consumers of dq_out -> q_in
    for n in model.graph.node:
        n.input[:] = [q_in if x == dq_out else x for x in n.input]
    # also fix graph outputs
    for o in model.graph.output:
        if o.name == dq_out:
            o.name = q_in
    # delete the two nodes
    keep = [n for n in model.graph.node if n.name not in (q_name, dq_name)]
    del model.graph.node[:]
    model.graph.node.extend(keep)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="STT")
    ap.add_argument("--in", dest="inp", default="C")
    ap.add_argument("--out", dest="out", default="D_manual")
    ap.add_argument("--pairs", default="",
                    help="semicolon-sep 'qname,dqname' pairs to remove")
    ap.add_argument("--pairs-file", default=None,
                    help="JSON from Pass 3 with pairs list (alternative to --pairs)")
    args = ap.parse_args()
    in_dir = stage_dir(args.model, args.inp)
    out_dir = stage_dir(args.model, args.out)
    ensure_empty_dir(out_dir)

    pairs = []
    if args.pairs:
        for item in args.pairs.split(";"):
            q, dq = item.split(",")
            pairs.append((q.strip(), dq.strip()))
    if args.pairs_file:
        import json
        data = json.loads(Path(args.pairs_file).read_text())
        for pr in data.get("pairs", []):
            pairs.append((pr["q"], pr["dq"]))
    banner(f"manual edit  {in_dir} -> {out_dir}  ({len(pairs)} pairs)")

    for p in list_onnx(in_dir):
        m = onnx.load(str(p))
        removed = 0
        for qn, dn in pairs:
            if remove_qdq_pair(m, qn, dn):
                removed += 1
        onnx.save(m, str(out_dir / p.name))
        onnx.checker.check_model(str(out_dir / p.name))
        print(f"  {p.name}: removed {removed}/{len(pairs)} pairs; checker OK")
    write_manifest(args.model, args.out, created_by="stage5_manual_edit.py",
                   args={"from": args.inp, "pairs": pairs}, extra={"dtype": "int8"})
    upsert_variant(args.model, args.out, path=args.out, dtype="int8",
                   note=f"manual QDQ removal from {args.inp}", created_by="stage5_manual_edit.py",
                   base=args.inp)



if __name__ == "__main__":
    main()