"""Pass 1: operator statistics  (STUB -- implement me).

CONTRACT
    run(model) -> dict
      model  : onnx.ModelProto (in memory) | str | pathlib.Path  (an .onnx file)
      return : {
          "model_path": str,
          "total_nodes": int,
          "ops": { op_type(str): count(int), ... },   # every node in graph.node
      }

PURPOSE (outline Pass 1)
    Run on the pre-optimized FP32 graph (variant B) to see the op distribution and
    predict where quantization Fusion / runtime bottlenecks will sit.

HINT
    import onnx
    if isinstance(model, (str, Path)): model = onnx.load(str(model))
    from collections import Counter
    ops = Counter(n.op_type for n in model.graph.node)
    return {"model_path": str(...), "total_nodes": len(model.graph.node), "ops": dict(ops)}
"""
from __future__ import annotations
from pathlib import Path


def run(model):
    raise NotImplementedError(
        "Pass 1 (op_stats) not implemented yet -- fill in scripts/passes/op_stats.py:run()"
    )