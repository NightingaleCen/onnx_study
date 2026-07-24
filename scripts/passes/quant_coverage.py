"""Pass 2: quantization coverage  (STUB -- implement me).

CONTRACT
    run(model) -> dict
      model  : onnx.ModelProto | str | pathlib.Path   (a quantized QDQ model: C or D)
      return : {
          "model_path": str,
          "total": int,                 # non-Q/DQ compute nodes
          "quantized": int,             # fully wrapped by Q/DQ
          "partial": int,               # some tensors (un)quantized
          "coverage_pct": float,        # quantized/total*100
          "not_quantized": [ { "name", "op_type" }, ... ],
      }

PURPOSE (outline Pass 2)
    For every non-QuantizeLinear/DequantizeLinear compute node, check whether all
    its inputs originate from a DequantizeLinear (or are weights/constants that are
    quantized) and all its outputs flow toward a QuantizeLinear. A node is
    "quantized" iff all relevant tensors are DQ-fed / Q-sunk. Compare on C vs D_xnn
    to see what runtime EP fusion actually wrapped.

HINT
    Build producer map: tensor_name -> node that produced it (walk graph.node).
    For each compute node, inspect inputs[i] against the map.
"""
from __future__ import annotations
from pathlib import Path


def run(model):
    raise NotImplementedError(
        "Pass 2 (quant_coverage) not implemented yet -- fill in scripts/passes/quant_coverage.py:run()"
    )