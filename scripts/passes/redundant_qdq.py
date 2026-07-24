"""Pass 3: redundant Q/DQ detection  (STUB -- implement me).

CONTRACT
    run(model) -> dict
      model  : onnx.ModelProto | str | pathlib.Path   (a quantized QDQ model: C or D)
      return : {
          "model_path": str,
          "redundant_pairs": int,
          "pairs": [ {
              "q": "QuantizeLinear node name",
              "dq": "DequantizeLinear node name",
              "tensor": "shared tensor name (Q output == DQ input)",
              "scale_equal": bool, "zp_equal": bool,
          }, ... ],
      }

PURPOSE (outline Pass 3)
    Find back-to-back QuantizeLinear -> DequantizeLinear pairs whose scale and
    zero_point match exactly (a no-op round-trip left by the static quantizer).
    These are candidates for stage5_manual_edit.py removal / graph rewriting.
    Compare counts on C vs D_xnn: QDQFinalCleanupTransformer should drop some.

HINT
    Iterate nodes; when a QuantizeLinear's output feeds ONLY a DequantizeLinear,
    compare their scale/zero_point initializers (graph.initializer or Constant ops).
"""
from __future__ import annotations
from pathlib import Path


def run(model):
    raise NotImplementedError(
        "Pass 3 (redundant_qdq) not implemented yet -- fill in scripts/passes/redundant_qdq.py:run()"
    )