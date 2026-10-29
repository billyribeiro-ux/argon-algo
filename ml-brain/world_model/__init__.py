"""
Self-Supervised Financial World Model (`ml-brain/world_model`).

Contains:
- `rust_bridge`: Zero training-serving skew FFI bridge to `crates/raw-fingerprint-core`.
- `tokenizer`: Candlestick & Flow Tokenizer + Multi-Horizon Reverse-Trace Rewind (Polars 2.0 `maintain_order=True`).
- `sequence_backbone`: Mamba-SSM + FlexAttention sequence encoder (PyTorch 2.15 MPS FP8 / CUDA / Native).
- `explainer_and_onnx`: LightGBM + TreeSHAP 4-Stream cause-and-effect explainer & ONNX exporter.
"""

from .explainer_and_onnx import (
    FEATURE_NAMES_32,
    OnnxBrainArtifactExporter,
    OnnxBrainRuntime,
    TreeShapRawExplainer,
)
from .rust_bridge import (
    FINGERPRINT_FEATURE_DIM,
    SUB_PENNY_SCALE,
    TRACE_EMBEDDING_DIM,
    RawBarInput,
    RawMarketInternalsInput,
    RawOptionsInput,
    RawShortFloatInput,
    RustFingerprintBridge,
)
from .sequence_backbone import MambaFlexWorldModelBackbone, WorldModelPrediction
from .tokenizer import CandlestickFlowTokenizer, ReverseTraceSample

__all__ = [
    "FINGERPRINT_FEATURE_DIM",
    "SUB_PENNY_SCALE",
    "TRACE_EMBEDDING_DIM",
    "FEATURE_NAMES_32",
    "RawBarInput",
    "RawOptionsInput",
    "RawShortFloatInput",
    "RawMarketInternalsInput",
    "RustFingerprintBridge",
    "CandlestickFlowTokenizer",
    "ReverseTraceSample",
    "MambaFlexWorldModelBackbone",
    "WorldModelPrediction",
    "TreeShapRawExplainer",
    "OnnxBrainArtifactExporter",
    "OnnxBrainRuntime",
]

