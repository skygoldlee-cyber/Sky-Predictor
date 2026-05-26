"""Prediction package.

This package contains the current recommended implementation of the
Transformer(+placeholder) + LLM pipeline.

Public API:
- `PredictionPipeline`
"""

from .pipeline import PredictionPipeline

# 하위 호환을 위한 re-export (DESIGN-1 수정)
from .features import ADAPT_KEYS
from .features.option_features import calc_oi_levels, build_option_snapshot

__all__ = [
    "PredictionPipeline",
    "ADAPT_KEYS",
    "calc_oi_levels",
    "build_option_snapshot",
]
