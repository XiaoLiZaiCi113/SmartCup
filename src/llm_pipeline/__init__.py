"""Minimal COVFEE-to-LLM smart-cup experiment pipeline."""

from .preprocessing import RecordingConfig, build_manual_labels_from_covfee
from .prompts import write_prompt_bundle
from .evaluation import evaluate_llm_predictions

__all__ = [
    "RecordingConfig",
    "build_manual_labels_from_covfee",
    "write_prompt_bundle",
    "evaluate_llm_predictions",
]
