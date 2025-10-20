"""Evaluation and metrics modules."""

from .metrics import (
    calculate_wer,
    calculate_cer,
    calculate_detailed_metrics,
    create_comparison_report,
    bootstrap_confidence_interval,
    save_predictions,
)
from .asr_evaluation import (
    evaluate_asr,
    transcribe_audio_batch,
    load_evaluation_dataset,
)

__all__ = [
    "calculate_wer",
    "calculate_cer",
    "calculate_detailed_metrics",
    "create_comparison_report",
    "bootstrap_confidence_interval",
    "save_predictions",
    "evaluate_asr",
    "transcribe_audio_batch",
    "load_evaluation_dataset",
]

