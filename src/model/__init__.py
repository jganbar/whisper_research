"""Model components for Whisper decoder training."""

from .decoder_extractor import WhisperDecoderLM, extract_decoder, save_decoder, load_decoder
from .decoder_trainer import DecoderTrainer, TrainingConfig
from .model_integration import (
    integrate_decoder,
    verify_integration,
    save_integrated_model,
    load_integrated_model,
    extract_decoder_from_checkpoint,
)

__all__ = [
    "WhisperDecoderLM",
    "extract_decoder",
    "save_decoder",
    "load_decoder",
    "DecoderTrainer",
    "TrainingConfig",
    "integrate_decoder",
    "verify_integration",
    "save_integrated_model",
    "load_integrated_model",
    "extract_decoder_from_checkpoint",
]

