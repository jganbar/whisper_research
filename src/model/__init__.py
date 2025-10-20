"""Model components for Whisper decoder training."""

from .decoder_extractor import WhisperDecoderLM, extract_decoder, save_decoder, load_decoder
from .decoder_trainer import DecoderTrainer, TrainingConfig

__all__ = [
    "WhisperDecoderLM",
    "extract_decoder",
    "save_decoder",
    "load_decoder",
    "DecoderTrainer",
    "TrainingConfig",
]

