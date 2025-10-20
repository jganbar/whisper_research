"""Model components for Whisper decoder training."""

from .decoder_extractor import WhisperDecoderLM, extract_decoder, save_decoder, load_decoder

__all__ = [
    "WhisperDecoderLM",
    "extract_decoder",
    "save_decoder",
    "load_decoder",
]

