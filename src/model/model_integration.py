"""
Model Integration Module

This module handles integrating the fine-tuned decoder back into the Whisper model.
"""

import logging
import os
from typing import Optional, Dict, Any

import torch
from transformers import (
    WhisperForConditionalGeneration,
    WhisperTokenizer,
    WhisperProcessor,
)

from .decoder_extractor import WhisperDecoderLM

logger = logging.getLogger(__name__)


def integrate_decoder(
    finetuned_decoder: WhisperDecoderLM,
    base_model_name: str = "openai/whisper-large-v3",
    output_dir: Optional[str] = None,
    device: str = "cpu",
) -> WhisperForConditionalGeneration:
    """
    Integrate fine-tuned decoder back into Whisper model.
    
    Args:
        finetuned_decoder: Fine-tuned WhisperDecoderLM
        base_model_name: Base Whisper model name
        output_dir: Directory to save integrated model
        device: Device to load model on
        
    Returns:
        Integrated WhisperForConditionalGeneration model
    """
    logger.info(f"Loading base Whisper model: {base_model_name}")
    
    whisper_model = WhisperForConditionalGeneration.from_pretrained(
        base_model_name,
    ).to(device)
    
    logger.info("Integrating fine-tuned decoder...")
    
    # Replace decoder weights
    whisper_model.model.decoder = finetuned_decoder.decoder
    whisper_model.model.decoder.embed_tokens = finetuned_decoder.embed_tokens
    whisper_model.model.decoder.embed_positions = finetuned_decoder.embed_positions
    
    # Update projection head
    if hasattr(whisper_model, "proj_out"):
        whisper_model.proj_out.weight = finetuned_decoder.lm_head.weight
    
    logger.info("Decoder integration completed")
    
    # Verify integration
    verify_integration(whisper_model, device)
    
    # Save integrated model
    if output_dir:
        save_integrated_model(whisper_model, output_dir)
    
    return whisper_model


def verify_integration(
    model: WhisperForConditionalGeneration,
    device: str = "cpu",
) -> bool:
    """
    Verify that the integrated model works correctly.
    
    Args:
        model: Integrated Whisper model
        device: Device to run verification on
        
    Returns:
        True if verification passes
    """
    logger.info("Verifying integrated model...")
    
    try:
        batch_size = 2
        seq_len = 100
        audio_seq_len = 1500
        
        # Dummy audio features (encoder input)
        input_features = torch.randn(
            batch_size, 128, audio_seq_len, device=device
        )
        
        # Dummy decoder input
        decoder_input_ids = torch.randint(
            0, model.config.vocab_size, (batch_size, seq_len), device=device
        )
        
        # Forward pass
        model.eval()
        with torch.no_grad():
            outputs = model(
                input_features=input_features,
                decoder_input_ids=decoder_input_ids,
            )
        
        # Check output shape
        expected_shape = (batch_size, seq_len, model.config.vocab_size)
        actual_shape = outputs.logits.shape
        
        if actual_shape != expected_shape:
            logger.error(f"Output shape mismatch: expected {expected_shape}, got {actual_shape}")
            return False
        
        logger.info("✓ Model verification passed")
        logger.info(f"  - Output shape: {actual_shape}")
        
        return True
    
    except Exception as e:
        logger.error(f"Model verification failed: {e}")
        return False


def save_integrated_model(
    model: WhisperForConditionalGeneration,
    output_dir: str,
) -> None:
    """
    Save the integrated Whisper model.
    
    Args:
        model: Integrated Whisper model
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Saving integrated model to {output_dir}")
    
    # Save model
    model.save_pretrained(output_dir)
    
    # Also save processor/tokenizer
    try:
        processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
        processor.save_pretrained(output_dir)
    except Exception as e:
        logger.warning(f"Failed to save processor: {e}")
    
    logger.info(f"Model saved successfully to {output_dir}")


def load_integrated_model(
    model_path: str,
    device: str = "cpu",
) -> tuple[WhisperForConditionalGeneration, WhisperProcessor]:
    """
    Load an integrated Whisper model.
    
    Args:
        model_path: Path to the saved model
        device: Device to load model on
        
    Returns:
        Tuple of (model, processor)
    """
    logger.info(f"Loading integrated model from {model_path}")
    
    model = WhisperForConditionalGeneration.from_pretrained(
        model_path,
    ).to(device)
    
    try:
        processor = WhisperProcessor.from_pretrained(model_path)
    except:
        logger.warning("Processor not found, using default")
        processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
    
    logger.info("Model loaded successfully")
    
    return model, processor


def extract_decoder_from_checkpoint(
    checkpoint_path: str,
    base_model_name: str = "openai/whisper-large-v3",
    device: str = "cpu",
) -> WhisperDecoderLM:
    """
    Extract decoder from a training checkpoint.
    
    Args:
        checkpoint_path: Path to training checkpoint
        base_model_name: Base Whisper model name
        device: Device to load on
        
    Returns:
        WhisperDecoderLM
    """
    from .decoder_extractor import extract_decoder
    
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract base decoder structure
    decoder_lm, _ = extract_decoder(base_model_name, device=device)
    
    # Load fine-tuned weights
    decoder_lm.load_state_dict(checkpoint["model_state_dict"])
    
    logger.info("Decoder extracted from checkpoint")
    
    return decoder_lm

