#!/usr/bin/env python3
"""
Script 1 (Alternative): Extract Decoder from Whisper Large v3 using OpenAI Whisper

This script uses the openai-whisper library instead of transformers to avoid
HuggingFace Hub download issues. It extracts the decoder and converts it to
a format compatible with transformers.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import torch
import whisper
from transformers import WhisperTokenizer, WhisperConfig

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.decoder_extractor import WhisperDecoderLM

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def extract_decoder_from_openai_whisper(
    model_size: str = "large-v3",
    download_root: str = "./cache/whisper_models",
    device: str = "cuda",
) -> tuple:
    """
    Extract decoder from OpenAI Whisper model.
    
    Args:
        model_size: Size of Whisper model (tiny, base, small, medium, large, large-v2, large-v3)
        download_root: Directory to download/cache the model
        device: Device to load model on
        
    Returns:
        Tuple of (WhisperDecoderLM, tokenizer, config)
    """
    logger.info(f"Loading OpenAI Whisper model: {model_size}")
    
    # Load the OpenAI Whisper model
    openai_model = whisper.load_model(model_size, download_root=download_root, device=device)
    
    # Extract decoder components
    decoder = openai_model.decoder
    embed_tokens = openai_model.decoder.token_embedding
    embed_positions = openai_model.decoder.positional_embedding
    
    # Create Whisper config from model dimensions
    dims = openai_model.dims
    config = WhisperConfig(
        vocab_size=dims.n_vocab,
        d_model=dims.n_text_state,
        encoder_attention_heads=dims.n_audio_head,
        decoder_attention_heads=dims.n_text_head,
        encoder_layers=dims.n_audio_layer,
        decoder_layers=dims.n_text_layer,
        encoder_ffn_dim=dims.n_text_state * 4,
        decoder_ffn_dim=dims.n_text_state * 4,
        max_source_positions=dims.n_audio_ctx,
        max_target_positions=dims.n_text_ctx,
    )
    
    logger.info(f"Model dimensions:")
    logger.info(f"  Vocab size: {dims.n_vocab}")
    logger.info(f"  Hidden size: {dims.n_text_state}")
    logger.info(f"  Decoder layers: {dims.n_text_layer}")
    logger.info(f"  Attention heads: {dims.n_text_head}")
    
    # Create standalone decoder LM
    decoder_lm = WhisperDecoderLM(
        decoder=decoder,
        embed_tokens=embed_tokens,
        embed_positions=embed_positions,
        config=config,
    )
    
    # Load tokenizer from transformers (compatible)
    logger.info("Loading tokenizer from transformers...")
    tokenizer = WhisperTokenizer.from_pretrained(
        "openai/whisper-large-v3",
        cache_dir="./cache/tokenizers"
    )
    
    logger.info(f"Successfully extracted decoder with {decoder_lm.num_parameters():,} parameters")
    
    return decoder_lm, tokenizer, config


def save_decoder(
    model: WhisperDecoderLM,
    tokenizer: WhisperTokenizer,
    config: WhisperConfig,
    output_dir: str,
):
    """
    Save the decoder model, tokenizer, and config.
    
    Args:
        model: WhisperDecoderLM model
        tokenizer: Whisper tokenizer
        config: Whisper configuration
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model state dict
    model_path = os.path.join(output_dir, "decoder_lm.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config.to_dict(),
    }, model_path)
    
    # Save config separately
    config.save_pretrained(output_dir)
    
    # Save tokenizer
    tokenizer.save_pretrained(output_dir)
    
    logger.info(f"✓ Saved decoder model to {output_dir}")
    logger.info(f"  - Model weights: {model_path}")
    logger.info(f"  - Config: {output_dir}/config.json")
    logger.info(f"  - Tokenizer: {output_dir}/tokenizer.json")


def main():
    parser = argparse.ArgumentParser(
        description="Extract Whisper decoder using OpenAI Whisper library"
    )
    parser.add_argument(
        "--model-size",
        type=str,
        default="large-v3",
        choices=["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"],
        help="Whisper model size"
    )
    parser.add_argument(
        "--download-root",
        type=str,
        default="./cache/whisper_models",
        help="Directory to cache downloaded models"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./experiments/decoder_extracted",
        help="Output directory for extracted decoder"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda/cpu)"
    )
    args = parser.parse_args()
    
    logger.info("=" * 70)
    logger.info("Step 1: Extracting Whisper Decoder (OpenAI Whisper Method)")
    logger.info("=" * 70)
    logger.info(f"Model size: {args.model_size}")
    logger.info(f"Device: {args.device}")
    logger.info("")
    
    # Extract decoder
    decoder_lm, tokenizer, config = extract_decoder_from_openai_whisper(
        model_size=args.model_size,
        download_root=args.download_root,
        device=args.device,
    )
    
    # Save decoder
    save_decoder(decoder_lm, tokenizer, config, args.output_dir)
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("✓ Decoder extraction completed successfully!")
    logger.info("=" * 70)
    logger.info(f"Total parameters: {decoder_lm.num_parameters():,}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info("")
    logger.info("Next step: Run 02_prepare_data.py to prepare the DOLLMA dataset")


if __name__ == "__main__":
    main()

