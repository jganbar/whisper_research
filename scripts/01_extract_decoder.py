#!/usr/bin/env python3
"""
Script 1: Extract Decoder from Whisper Large v3

This script extracts the decoder component from Whisper Large v3
and saves it as a standalone language model.
"""

import argparse
import logging
import yaml
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import extract_decoder, save_decoder

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Extract Whisper decoder")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/model_config.yaml",
        help="Path to model config file"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda/cpu)"
    )
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    model_config = config['base_model']
    save_config = config['save']
    
    logger.info("=" * 50)
    logger.info("Step 1: Extracting Whisper Decoder")
    logger.info("=" * 50)
    
    # Extract decoder
    decoder_lm, tokenizer = extract_decoder(
        model_name=model_config['name'],
        cache_dir=model_config.get('cache_dir'),
        device=args.device,
    )
    
    # Save decoder
    output_dir = save_config['output_dir']
    save_decoder(decoder_lm, tokenizer, output_dir)
    
    logger.info(f"\n✓ Decoder extracted and saved to {output_dir}")
    logger.info(f"  Model parameters: {decoder_lm.num_parameters():,}")


if __name__ == "__main__":
    main()

