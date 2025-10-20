#!/usr/bin/env python3
"""
Script 4: Integrate Fine-tuned Decoder

This script integrates the fine-tuned decoder back into Whisper.
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import (
    extract_decoder_from_checkpoint,
    integrate_decoder,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Integrate fine-tuned decoder")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to fine-tuned decoder checkpoint"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./experiments/whisper_integrated",
        help="Output directory for integrated model"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use"
    )
    args = parser.parse_args()
    
    logger.info("=" * 50)
    logger.info("Step 4: Integrating Fine-tuned Decoder")
    logger.info("=" * 50)
    
    # Load fine-tuned decoder from checkpoint
    finetuned_decoder = extract_decoder_from_checkpoint(
        checkpoint_path=args.checkpoint,
        device=args.device,
    )
    
    # Integrate into Whisper
    integrated_model = integrate_decoder(
        finetuned_decoder=finetuned_decoder,
        output_dir=args.output_dir,
        device=args.device,
    )
    
    logger.info(f"\n✓ Decoder integrated and saved to {args.output_dir}")


if __name__ == "__main__":
    main()

