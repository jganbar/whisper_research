#!/usr/bin/env python3
"""
Script 2: Prepare DOLLMA Dataset

This script loads and preprocesses the DOLLMA Azerbaijani dataset.
"""

import argparse
import logging
import yaml
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import load_dollma_dataset, prepare_dataset, get_dataset_statistics
from transformers import WhisperTokenizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Prepare DOLLMA dataset")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training_config.yaml",
        help="Path to training config file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./cache/processed_texts.pkl",
        help="Path to save processed texts"
    )
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    dataset_config = config['dataset']
    
    logger.info("=" * 50)
    logger.info("Step 2: Preparing DOLLMA Dataset")
    logger.info("=" * 50)
    
    # Load dataset
    dataset = load_dollma_dataset(
        dataset_name=dataset_config['name'],
        cache_dir=dataset_config.get('cache_dir'),
        streaming=dataset_config.get('streaming', False),
    )
    
    # Load tokenizer
    tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-large-v3")
    
    # Prepare dataset
    texts = prepare_dataset(
        dataset=dataset,
        tokenizer=tokenizer,
        min_length=dataset_config['preprocessing']['min_length'],
        max_length=dataset_config['preprocessing']['max_length'],
    )
    
    # Get statistics
    stats = get_dataset_statistics(texts, tokenizer)
    
    # Save processed texts
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(texts, f)
    
    logger.info(f"\n✓ Dataset prepared and saved to {output_path}")
    logger.info(f"  Total examples: {len(texts)}")


if __name__ == "__main__":
    main()

