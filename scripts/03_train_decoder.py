#!/usr/bin/env python3
"""
Script 3: Train Decoder on Azerbaijani Text

This script trains the extracted decoder on DOLLMA dataset.
"""

import argparse
import logging
import yaml
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import load_decoder, DecoderTrainer, TrainingConfig
from src.data import create_dataloaders
from transformers import WhisperTokenizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train Whisper decoder")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training_config.yaml",
        help="Path to training config file"
    )
    parser.add_argument(
        "--decoder_path",
        type=str,
        default="./experiments/decoder_extracted",
        help="Path to extracted decoder"
    )
    parser.add_argument(
        "--texts_path",
        type=str,
        default="./cache/processed_texts.pkl",
        help="Path to processed texts"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use"
    )
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info("=" * 50)
    logger.info("Step 3: Training Decoder")
    logger.info("=" * 50)
    
    # Load decoder
    decoder_lm, tokenizer = load_decoder(args.decoder_path, device=args.device)
    
    # Load processed texts
    with open(args.texts_path, 'rb') as f:
        texts = pickle.load(f)
    
    logger.info(f"Loaded {len(texts)} text examples")
    
    # Create dataloaders
    dataloader_config = config['dataloader']
    train_dataloader, val_dataloader = create_dataloaders(
        texts=texts,
        tokenizer=tokenizer,
        train_split=config['dataset']['train_split'],
        batch_size=dataloader_config['batch_size'],
        num_workers=dataloader_config['num_workers'],
        seed=config['seed'],
    )
    
    # Create training config
    training_config = config['training']
    train_cfg = TrainingConfig(
        learning_rate=training_config['learning_rate'],
        weight_decay=training_config['weight_decay'],
        warmup_steps=training_config['warmup_steps'],
        num_epochs=training_config['num_epochs'],
        gradient_accumulation_steps=training_config['gradient_accumulation_steps'],
        max_grad_norm=training_config['max_grad_norm'],
        fp16=training_config['fp16'],
        bf16=training_config['bf16'],
        save_steps=training_config['save_steps'],
        eval_steps=training_config['eval_steps'],
        logging_steps=training_config['logging_steps'],
        output_dir=training_config['output_dir'],
        log_to_wandb=config.get('wandb', {}).get('project') is not None,
        device=args.device,
        seed=config['seed'],
    )
    
    # Create trainer
    trainer = DecoderTrainer(
        model=decoder_lm,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        config=train_cfg,
    )
    
    # Train
    trainer.train()
    
    logger.info("\n✓ Training completed!")
    logger.info(f"  Best validation loss: {trainer.best_val_loss:.4f}")


if __name__ == "__main__":
    main()

