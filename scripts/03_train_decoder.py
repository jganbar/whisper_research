#!/usr/bin/env python3
"""
Script 3: Train Decoder on Azerbaijani Text

This script trains the extracted Whisper decoder on text data from HuggingFace.
The dataset should be pre-prepared and contain a 'text' column.
"""

import argparse
import logging
import yaml
import sys
import warnings
from pathlib import Path

# Suppress all warnings for cleaner output
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import load_decoder, DecoderTrainer, TrainingConfig
from src.data import load_text_dataset, create_dataloaders
from transformers import WhisperTokenizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Train Whisper decoder on text data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python scripts/03_train_decoder.py --device cuda
  python scripts/03_train_decoder.py --config configs/training_config.yaml --device cuda
        """
    )
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
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu", "mps"],
        help="Device to use for training"
    )
    args = parser.parse_args()
    
    # Load configuration
    logger.info("Loading configuration...")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info("=" * 70)
    logger.info("Step 3: Training Whisper Decoder on Text Data")
    logger.info("=" * 70)
    
    # Verify CUDA is available and working
    import torch
    if args.device == "cuda":
        if not torch.cuda.is_available():
            logger.error("❌ CUDA requested but not available!")
            logger.error("   Please check your PyTorch installation and GPU drivers")
            sys.exit(1)
        
        logger.info(f"\n🚀 CUDA Device Check:")
        logger.info(f"   CUDA Available: {torch.cuda.is_available()}")
        logger.info(f"   CUDA Device: {torch.cuda.get_device_name(0)}")
        logger.info(f"   CUDA Version: {torch.version.cuda}")
        logger.info(f"   PyTorch Version: {torch.__version__}")
        logger.info(f"   Device Count: {torch.cuda.device_count()}")
        
        # Set CUDA device if CUDA_VISIBLE_DEVICES is set
        import os
        if 'CUDA_VISIBLE_DEVICES' in os.environ:
            logger.info(f"   CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
    
    # Load decoder and tokenizer
    logger.info(f"\nLoading decoder from: {args.decoder_path}")
    decoder_lm, tokenizer = load_decoder(args.decoder_path, device=args.device)
    logger.info("✓ Decoder loaded successfully")
    logger.info(f"   Model on device: {next(decoder_lm.parameters()).device}")
    
    # Load dataset
    dataset_config = config['dataset']
    logger.info(f"\nLoading dataset: {dataset_config['name']}")
    dataset = load_text_dataset(
        dataset_name=dataset_config['name'],
        cache_dir=dataset_config.get('cache_dir'),
        streaming=dataset_config.get('streaming', False),
        split='train',
        text_column=dataset_config.get('text_column', 'text'),
    )
    logger.info("✓ Dataset loaded successfully")
    
    # Create dataloaders
    logger.info("\nCreating dataloaders...")
    dataloader_config = config['dataloader']
    train_dataloader, val_dataloader = create_dataloaders(
        dataset=dataset,
        tokenizer=tokenizer,
        text_column=dataset_config.get('text_column', 'text'),
        train_split=dataset_config.get('train_split', 0.95),
        val_split=dataset_config.get('val_split', 0.05),
        max_train_samples=dataset_config.get('max_train_samples', None),
        max_val_samples=dataset_config.get('max_val_samples', None),
        batch_size=dataloader_config['batch_size'],
        max_length=dataset_config.get('max_seq_length', 448),
        num_workers=dataloader_config.get('num_workers', 4),
        pin_memory=dataloader_config.get('pin_memory', True),
        seed=dataset_config.get('seed', 42),
    )
    logger.info("✓ Dataloaders created successfully")
    
    # Create training configuration
    logger.info("\nSetting up training configuration...")
    training_config = config['training']
    tensorboard_config = config.get('tensorboard', {})
    wandb_config = config.get('wandb', {})
    
    train_cfg = TrainingConfig(
        learning_rate=training_config['learning_rate'],
        weight_decay=training_config['weight_decay'],
        warmup_steps=training_config['warmup_steps'],
        num_epochs=training_config['num_epochs'],
        gradient_accumulation_steps=training_config['gradient_accumulation_steps'],
        max_grad_norm=training_config['max_grad_norm'],
        fp16=training_config.get('fp16', False),
        bf16=training_config.get('bf16', True),
        save_steps=training_config['save_steps'],
        eval_steps=training_config['eval_steps'],
        logging_steps=training_config['logging_steps'],
        output_dir=training_config['output_dir'],
        log_to_tensorboard=tensorboard_config.get('log_dir') is not None,
        tensorboard_dir=tensorboard_config.get('log_dir', './experiments/runs'),
        log_to_wandb=wandb_config.get('enabled', False),
        device=args.device,
        seed=config.get('seed', 42),
    )
    
    # Display training settings
    logger.info("\n" + "=" * 70)
    logger.info("Training Configuration:")
    logger.info("=" * 70)
    logger.info(f"  Device: {args.device}")
    logger.info(f"  Learning rate: {train_cfg.learning_rate}")
    logger.info(f"  Batch size: {dataloader_config['batch_size']}")
    logger.info(f"  Gradient accumulation: {train_cfg.gradient_accumulation_steps}")
    logger.info(f"  Effective batch size: {dataloader_config['batch_size'] * train_cfg.gradient_accumulation_steps}")
    logger.info(f"  Epochs: {train_cfg.num_epochs}")
    logger.info(f"  Mixed precision: {'bf16' if train_cfg.bf16 else 'fp16' if train_cfg.fp16 else 'fp32'}")
    logger.info(f"  Output directory: {train_cfg.output_dir}")
    logger.info(f"  TensorBoard: {train_cfg.log_to_tensorboard}")
    if train_cfg.log_to_tensorboard:
        logger.info(f"  TensorBoard dir: {train_cfg.tensorboard_dir}")
    logger.info("=" * 70)
    
    # Create trainer
    logger.info("\nInitializing trainer...")
    trainer = DecoderTrainer(
        model=decoder_lm,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        config=train_cfg,
    )
    logger.info("✓ Trainer initialized")
    
    # Start training
    logger.info("\n" + "=" * 70)
    logger.info("Starting Training")
    logger.info("=" * 70)
    
    if train_cfg.log_to_tensorboard:
        logger.info(f"\n💡 Monitor training with TensorBoard:")
        logger.info(f"   tensorboard --logdir {train_cfg.tensorboard_dir}")
        logger.info(f"   Then open http://localhost:6006 in your browser\n")
    
    try:
        trainer.train()
        
        logger.info("\n" + "=" * 70)
        logger.info("✓ Training Completed Successfully!")
        logger.info("=" * 70)
        logger.info(f"  Best validation loss: {trainer.best_val_loss:.4f}")
        logger.info(f"  Best model saved to: {train_cfg.output_dir}/best_model.pt")
        logger.info("=" * 70)
        
    except KeyboardInterrupt:
        logger.info("\n\nTraining interrupted by user")
        logger.info("Checkpoint saved, you can resume training later")
    except Exception as e:
        logger.error(f"\n\nTraining failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
