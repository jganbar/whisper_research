"""
Decoder Training Module

This module handles training the Whisper decoder on Azerbaijani text data.
"""

import logging
import os
from typing import Optional
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from .decoder_extractor import WhisperDecoderLM

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for training."""
    
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    num_epochs: int = 3
    max_steps: int = -1
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    fp16: bool = False
    bf16: bool = True
    save_steps: int = 1000
    save_total_limit: int = 3
    output_dir: str = "./experiments/decoder_training"
    eval_steps: int = 500
    logging_steps: int = 100
    log_to_wandb: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42


class DecoderTrainer:
    """Trainer for Whisper Decoder Language Model."""
    
    def __init__(
        self,
        model: WhisperDecoderLM,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        config: TrainingConfig,
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config
        
        self._set_seed(config.seed)
        self.model = self.model.to(config.device)
        
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        self.scaler = None
        if config.fp16:
            self.scaler = torch.cuda.amp.GradScaler()
        
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        
        os.makedirs(config.output_dir, exist_ok=True)
        
        if config.log_to_wandb:
            self._init_wandb()
        
        logger.info(f"Trainer initialized. Device: {config.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    def _set_seed(self, seed: int):
        """Set random seed for reproducibility."""
        import random
        import numpy as np
        
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    def _create_optimizer(self) -> AdamW:
        """Create optimizer."""
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        
        return AdamW(optimizer_grouped_parameters, lr=self.config.learning_rate)
    
    def _create_scheduler(self):
        """Create learning rate scheduler with warmup."""
        if self.config.max_steps > 0:
            total_steps = self.config.max_steps
        else:
            total_steps = len(self.train_dataloader) * self.config.num_epochs // self.config.gradient_accumulation_steps
        
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=self.config.warmup_steps,
        )
        
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps - self.config.warmup_steps,
        )
        
        return SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[self.config.warmup_steps],
        )
    
    def _init_wandb(self):
        """Initialize Weights & Biases logging."""
        try:
            import wandb
            wandb.init(
                project="whisper-decoder-azerbaijani",
                config=vars(self.config),
            )
            wandb.watch(self.model, log="all", log_freq=self.config.logging_steps)
        except Exception as e:
            logger.warning(f"Failed to initialize wandb: {e}")
            self.config.log_to_wandb = False
    
    def train(self):
        """Main training loop."""
        logger.info("Starting training...")
        
        self.model.train()
        total_loss = 0
        
        for epoch in range(self.config.num_epochs):
            self.epoch = epoch
            logger.info(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")
            
            progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch + 1}")
            
            for step, batch in enumerate(progress_bar):
                batch = {k: v.to(self.config.device) for k, v in batch.items()}
                
                # Forward pass with mixed precision
                with torch.cuda.amp.autocast(enabled=self.config.fp16 or self.config.bf16, dtype=torch.bfloat16 if self.config.bf16 else torch.float16):
                    outputs = self.model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"],
                    )
                    
                    loss = outputs["loss"]
                    loss = loss / self.config.gradient_accumulation_steps
                
                # Backward pass
                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                total_loss += loss.item()
                
                # Gradient accumulation
                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    if self.scaler is not None:
                        self.scaler.unscale_(self.optimizer)
                    
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    
                    if self.scaler is not None:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    
                    self.global_step += 1
                    
                    # Logging
                    if self.global_step % self.config.logging_steps == 0:
                        avg_loss = total_loss / self.config.logging_steps
                        perplexity = torch.exp(torch.tensor(avg_loss)).item()
                        lr = self.scheduler.get_last_lr()[0]
                        
                        progress_bar.set_postfix({
                            "loss": f"{avg_loss:.4f}",
                            "ppl": f"{perplexity:.2f}",
                            "lr": f"{lr:.2e}",
                        })
                        
                        if self.config.log_to_wandb:
                            import wandb
                            wandb.log({
                                "train/loss": avg_loss,
                                "train/perplexity": perplexity,
                                "train/learning_rate": lr,
                                "train/global_step": self.global_step,
                            })
                        
                        total_loss = 0
                    
                    # Evaluation
                    if self.global_step % self.config.eval_steps == 0:
                        val_loss = self.evaluate()
                        
                        if val_loss < self.best_val_loss:
                            self.best_val_loss = val_loss
                            self.save_checkpoint(is_best=True)
                        
                        self.model.train()
                    
                    # Save checkpoint
                    if self.global_step % self.config.save_steps == 0:
                        self.save_checkpoint()
                    
                    # Check max steps
                    if self.config.max_steps > 0 and self.global_step >= self.config.max_steps:
                        logger.info(f"Reached max steps: {self.config.max_steps}")
                        return
        
        logger.info("Training completed!")
        self.save_checkpoint(is_final=True)
    
    def evaluate(self) -> float:
        """Evaluate on validation set."""
        logger.info("Running evaluation...")
        
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Evaluating"):
                batch = {k: v.to(self.config.device) for k, v in batch.items()}
                
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                
                total_loss += outputs["loss"].item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        logger.info(f"Validation loss: {avg_loss:.4f}, Perplexity: {perplexity:.2f}")
        
        if self.config.log_to_wandb:
            import wandb
            wandb.log({
                "val/loss": avg_loss,
                "val/perplexity": perplexity,
            })
        
        return avg_loss
    
    def save_checkpoint(self, is_best: bool = False, is_final: bool = False):
        """Save model checkpoint."""
        checkpoint_dir = self.config.output_dir
        
        if is_best:
            checkpoint_path = os.path.join(checkpoint_dir, "best_model.pt")
        elif is_final:
            checkpoint_path = os.path.join(checkpoint_dir, "final_model.pt")
        else:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint-{self.global_step}.pt")
        
        torch.save({
            "global_step": self.global_step,
            "epoch": self.epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "config": vars(self.config),
        }, checkpoint_path)
        
        logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        if not is_best and not is_final:
            self._cleanup_checkpoints()
    
    def _cleanup_checkpoints(self):
        """Remove old checkpoints to save disk space."""
        checkpoint_dir = self.config.output_dir
        checkpoints = [
            f for f in os.listdir(checkpoint_dir)
            if f.startswith("checkpoint-") and f.endswith(".pt")
        ]
        
        if len(checkpoints) > self.config.save_total_limit:
            checkpoints.sort(key=lambda x: int(x.split("-")[1].split(".")[0]))
            
            for checkpoint in checkpoints[:-self.config.save_total_limit]:
                checkpoint_path = os.path.join(checkpoint_dir, checkpoint)
                os.remove(checkpoint_path)

