"""
Decoder Training Module

This module handles training the Whisper decoder on Azerbaijani text data.
"""

import logging
import math
import os
from typing import Optional
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
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
    log_to_tensorboard: bool = True
    tensorboard_dir: str = "./experiments/runs"
    log_to_wandb: bool = False
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
        self._resume_epoch = 0
        self._resume_batch = 0
        self._resuming = False
        
        os.makedirs(config.output_dir, exist_ok=True)
        
        # Initialize TensorBoard
        self.writer = None
        if config.log_to_tensorboard:
            self._init_tensorboard()
        
        # Initialize W&B (optional)
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
            total_steps = max(self.config.max_steps, 1)
        else:
            steps_per_epoch = math.ceil(len(self.train_dataloader) / self.config.gradient_accumulation_steps)
            total_steps = max(steps_per_epoch * self.config.num_epochs, 1)
        
        warmup_iters = max(0, self.config.warmup_steps)
        if warmup_iters >= total_steps:
            if warmup_iters > 0:
                logger.warning(
                    "Warmup steps (%s) >= total training steps (%s); using linear warmup only.",
                    warmup_iters,
                    total_steps,
                )
            warmup_iters = total_steps
        
        if warmup_iters == 0:
            return CosineAnnealingLR(self.optimizer, T_max=total_steps)
        
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_iters,
        )
        
        if warmup_iters == total_steps:
            return warmup_scheduler
        
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=max(total_steps - warmup_iters, 1),
        )
        
        return SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_iters],
        )

    def _move_optimizer_state_to_device(self):
        """Ensure optimizer state tensors reside on the target device."""
        device = torch.device(self.config.device)
        for state in self.optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(device)

    def load_training_state(self, checkpoint_path: str):
        """Load model, optimizer, and scheduler state from a checkpoint."""
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        logger.info(f"Loading training state from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.config.device)
        
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self._move_optimizer_state_to_device()
        if "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if hasattr(self.scheduler, "last_epoch"):
            self.scheduler.last_epoch = self.global_step
        if self.scaler is not None and checkpoint.get("scaler_state_dict") is not None:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        
        self.global_step = checkpoint.get("global_step", 0)
        self.epoch = checkpoint.get("epoch", 0)
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        
        batch_in_epoch = checkpoint.get("batch_in_epoch", -1)
        self._resume_epoch = min(self.epoch, self.config.num_epochs - 1)
        self._resume_batch = max(0, batch_in_epoch + 1)
        steps_per_epoch = len(self.train_dataloader)
        if steps_per_epoch > 0 and self._resume_batch >= steps_per_epoch:
            self._resume_batch = 0
            self._resume_epoch = min(self._resume_epoch + 1, self.config.num_epochs - 1)
        
        self.optimizer.zero_grad(set_to_none=True)
        self._resuming = True
        
        logger.info(
            "Checkpoint loaded (epoch=%s, batch=%s, global_step=%s, best_val_loss=%.4f)",
            self._resume_epoch,
            self._resume_batch,
            self.global_step,
            self.best_val_loss,
        )
    
    def _init_tensorboard(self):
        """Initialize TensorBoard logging."""
        try:
            log_dir = os.path.join(self.config.tensorboard_dir, "decoder_training")
            os.makedirs(log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=log_dir)
            logger.info(f"TensorBoard logging to {log_dir}")
            logger.info(f"View with: tensorboard --logdir {self.config.tensorboard_dir}")
        except Exception as e:
            logger.warning(f"Failed to initialize TensorBoard: {e}")
            self.config.log_to_tensorboard = False
    
    def _init_wandb(self):
        """Initialize Weights & Biases logging (optional)."""
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
        
        start_epoch = self._resume_epoch if self._resuming else 0
        
        for epoch in range(start_epoch, self.config.num_epochs):
            self.epoch = epoch
            logger.info(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")
            
            progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch + 1}")
            resume_batch = self._resume_batch if self._resuming and epoch == start_epoch else 0
            
            for step, batch in enumerate(progress_bar):
                if resume_batch and step < resume_batch:
                    continue
                
                batch = {k: v.to(self.config.device) for k, v in batch.items()}
                
                # Forward pass with mixed precision (updated to avoid FutureWarning)
                with torch.amp.autocast('cuda', enabled=self.config.fp16 or self.config.bf16, dtype=torch.bfloat16 if self.config.bf16 else torch.float16):
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
                        
                        # Log to TensorBoard
                        if self.writer is not None:
                            self.writer.add_scalar("train/loss", avg_loss, self.global_step)
                            self.writer.add_scalar("train/perplexity", perplexity, self.global_step)
                            self.writer.add_scalar("train/learning_rate", lr, self.global_step)
                            self.writer.add_scalar("train/epoch", epoch, self.global_step)
                        
                        # Log to W&B (optional)
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
                            self.save_checkpoint(batch_step=step, is_best=True)
                        
                        self.model.train()
                    
                    # Save checkpoint
                    if self.global_step % self.config.save_steps == 0:
                        self.save_checkpoint(batch_step=step)
                    
                    # Check max steps
                    if self.config.max_steps > 0 and self.global_step >= self.config.max_steps:
                        logger.info(f"Reached max steps: {self.config.max_steps}")
                        self._resuming = False
                        self._resume_batch = 0
                        return
            
            if self._resuming and epoch == start_epoch:
                self._resuming = False
                self._resume_batch = 0
        
        logger.info("Training completed!")
        self.save_checkpoint(is_final=True)
        
        # Close TensorBoard writer
        if self.writer is not None:
            self.writer.close()
    
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
        
        # Log to TensorBoard
        if self.writer is not None:
            self.writer.add_scalar("val/loss", avg_loss, self.global_step)
            self.writer.add_scalar("val/perplexity", perplexity, self.global_step)
        
        # Log to W&B (optional)
        if self.config.log_to_wandb:
            import wandb
            wandb.log({
                "val/loss": avg_loss,
                "val/perplexity": perplexity,
            })
        
        return avg_loss
    
    def save_checkpoint(self, batch_step: Optional[int] = None, is_best: bool = False, is_final: bool = False):
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
            "batch_in_epoch": batch_step if batch_step is not None else -1,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "scaler_state_dict": self.scaler.state_dict() if self.scaler is not None else None,
            "config": vars(self.config),
        }, checkpoint_path)
        
        logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        if not is_best and not is_final:
            self._cleanup_checkpoints()
    
    def _cleanup_checkpoints(self):
        """Remove old checkpoints to save disk space."""
        checkpoint_dir = self.config.output_dir
        if not os.path.exists(checkpoint_dir):
            return
        
        checkpoints = [
            f for f in os.listdir(checkpoint_dir)
            if f.startswith("checkpoint-") and f.endswith(".pt")
        ]
        
        if len(checkpoints) > self.config.save_total_limit:
            checkpoints.sort(key=lambda x: int(x.split("-")[1].split(".")[0]))
            
            for checkpoint in checkpoints[:-self.config.save_total_limit]:
                checkpoint_path = os.path.join(checkpoint_dir, checkpoint)
                os.remove(checkpoint_path)
