"""
Dataset Loading Module

This module handles loading datasets from HuggingFace Hub and creating
PyTorch DataLoaders for training the Whisper decoder.
"""

import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset, Dataset as HFDataset
from transformers import WhisperTokenizer

logger = logging.getLogger(__name__)


@dataclass
class DataCollator:
    """Data collator for causal language modeling."""
    
    tokenizer: WhisperTokenizer
    max_length: int = 448
    
    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of examples.
        
        Args:
            examples: List of examples with 'input_ids'
            
        Returns:
            Batched tensors with input_ids, attention_mask, and labels
        """
        input_ids = [ex["input_ids"] for ex in examples]
        
        # Pad sequences
        padded = self.tokenizer.pad(
            {"input_ids": input_ids},
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        
        # For causal LM, labels are the same as input_ids
        # Ignore padding tokens in loss computation
        labels = padded["input_ids"].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            "input_ids": padded["input_ids"],
            "attention_mask": padded["attention_mask"],
            "labels": labels,
        }


class TextDataset(Dataset):
    """PyTorch Dataset for text data."""
    
    def __init__(
        self,
        texts: List[str],
        tokenizer: WhisperTokenizer,
        max_length: int = 448,
    ):
        """
        Initialize the dataset.
        
        Args:
            texts: List of text strings
            tokenizer: Whisper tokenizer
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single tokenized example."""
        text = self.texts[idx]
        
        # Tokenize text
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
        }


def load_text_dataset(
    dataset_name: str,
    cache_dir: Optional[str] = None,
    streaming: bool = False,
    split: str = "train",
    text_column: str = "text",
) -> HFDataset:
    """
    Load a text dataset from HuggingFace Hub.
    
    Args:
        dataset_name: Name of the dataset on HuggingFace Hub (e.g., "username/dataset")
        cache_dir: Directory to cache the dataset
        streaming: Whether to use streaming mode
        split: Dataset split to load (e.g., "train", "test")
        text_column: Name of the text column in the dataset
        
    Returns:
        Loaded HuggingFace dataset
        
    Example:
        >>> dataset = load_text_dataset("username/azerbaijani-text", text_column="text")
    """
    logger.info(f"Loading dataset: {dataset_name}")
    logger.info(f"  Split: {split}")
    logger.info(f"  Text column: {text_column}")
    
    try:
        dataset = load_dataset(
            dataset_name,
            cache_dir=cache_dir,
            streaming=streaming,
            split=split,
        )
        
        if not streaming:
            logger.info(f"  Loaded {len(dataset)} examples")
        
        # Verify text column exists
        if not streaming:
            if text_column not in dataset.column_names:
                raise ValueError(
                    f"Text column '{text_column}' not found in dataset. "
                    f"Available columns: {dataset.column_names}"
                )
        
        return dataset
    
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise


def prepare_texts(
    dataset: HFDataset,
    text_column: str = "text",
    max_seq_length: int = 448,
    tokenizer: Optional[WhisperTokenizer] = None,
) -> List[str]:
    """
    Extract and prepare texts from the dataset.
    
    Args:
        dataset: HuggingFace dataset
        text_column: Name of the text column
        max_seq_length: Maximum sequence length (for filtering)
        tokenizer: Optional tokenizer for length checking
        
    Returns:
        List of text strings
    """
    logger.info("Extracting texts from dataset...")
    
    texts = []
    skipped = 0
    
    for example in dataset:
        text = example[text_column]
        
        # Basic validation
        if not text or not isinstance(text, str):
            skipped += 1
            continue
        
        # Strip whitespace
        text = text.strip()
        
        if len(text) == 0:
            skipped += 1
            continue
        
        # Optional: Check tokenized length if tokenizer provided
        if tokenizer is not None:
            tokens = tokenizer.encode(text)
            if len(tokens) > max_seq_length:
                # Truncate text approximately
                approx_chars = int(len(text) * (max_seq_length / len(tokens)))
                text = text[:approx_chars]
        
        texts.append(text)
    
    logger.info(f"Extracted {len(texts)} valid texts")
    if skipped > 0:
        logger.info(f"Skipped {skipped} invalid examples")
    
    return texts


def create_dataloaders(
    dataset: HFDataset,
    tokenizer: WhisperTokenizer,
    text_column: str = "text",
    train_split: float = 0.95,
    batch_size: int = 32,
    max_length: int = 448,
    num_workers: int = 4,
    pin_memory: bool = True,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders from HuggingFace dataset.
    
    Args:
        dataset: HuggingFace dataset
        tokenizer: Whisper tokenizer
        text_column: Name of the text column
        train_split: Proportion of data for training (0.0-1.0)
        batch_size: Batch size
        max_length: Maximum sequence length
        num_workers: Number of dataloader workers
        pin_memory: Whether to pin memory for faster GPU transfer
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_dataloader, val_dataloader)
    """
    logger.info("Creating dataloaders...")
    
    # Extract texts
    texts = prepare_texts(dataset, text_column, max_length, tokenizer)
    
    # Shuffle and split
    import random
    random.seed(seed)
    random.shuffle(texts)
    
    split_idx = int(len(texts) * train_split)
    train_texts = texts[:split_idx]
    val_texts = texts[split_idx:]
    
    logger.info(f"  Training examples: {len(train_texts)}")
    logger.info(f"  Validation examples: {len(val_texts)}")
    
    # Create PyTorch datasets
    train_dataset = TextDataset(train_texts, tokenizer, max_length)
    val_dataset = TextDataset(val_texts, tokenizer, max_length)
    
    # Create data collator
    collator = DataCollator(tokenizer, max_length)
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collator,
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collator,
    )
    
    logger.info(f"  Train batches: {len(train_dataloader)}")
    logger.info(f"  Val batches: {len(val_dataloader)}")
    
    return train_dataloader, val_dataloader


def get_dataset_statistics(texts: List[str], tokenizer: WhisperTokenizer) -> Dict[str, Any]:
    """
    Compute basic statistics about the dataset.
    
    Args:
        texts: List of text strings
        tokenizer: Tokenizer for token length computation
        
    Returns:
        Dictionary with dataset statistics
    """
    import numpy as np
    
    logger.info("Computing dataset statistics...")
    
    # Character lengths
    char_lengths = [len(text) for text in texts]
    
    # Token lengths (sample first 1000 for efficiency)
    sample_size = min(1000, len(texts))
    token_lengths = [len(tokenizer.encode(text)) for text in texts[:sample_size]]
    
    stats = {
        "num_examples": len(texts),
        "char_length_mean": float(np.mean(char_lengths)),
        "char_length_std": float(np.std(char_lengths)),
        "char_length_min": int(np.min(char_lengths)),
        "char_length_max": int(np.max(char_lengths)),
        "token_length_mean": float(np.mean(token_lengths)),
        "token_length_std": float(np.std(token_lengths)),
        "token_length_min": int(np.min(token_lengths)),
        "token_length_max": int(np.max(token_lengths)),
    }
    
    logger.info("Dataset Statistics:")
    logger.info(f"  Total examples: {stats['num_examples']}")
    logger.info(f"  Char length: {stats['char_length_mean']:.1f} ± {stats['char_length_std']:.1f} "
                f"(min: {stats['char_length_min']}, max: {stats['char_length_max']})")
    logger.info(f"  Token length: {stats['token_length_mean']:.1f} ± {stats['token_length_std']:.1f} "
                f"(min: {stats['token_length_min']}, max: {stats['token_length_max']})")
    
    return stats
