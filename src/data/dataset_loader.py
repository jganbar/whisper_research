"""
Dataset Loading Module

This module handles loading and preparing the DOLLMA Azerbaijani dataset
for training the Whisper decoder.
"""

import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset, Dataset as HFDataset
from transformers import WhisperTokenizer

from .preprocessing import preprocess_text

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
            Batched tensors
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
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single example."""
        text = self.texts[idx]
        
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


def load_dollma_dataset(
    dataset_name: str = "allmalab/DOLLMA",
    cache_dir: Optional[str] = None,
    streaming: bool = False,
    split: Optional[str] = None,
    configs: Optional[List[str]] = None,
) -> HFDataset:
    """
    Load the DOLLMA dataset from HuggingFace.
    
    DOLLMA has multiple configs (subsets) for different Azerbaijani text sources.
    This function can load one or all of them.
    
    Args:
        dataset_name: Name of the dataset on HuggingFace Hub
        cache_dir: Directory to cache the dataset
        streaming: Whether to use streaming mode
        split: Specific split to load
        configs: List of dataset configs to load. If None, loads all available configs.
                Available: ['anl-news', 'azwiki', 'bhos', 'elite-blogs', 
                           'elite-books', 'eqanun', 'mediocore-books', 'translated-enwiki']
        
    Returns:
        Loaded dataset (concatenated if multiple configs)
    """
    logger.info(f"Loading dataset: {dataset_name}")
    
    # Available DOLLMA configs
    available_configs = [
        'anl-news', 'azwiki', 'bhos', 'elite-blogs', 
        'elite-books', 'eqanun', 'mediocore-books', 'translated-enwiki'
    ]
    
    if configs is None:
        configs = available_configs
        logger.info(f"Loading all {len(configs)} DOLLMA subsets")
    
    try:
        datasets = []
        for config in configs:
            logger.info(f"  Loading config: {config}")
            ds = load_dataset(
                dataset_name,
                config,
                cache_dir=cache_dir,
                streaming=streaming,
                split=split or "train",
            )
            datasets.append(ds)
            if not streaming:
                logger.info(f"    Loaded {len(ds)} examples from {config}")
        
        # Concatenate all datasets
        if len(datasets) > 1:
            from datasets import concatenate_datasets
            dataset = concatenate_datasets(datasets)
            logger.info(f"Total dataset size: {len(dataset)} examples")
        else:
            dataset = datasets[0]
        
        return dataset
    
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise


def prepare_dataset(
    dataset: HFDataset,
    tokenizer: WhisperTokenizer,
    text_column: str = "text",
    min_length: int = 10,
    max_length: int = 10000,
    max_seq_length: int = 448,
) -> List[str]:
    """
    Prepare and preprocess the dataset.
    
    Args:
        dataset: HuggingFace dataset
        tokenizer: Tokenizer to use
        text_column: Name of the text column
        min_length: Minimum text length in characters
        max_length: Maximum text length in characters
        max_seq_length: Maximum sequence length for tokenization
        
    Returns:
        List of preprocessed texts
    """
    from tqdm import tqdm
    
    logger.info("Preprocessing dataset...")
    logger.info(f"Total examples to process: {len(dataset):,}")
    
    texts = []
    skipped = 0
    
    # Add progress bar for better tracking
    for example in tqdm(dataset, desc="Preprocessing", unit=" examples"):
        if text_column not in example:
            text_column = _find_text_column(example)
        
        text = example[text_column]
        
        # Preprocess
        processed_text = preprocess_text(
            text,
            lowercase=False,
            remove_punctuation=False,
        )
        
        # Filter by length
        if len(processed_text) < min_length or len(processed_text) > max_length:
            skipped += 1
            continue
        
        # Check tokenized length
        tokens = tokenizer.encode(processed_text)
        if len(tokens) > max_seq_length:
            chunks = _chunk_text(processed_text, tokenizer, max_seq_length)
            texts.extend(chunks)
        else:
            texts.append(processed_text)
    
    logger.info(f"Prepared {len(texts)} text examples")
    logger.info(f"Skipped {skipped} examples")
    
    return texts


def _find_text_column(example: Dict[str, Any]) -> str:
    """Find the text column in the dataset."""
    candidates = ["text", "content", "sentence", "document"]
    
    for key in example.keys():
        if key.lower() in candidates:
            return key
        if isinstance(example[key], str):
            return key
    
    raise ValueError(f"Could not find text column. Available: {list(example.keys())}")


def _chunk_text(
    text: str,
    tokenizer: WhisperTokenizer,
    max_length: int,
    overlap: int = 50,
) -> List[str]:
    """Chunk long text into smaller pieces."""
    tokens = tokenizer.encode(text)
    
    chunks = []
    start = 0
    
    while start < len(tokens):
        end = min(start + max_length, len(tokens))
        chunk_tokens = tokens[start:end]
        
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        chunks.append(chunk_text)
        
        start = end - overlap
        
        if start >= len(tokens):
            break
    
    return chunks


def create_dataloaders(
    texts: List[str],
    tokenizer: WhisperTokenizer,
    train_split: float = 0.95,
    batch_size: int = 32,
    max_length: int = 448,
    num_workers: int = 4,
    pin_memory: bool = True,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders.
    
    Args:
        texts: List of preprocessed texts
        tokenizer: Tokenizer to use
        train_split: Proportion of data for training
        batch_size: Batch size
        max_length: Maximum sequence length
        num_workers: Number of dataloader workers
        pin_memory: Whether to pin memory
        seed: Random seed
        
    Returns:
        Tuple of (train_dataloader, val_dataloader)
    """
    import random
    random.seed(seed)
    random.shuffle(texts)
    
    # Split into train and validation
    split_idx = int(len(texts) * train_split)
    train_texts = texts[:split_idx]
    val_texts = texts[split_idx:]
    
    logger.info(f"Training examples: {len(train_texts)}")
    logger.info(f"Validation examples: {len(val_texts)}")
    
    # Create datasets
    train_dataset = TextDataset(train_texts, tokenizer, max_length)
    val_dataset = TextDataset(val_texts, tokenizer, max_length)
    
    # Create collator
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
    
    return train_dataloader, val_dataloader


def get_dataset_statistics(texts: List[str], tokenizer: WhisperTokenizer) -> Dict[str, Any]:
    """
    Compute statistics about the dataset.
    
    Args:
        texts: List of texts
        tokenizer: Tokenizer to use
        
    Returns:
        Dictionary of statistics
    """
    import numpy as np
    
    text_lengths = [len(text) for text in texts]
    token_lengths = [len(tokenizer.encode(text)) for text in texts[:1000]]
    
    stats = {
        "num_examples": len(texts),
        "text_length_mean": np.mean(text_lengths),
        "text_length_std": np.std(text_lengths),
        "text_length_min": np.min(text_lengths),
        "text_length_max": np.max(text_lengths),
        "token_length_mean": np.mean(token_lengths),
        "token_length_std": np.std(token_lengths),
        "token_length_min": np.min(token_lengths),
        "token_length_max": np.max(token_lengths),
    }
    
    logger.info("Dataset Statistics:")
    for key, value in stats.items():
        logger.info(f"  {key}: {value:.2f}")
    
    return stats

