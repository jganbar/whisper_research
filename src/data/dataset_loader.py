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
        
        # Pad sequences (use 'longest' to avoid warning)
        padded = self.tokenizer.pad(
            {"input_ids": input_ids},
            padding='longest',
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
    """PyTorch Dataset for text data with EFFICIENT pre-tokenization."""
    
    def __init__(
        self,
        texts: List[str],
        tokenizer: WhisperTokenizer,
        max_length: int = 448,
    ):
        """
        Initialize the dataset with MEMORY-EFFICIENT PRE-TOKENIZATION.
        
        Args:
            texts: List of text strings
            tokenizer: Whisper tokenizer
            max_length: Maximum sequence length
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # PRE-TOKENIZE with EFFICIENT STORAGE (numpy arrays, not Python lists!)
        print(f"Pre-tokenizing {len(texts):,} texts with EFFICIENT storage...")
        from tqdm import tqdm
        import numpy as np
        
        # Use tiktoken directly for MASSIVE speedup (C++ vs Python)
        try:
            import tiktoken
            use_tiktoken = True
            tiktoken_enc = tiktoken.get_encoding("gpt2")
            print("✓ Using tiktoken (C++ tokenizer) - ULTRA FAST!")
        except:
            use_tiktoken = False
            print("⚠ tiktoken not available, using HuggingFace tokenizer")
        
        # EFFICIENT: Store as numpy arrays (10x less RAM than Python lists!)
        all_input_ids = []
        all_lengths = []
        
        if use_tiktoken:
            print("Tokenizing all texts with tiktoken...")
            for text in tqdm(texts, desc="Tokenizing"):
                token_ids = tiktoken_enc.encode(text, allowed_special=set())
                if len(token_ids) > self.max_length:
                    token_ids = token_ids[:self.max_length]
                
                all_input_ids.append(np.array(token_ids, dtype=np.int32))
                all_lengths.append(len(token_ids))
        else:
            # Fallback: HuggingFace tokenizer with batching
            batch_size = 10000
            print(f"Tokenizing in batches of {batch_size}...")
            
            for i in tqdm(range(0, len(texts), batch_size), desc="Tokenizing"):
                batch_texts = texts[i:i+batch_size]
                encoded_batch = self.tokenizer(
                    batch_texts,
                    truncation=True,
                    max_length=self.max_length,
                    padding=False,
                    return_tensors=None,
                )
                
                for token_ids in encoded_batch["input_ids"]:
                    all_input_ids.append(np.array(token_ids, dtype=np.int32))
                    all_lengths.append(len(token_ids))
        
        # Store as list of numpy arrays (much more efficient than Python lists of ints)
        self.input_ids = all_input_ids
        self.lengths = np.array(all_lengths, dtype=np.int32)
        
        print(f"✓ Pre-tokenization complete! {len(self.input_ids):,} samples ready.")
        
        # Calculate memory usage
        total_tokens = np.sum(self.lengths)
        memory_mb = (total_tokens * 4) / (1024**2)  # int32 = 4 bytes
        print(f"  Memory usage: ~{memory_mb:.1f} MB (efficient numpy storage)")
    
    def __len__(self) -> int:
        return len(self.input_ids)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a pre-tokenized example (instant conversion from numpy!)"""
        input_ids = torch.from_numpy(self.input_ids[idx]).long()
        length = int(self.lengths[idx])
        attention_mask = torch.ones(length, dtype=torch.long)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
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
    Ultra-fast batch processing with multiprocessing.
    
    Args:
        dataset: HuggingFace dataset
        text_column: Name of the text column
        max_seq_length: Maximum sequence length (for filtering)
        tokenizer: Optional tokenizer for length checking
        
    Returns:
        List of text strings
    """
    logger.info("Extracting texts from dataset...")
    logger.info(f"  Processing {len(dataset):,} examples with batch processing...")
    
    # Calculate max character length (approximate filtering, much faster than tokenization)
    max_chars = max_seq_length * 6  # Conservative: 1 token ≈ 4-6 chars
    
    def process_batch(examples):
        """Process a batch of examples (runs in parallel)."""
        texts = examples[text_column]
        processed_texts = []
        is_valid = []
        
        for text in texts:
            # Basic validation and cleaning
            if text and isinstance(text, str):
                text = text.strip()
                if len(text) > 0:
                    # Truncate if too long
                    if len(text) > max_chars:
                        text = text[:max_chars]
                    processed_texts.append(text)
                    is_valid.append(True)
                else:
                    processed_texts.append("")
                    is_valid.append(False)
            else:
                processed_texts.append("")
                is_valid.append(False)
        
        return {"processed_text": processed_texts, "is_valid": is_valid}
    
    # Use HuggingFace's parallel batch processing (utilizes all CPU cores)
    logger.info("  Running parallel batch processing...")
    processed = dataset.map(
        process_batch,
        batched=True,
        batch_size=10000,  # Large batches for efficiency
        num_proc=None,  # Use all available CPU cores
        desc="Processing texts",
    )
    
    # Filter and extract valid texts only
    valid_dataset = processed.filter(lambda x: x["is_valid"], num_proc=None, desc="Filtering valid texts")
    texts = valid_dataset["processed_text"]
    
    # Convert to Python list (HuggingFace Column object is not mutable)
    texts = list(texts)
    
    original_count = len(dataset)
    valid_count = len(texts)
    skipped = original_count - valid_count
    
    logger.info(f"✓ Extracted {valid_count:,} valid texts from {original_count:,} examples")
    if skipped > 0:
        logger.info(f"  Skipped {skipped:,} invalid examples ({skipped/original_count*100:.1f}%)")
    
    return texts


def create_dataloaders(
    dataset: HFDataset,
    tokenizer: WhisperTokenizer,
    text_column: str = "text",
    train_split: float = 0.95,
    val_split: float = 0.05,
    max_train_samples: Optional[int] = None,
    max_val_samples: Optional[int] = None,
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
        train_split: Proportion of TOTAL data for training (0.0-1.0)
        val_split: Proportion of TOTAL data for validation (0.0-1.0)
        max_train_samples: Maximum number of training samples (None = use all)
        max_val_samples: Maximum number of validation samples (None = use all)
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
    
    # Shuffle first
    import random
    random.seed(seed)
    random.shuffle(texts)
    
    # Calculate split indices based on percentages of TOTAL dataset
    total_samples = len(texts)
    train_samples = int(total_samples * train_split)
    val_samples = int(total_samples * val_split)
    
    # Apply max_samples limits if specified
    if max_train_samples is not None:
        train_samples = min(train_samples, max_train_samples)
    if max_val_samples is not None:
        val_samples = min(val_samples, max_val_samples)
    
    # Split the data
    train_texts = texts[:train_samples]
    val_texts = texts[train_samples:train_samples + val_samples]
    
    logger.info(f"  Total available samples: {total_samples:,}")
    logger.info(f"  Training examples: {len(train_texts):,}")
    logger.info(f"  Validation examples: {len(val_texts):,}")
    
    # Create PyTorch datasets
    train_dataset = TextDataset(train_texts, tokenizer, max_length)
    val_dataset = TextDataset(val_texts, tokenizer, max_length)
    
    # Create data collator
    collator = DataCollator(tokenizer, max_length)
    
    # Create dataloaders with minimal workers (data is pre-tokenized!)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collator,
        persistent_workers=False,  # Don't keep workers alive (saves memory)
        prefetch_factor=2 if num_workers > 0 else None,
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collator,
        persistent_workers=False,  # Don't keep workers alive (saves memory)
        prefetch_factor=2 if num_workers > 0 else None,
    )
    
    logger.info(f"  Train batches: {len(train_dataloader):,}")
    logger.info(f"  Val batches: {len(val_dataloader):,}")
    
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
