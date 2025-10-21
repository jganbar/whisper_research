"""
Text Preprocessing Module

This module contains basic text preprocessing utilities.
Since the dataset is pre-prepared, we keep preprocessing minimal.
"""

import re
import logging
import unicodedata
from typing import List

logger = logging.getLogger(__name__)


def normalize_unicode(text: str) -> str:
    """
    Normalize unicode characters to NFC form.
    
    Args:
        text: Input text
        
    Returns:
        Normalized text
    """
    return unicodedata.normalize('NFC', text)


def remove_control_characters(text: str) -> str:
    """
    Remove control characters from text.
    
    Args:
        text: Input text
        
    Returns:
        Text with control characters removed
    """
    return re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)


def clean_whitespace(text: str) -> str:
    """
    Clean extra whitespace from text.
    
    Args:
        text: Input text
        
    Returns:
        Text with normalized whitespace
    """
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)
    # Strip leading/trailing whitespace
    text = text.strip()
    return text


def preprocess_text(text: str) -> str:
    """
    Apply basic preprocessing to text.
    
    This function applies minimal preprocessing since the dataset
    is already prepared. It only normalizes unicode and removes
    control characters.
    
    Args:
        text: Input text
        
    Returns:
        Preprocessed text
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Normalize unicode
    text = normalize_unicode(text)
    
    # Remove control characters
    text = remove_control_characters(text)
    
    # Clean whitespace
    text = clean_whitespace(text)
    
    return text


def batch_preprocess(
    texts: List[str],
    show_progress: bool = True,
) -> List[str]:
    """
    Preprocess a batch of texts.
    
    Args:
        texts: List of input texts
        show_progress: Whether to show progress bar
        
    Returns:
        List of preprocessed texts
    """
    from tqdm import tqdm
    
    preprocessed = []
    
    iterator = tqdm(texts, desc="Preprocessing") if show_progress else texts
    
    for text in iterator:
        processed = preprocess_text(text)
        if processed:  # Only add non-empty texts
            preprocessed.append(processed)
    
    logger.info(f"Preprocessed {len(preprocessed)} / {len(texts)} texts")
    
    return preprocessed
