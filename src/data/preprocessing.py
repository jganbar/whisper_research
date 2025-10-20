"""
Text Preprocessing Module

This module contains functions for preprocessing Azerbaijani text data.
"""

import re
import logging
from typing import List

logger = logging.getLogger(__name__)


def preprocess_text(
    text: str,
    lowercase: bool = False,
    remove_punctuation: bool = False,
    remove_numbers: bool = False,
    remove_extra_whitespace: bool = True,
) -> str:
    """
    Preprocess a single text string.
    
    Args:
        text: Input text
        lowercase: Whether to convert to lowercase
        remove_punctuation: Whether to remove punctuation
        remove_numbers: Whether to remove numbers
        remove_extra_whitespace: Whether to remove extra whitespace
        
    Returns:
        Preprocessed text
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Remove control characters
    text = remove_control_characters(text)
    
    # Normalize unicode
    text = normalize_unicode(text)
    
    # Convert to lowercase if requested
    if lowercase:
        text = text.lower()
    
    # Remove URLs
    text = remove_urls(text)
    
    # Remove email addresses
    text = remove_emails(text)
    
    # Remove numbers if requested
    if remove_numbers:
        text = re.sub(r'\d+', '', text)
    
    # Remove punctuation if requested
    if remove_punctuation:
        text = re.sub(r'[^\w\sƏəİıÖöÜüĞğŞşÇç]', '', text)
    
    # Remove extra whitespace
    if remove_extra_whitespace:
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
    
    return text


def remove_control_characters(text: str) -> str:
    """Remove control characters from text."""
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
    return text


def normalize_unicode(text: str) -> str:
    """Normalize unicode characters."""
    import unicodedata
    text = unicodedata.normalize('NFC', text)
    return text


def remove_urls(text: str) -> str:
    """Remove URLs from text."""
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'www\.\S+', '', text)
    return text


def remove_emails(text: str) -> str:
    """Remove email addresses from text."""
    text = re.sub(r'\S+@\S+', '', text)
    return text


def clean_azerbaijani_text(text: str) -> str:
    """
    Apply Azerbaijani-specific text cleaning.
    
    Args:
        text: Input text
        
    Returns:
        Cleaned text
    """
    # Azerbaijani alphabet normalization
    replacements = {
        'ə': 'ə',
        'Ə': 'Ə',
        'ı': 'ı',
        'İ': 'İ',
        'ö': 'ö',
        'Ö': 'Ö',
        'ü': 'ü',
        'Ü': 'Ü',
        'ğ': 'ğ',
        'Ğ': 'Ğ',
        'ş': 'ş',
        'Ş': 'Ş',
        'ç': 'ç',
        'Ç': 'Ç',
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    return text


def filter_text_quality(
    text: str,
    min_length: int = 10,
    max_length: int = 10000,
) -> bool:
    """
    Check if text meets quality criteria.
    
    Args:
        text: Input text
        min_length: Minimum text length
        max_length: Maximum text length
        
    Returns:
        True if text passes quality checks
    """
    # Check length
    if len(text) < min_length or len(text) > max_length:
        return False
    
    # Check if text has enough alphabetic characters
    alpha_chars = sum(c.isalpha() for c in text)
    if alpha_chars < len(text) * 0.5:
        return False
    
    return True


def batch_preprocess(
    texts: List[str],
    lowercase: bool = False,
    remove_punctuation: bool = False,
    filter_quality: bool = True,
    show_progress: bool = True,
) -> List[str]:
    """
    Preprocess a batch of texts.
    
    Args:
        texts: List of input texts
        lowercase: Whether to convert to lowercase
        remove_punctuation: Whether to remove punctuation
        filter_quality: Whether to filter low-quality texts
        show_progress: Whether to show progress bar
        
    Returns:
        List of preprocessed texts
    """
    from tqdm import tqdm
    
    preprocessed = []
    
    iterator = tqdm(texts, desc="Preprocessing") if show_progress else texts
    
    for text in iterator:
        # Preprocess
        processed = preprocess_text(
            text,
            lowercase=lowercase,
            remove_punctuation=remove_punctuation,
        )
        
        # Apply Azerbaijani-specific cleaning
        processed = clean_azerbaijani_text(processed)
        
        # Filter quality
        if filter_quality and not filter_text_quality(processed):
            continue
        
        preprocessed.append(processed)
    
    logger.info(f"Preprocessed {len(preprocessed)} / {len(texts)} texts")
    
    return preprocessed

