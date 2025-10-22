"""
Whisper Decoder Extraction Module

This module extracts the decoder component from Whisper Large v3 and wraps it
as a standalone causal language model for unsupervised text training.
"""

import logging
import os
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
from transformers import (
    WhisperForConditionalGeneration,
    WhisperTokenizer,
    WhisperConfig,
)

logger = logging.getLogger(__name__)


class WhisperDecoderLM(nn.Module):
    """
    Standalone Whisper Decoder for Causal Language Modeling.
    
    This class wraps the Whisper decoder to function as a GPT-like
    causal language model that can be trained on text data.
    """
    
    def __init__(
        self,
        decoder,
        embed_tokens,
        embed_positions,
        config: WhisperConfig,
    ):
        """
        Initialize the decoder language model.
        
        Args:
            decoder: The Whisper decoder module
            embed_tokens: Token embedding layer
            embed_positions: Positional embedding layer
            config: Whisper configuration
        """
        super().__init__()
        
        self.config = config
        self.decoder = decoder
        self.embed_tokens = embed_tokens
        self.embed_positions = embed_positions
        
        # Language modeling head
        self.lm_head = nn.Linear(
            config.d_model,
            config.vocab_size,
            bias=False
        )
        
        # Tie weights with input embeddings
        self.lm_head.weight = self.embed_tokens.weight
        
        logger.info(
            f"Initialized WhisperDecoderLM with {self.num_parameters():,} parameters"
        )
    
    def num_parameters(self) -> int:
        """Return the total number of parameters."""
        return sum(p.numel() for p in self.parameters())
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for causal language modeling.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            labels: Labels for language modeling loss [batch_size, seq_len]
            return_dict: Whether to return a dictionary
            
        Returns:
            Dictionary containing loss and logits
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Get embeddings
        inputs_embeds = self.embed_tokens(input_ids)
        
        # Get positional embeddings
        positions = torch.arange(
            0, seq_len, dtype=torch.long, device=device
        ).unsqueeze(0).expand(batch_size, -1)
        position_embeds = self.embed_positions(positions)
        
        # Combine token and positional embeddings
        hidden_states = inputs_embeds + position_embeds
        
        # Let HF Whisper create its own efficient causal mask (Flash/SFDP compatible)
        # We pass the 2D attention_mask directly to unlock fused kernels.
        # Pass through decoder (decoder-only mode, no encoder)
        decoder_outputs = self.decoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        
        # Get logits from LM head
        hidden_states = decoder_outputs.last_hidden_state
        logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            # Shift logits and labels for next token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Calculate cross-entropy loss
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1)
            )
        
        if not return_dict:
            return (loss, logits) if loss is not None else (logits,)
        
        return {
            "loss": loss,
            "logits": logits,
            "hidden_states": hidden_states,
        }


## Note: We intentionally avoid building a custom 4D causal mask to allow
## PyTorch's scaled_dot_product_attention (Flash/SFDP) fast paths.


def extract_decoder(
    model_name: str = "openai/whisper-large-v3",
    cache_dir: Optional[str] = None,
    device: str = "cuda",
) -> tuple[WhisperDecoderLM, WhisperTokenizer]:
    """
    Extract the decoder from a Whisper model and create a standalone LM.
    
    Args:
        model_name: Name of the Whisper model on HuggingFace Hub
        cache_dir: Directory to cache the model
        device: Device to load the model on
        
    Returns:
        Tuple of (WhisperDecoderLM, WhisperTokenizer)
    """
    logger.info(f"Loading Whisper model: {model_name}")
    
    # Load the full Whisper model
    whisper_model = WhisperForConditionalGeneration.from_pretrained(
        model_name,
        cache_dir=cache_dir,
    )
    
    # Load FAST tokenizer (Rust-based, 10-100x faster!)
    tokenizer = WhisperTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        use_fast=True,
    )
    
    # Extract decoder components
    decoder = whisper_model.model.decoder
    embed_tokens = whisper_model.model.decoder.embed_tokens
    embed_positions = whisper_model.model.decoder.embed_positions
    config = whisper_model.config
    
    # Create standalone decoder LM
    decoder_lm = WhisperDecoderLM(
        decoder=decoder,
        embed_tokens=embed_tokens,
        embed_positions=embed_positions,
        config=config,
    )
    
    # Move to device
    decoder_lm = decoder_lm.to(device)
    
    logger.info(f"Successfully extracted decoder with {decoder_lm.num_parameters():,} parameters")
    logger.info(f"Model device: {device}")
    
    return decoder_lm, tokenizer


def save_decoder(
    model: WhisperDecoderLM,
    tokenizer: WhisperTokenizer,
    output_dir: str,
) -> None:
    """
    Save the decoder model and tokenizer.
    
    Args:
        model: WhisperDecoderLM model to save
        tokenizer: Tokenizer to save
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model state dict
    model_path = os.path.join(output_dir, "decoder_lm.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": model.config.to_dict(),
    }, model_path)
    
    # Save tokenizer
    tokenizer.save_pretrained(output_dir)
    
    logger.info(f"Saved decoder model and tokenizer to {output_dir}")


def load_decoder(
    model_path: str,
    device: str = "cuda",
) -> tuple[WhisperDecoderLM, WhisperTokenizer]:
    """
    Load a saved decoder model and tokenizer.
    
    Args:
        model_path: Path to the saved model directory
        device: Device to load the model on
        
    Returns:
        Tuple of (WhisperDecoderLM, WhisperTokenizer)
    """
    logger.info(f"Loading decoder from {model_path}")
    
    # Load FAST tokenizer (Rust-based, 10-100x faster!)
    tokenizer = WhisperTokenizer.from_pretrained(model_path, use_fast=True)
    
    # Load model checkpoint
    checkpoint_path = os.path.join(model_path, "decoder_lm.pt")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # Reconstruct config
    config = WhisperConfig.from_dict(checkpoint["config"])
    
    # Rebuild decoder structure without downloading the full pretrained weights
    base_model = WhisperForConditionalGeneration(config)
    decoder_lm = WhisperDecoderLM(
        decoder=base_model.model.decoder,
        embed_tokens=base_model.model.decoder.embed_tokens,
        embed_positions=base_model.model.decoder.embed_positions,
        config=config,
    )
    del base_model
    
    # Load fine-tuned weights
    decoder_lm.load_state_dict(checkpoint["model_state_dict"])
    
    # CRITICAL: Move model to device AFTER loading weights!
    decoder_lm = decoder_lm.to(device)
    
    logger.info(f"Loaded decoder model from {model_path}")
    logger.info(f"Model moved to device: {device}")
    
    return decoder_lm, tokenizer
