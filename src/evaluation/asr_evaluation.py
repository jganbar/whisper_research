"""
ASR Evaluation Module

This module handles ASR evaluation on Azerbaijani test datasets.
"""

import logging
from typing import List, Dict, Any, Optional
import os

import torch
from datasets import load_dataset
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from tqdm import tqdm

from .metrics import calculate_wer, calculate_cer, calculate_detailed_metrics

logger = logging.getLogger(__name__)


def load_evaluation_dataset(
    dataset_name: str = "mozilla-foundation/common_voice_11_0",
    language: str = "az",
    split: str = "test",
    cache_dir: Optional[str] = None,
):
    """
    Load ASR evaluation dataset.
    
    Args:
        dataset_name: Name of the dataset
        language: Language code
        split: Dataset split to load
        cache_dir: Cache directory
        
    Returns:
        Loaded dataset
    """
    logger.info(f"Loading evaluation dataset: {dataset_name} ({language})")
    
    try:
        dataset = load_dataset(
            dataset_name,
            language,
            split=split,
            cache_dir=cache_dir,
        )
        
        logger.info(f"Loaded {len(dataset)} examples")
        return dataset
    
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise


def transcribe_audio_batch(
    model: WhisperForConditionalGeneration,
    processor: WhisperProcessor,
    audio_arrays: List,
    sampling_rate: int = 16000,
    language: str = "az",
    task: str = "transcribe",
    batch_size: int = 16,
    device: str = "cuda",
) -> List[str]:
    """
    Transcribe a batch of audio arrays.
    
    Args:
        model: Whisper model
        processor: Whisper processor
        audio_arrays: List of audio arrays
        sampling_rate: Sampling rate
        language: Language code
        task: Task (transcribe or translate)
        batch_size: Batch size
        device: Device to run on
        
    Returns:
        List of transcriptions
    """
    model.eval()
    transcriptions = []
    
    with torch.no_grad():
        for i in range(0, len(audio_arrays), batch_size):
            batch_audios = audio_arrays[i:i+batch_size]
            
            # Process audio
            inputs = processor(
                batch_audios,
                sampling_rate=sampling_rate,
                return_tensors="pt",
            )
            
            input_features = inputs.input_features.to(device)
            
            # Generate transcriptions
            forced_decoder_ids = processor.get_decoder_prompt_ids(
                language=language,
                task=task
            )
            
            predicted_ids = model.generate(
                input_features,
                forced_decoder_ids=forced_decoder_ids,
            )
            
            # Decode transcriptions
            batch_transcriptions = processor.batch_decode(
                predicted_ids,
                skip_special_tokens=True
            )
            
            transcriptions.extend(batch_transcriptions)
    
    return transcriptions


def evaluate_asr(
    model: WhisperForConditionalGeneration,
    processor: WhisperProcessor,
    dataset,
    language: str = "az",
    batch_size: int = 16,
    device: str = "cuda",
    max_samples: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Evaluate ASR performance on a dataset.
    
    Args:
        model: Whisper model
        processor: Whisper processor
        dataset: Evaluation dataset
        language: Language code
        batch_size: Batch size
        device: Device to run on
        max_samples: Maximum number of samples to evaluate
        
    Returns:
        Dictionary with evaluation results
    """
    logger.info("Starting ASR evaluation...")
    
    model = model.to(device)
    model.eval()
    
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    references = []
    hypotheses = []
    
    # Collect audio and references
    audio_arrays = []
    for example in dataset:
        audio_arrays.append(example["audio"]["array"])
        references.append(example["sentence"])
    
    # Transcribe in batches
    logger.info(f"Transcribing {len(audio_arrays)} audio samples...")
    hypotheses = transcribe_audio_batch(
        model=model,
        processor=processor,
        audio_arrays=audio_arrays,
        language=language,
        batch_size=batch_size,
        device=device,
    )
    
    # Calculate metrics
    logger.info("Calculating metrics...")
    metrics = calculate_detailed_metrics(references, hypotheses)
    
    logger.info(f"\nEvaluation Results:")
    logger.info(f"  WER: {metrics['wer']:.2f}%")
    logger.info(f"  CER: {metrics['cer']:.2f}%")
    
    # Add predictions to results
    predictions = [
        {"reference": ref, "hypothesis": hyp}
        for ref, hyp in zip(references, hypotheses)
    ]
    
    results = {
        "metrics": metrics,
        "predictions": predictions,
        "num_samples": len(references),
    }
    
    return results


def compare_models(
    baseline_model: WhisperForConditionalGeneration,
    finetuned_model: WhisperForConditionalGeneration,
    processor: WhisperProcessor,
    dataset,
    language: str = "az",
    batch_size: int = 16,
    device: str = "cuda",
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Compare baseline and fine-tuned models.
    
    Args:
        baseline_model: Baseline Whisper model
        finetuned_model: Fine-tuned Whisper model
        processor: Whisper processor
        dataset: Evaluation dataset
        language: Language code
        batch_size: Batch size
        device: Device to run on
        output_dir: Directory to save results
        
    Returns:
        Comparison results
    """
    logger.info("Evaluating baseline model...")
    baseline_results = evaluate_asr(
        baseline_model,
        processor,
        dataset,
        language=language,
        batch_size=batch_size,
        device=device,
    )
    
    logger.info("\nEvaluating fine-tuned model...")
    finetuned_results = evaluate_asr(
        finetuned_model,
        processor,
        dataset,
        language=language,
        batch_size=batch_size,
        device=device,
    )
    
    # Create comparison report
    from .metrics import create_comparison_report
    
    report_path = os.path.join(output_dir, "comparison_report.json") if output_dir else None
    comparison = create_comparison_report(
        baseline_results["metrics"],
        finetuned_results["metrics"],
        output_path=report_path,
    )
    
    results = {
        "baseline": baseline_results,
        "finetuned": finetuned_results,
        "comparison": comparison,
    }
    
    # Save predictions
    if output_dir:
        from .metrics import save_predictions
        
        os.makedirs(output_dir, exist_ok=True)
        
        save_predictions(
            baseline_results["predictions"],
            os.path.join(output_dir, "baseline_predictions.json"),
            format="json",
        )
        
        save_predictions(
            finetuned_results["predictions"],
            os.path.join(output_dir, "finetuned_predictions.json"),
            format="json",
        )
    
    return results

