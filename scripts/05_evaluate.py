#!/usr/bin/env python3
"""
Script 5: Evaluate ASR Performance

This script evaluates and compares baseline and fine-tuned models.
"""

import argparse
import logging
import yaml
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation import load_evaluation_dataset, compare_models
from src.model import load_integrated_model
from transformers import WhisperForConditionalGeneration, WhisperProcessor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Evaluate ASR performance")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/evaluation_config.yaml",
        help="Path to evaluation config file"
    )
    parser.add_argument(
        "--finetuned_model",
        type=str,
        required=True,
        help="Path to fine-tuned model"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate"
    )
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info("=" * 50)
    logger.info("Step 5: Evaluating Models")
    logger.info("=" * 50)
    
    # Load evaluation dataset
    dataset_config = config['datasets']['common_voice']
    dataset = load_evaluation_dataset(
        dataset_name=dataset_config['name'],
        language=dataset_config['language'],
        split=dataset_config['split'],
        cache_dir=dataset_config.get('cache_dir'),
    )
    
    # Load baseline model
    baseline_config = config['models']['baseline']
    logger.info(f"Loading baseline model: {baseline_config['name']}")
    baseline_model = WhisperForConditionalGeneration.from_pretrained(
        baseline_config['name'],
        cache_dir=baseline_config.get('cache_dir'),
    )
    processor = WhisperProcessor.from_pretrained(baseline_config['name'])
    
    # Load fine-tuned model
    logger.info(f"Loading fine-tuned model: {args.finetuned_model}")
    finetuned_model, _ = load_integrated_model(args.finetuned_model, device=args.device)
    
    # Compare models
    transcription_config = config['transcription']
    output_config = config['output']
    
    results = compare_models(
        baseline_model=baseline_model,
        finetuned_model=finetuned_model,
        processor=processor,
        dataset=dataset,
        language=transcription_config['language'],
        batch_size=transcription_config['batch_size'],
        device=args.device,
        output_dir=output_config['results_dir'],
    )
    
    # Print summary
    logger.info("\n" + "=" * 50)
    logger.info("Evaluation Summary")
    logger.info("=" * 50)
    
    baseline_wer = results['baseline']['metrics']['wer']
    finetuned_wer = results['finetuned']['metrics']['wer']
    improvement = baseline_wer - finetuned_wer
    
    logger.info(f"\nBaseline WER:   {baseline_wer:.2f}%")
    logger.info(f"Fine-tuned WER: {finetuned_wer:.2f}%")
    logger.info(f"Improvement:    {improvement:.2f}%")
    
    logger.info(f"\n✓ Evaluation completed!")
    logger.info(f"  Results saved to {output_config['results_dir']}")


if __name__ == "__main__":
    main()

