"""
Metrics Module

This module contains functions for calculating evaluation metrics.
"""

import logging
from typing import List, Dict, Any, Optional
import numpy as np
from jiwer import wer, cer

logger = logging.getLogger(__name__)


def calculate_wer(
    references: List[str],
    hypotheses: List[str],
) -> float:
    """
    Calculate Word Error Rate (WER).
    
    Args:
        references: List of reference transcripts
        hypotheses: List of hypothesis transcripts
        
    Returns:
        WER as a percentage
    """
    if len(references) != len(hypotheses):
        raise ValueError("Number of references and hypotheses must match")
    
    if len(references) == 0:
        return 0.0
    
    wer_score = wer(references, hypotheses)
    return wer_score * 100


def calculate_cer(
    references: List[str],
    hypotheses: List[str],
) -> float:
    """
    Calculate Character Error Rate (CER).
    
    Args:
        references: List of reference transcripts
        hypotheses: List of hypothesis transcripts
        
    Returns:
        CER as a percentage
    """
    if len(references) != len(hypotheses):
        raise ValueError("Number of references and hypotheses must match")
    
    if len(references) == 0:
        return 0.0
    
    cer_score = cer(references, hypotheses)
    return cer_score * 100


def calculate_detailed_metrics(
    references: List[str],
    hypotheses: List[str],
) -> Dict[str, Any]:
    """
    Calculate detailed evaluation metrics.
    
    Args:
        references: List of reference transcripts
        hypotheses: List of hypothesis transcripts
        
    Returns:
        Dictionary of metrics
    """
    from jiwer import compute_measures
    
    measures = compute_measures(references, hypotheses)
    
    metrics = {
        "wer": measures["wer"] * 100,
        "cer": cer(references, hypotheses) * 100,
        "hits": measures["hits"],
        "substitutions": measures["substitutions"],
        "deletions": measures["deletions"],
        "insertions": measures["insertions"],
        "num_words": measures["hits"] + measures["substitutions"] + measures["deletions"],
    }
    
    return metrics


def create_comparison_report(
    baseline_metrics: Dict[str, Any],
    finetuned_metrics: Dict[str, Any],
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Create a comparison report between baseline and fine-tuned models.
    
    Args:
        baseline_metrics: Metrics for baseline model
        finetuned_metrics: Metrics for fine-tuned model
        output_path: Path to save report
        
    Returns:
        Comparison report
    """
    report = {
        "baseline": baseline_metrics,
        "finetuned": finetuned_metrics,
        "improvements": {},
    }
    
    # Calculate improvements
    for metric in baseline_metrics:
        if metric in finetuned_metrics and isinstance(baseline_metrics[metric], (int, float)):
            baseline_val = baseline_metrics[metric]
            finetuned_val = finetuned_metrics[metric]
            
            # For error rates, lower is better
            if metric in ["wer", "cer"]:
                improvement = baseline_val - finetuned_val
                relative_improvement = (improvement / baseline_val * 100) if baseline_val > 0 else 0
            else:
                improvement = finetuned_val - baseline_val
                relative_improvement = (improvement / baseline_val * 100) if baseline_val > 0 else 0
            
            report["improvements"][metric] = {
                "absolute": improvement,
                "relative": relative_improvement,
            }
    
    # Log summary
    logger.info("=" * 50)
    logger.info("Comparison Report")
    logger.info("=" * 50)
    
    for metric in ["wer", "cer"]:
        if metric in report["improvements"]:
            baseline_val = baseline_metrics[metric]
            finetuned_val = finetuned_metrics[metric]
            improvement = report["improvements"][metric]
            
            logger.info(f"\n{metric.upper()}:")
            logger.info(f"  Baseline:   {baseline_val:.2f}%")
            logger.info(f"  Fine-tuned: {finetuned_val:.2f}%")
            logger.info(f"  Improvement: {improvement['absolute']:.2f}% ({improvement['relative']:.2f}% relative)")
    
    logger.info("=" * 50)
    
    # Save report
    if output_path:
        import json
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Report saved to {output_path}")
    
    return report


def bootstrap_confidence_interval(
    references: List[str],
    hypotheses_baseline: List[str],
    hypotheses_finetuned: List[str],
    metric: str = "wer",
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
) -> Dict[str, Any]:
    """
    Calculate confidence intervals for metric differences using bootstrap.
    
    Args:
        references: List of reference transcripts
        hypotheses_baseline: Baseline model predictions
        hypotheses_finetuned: Fine-tuned model predictions
        metric: Metric to calculate (wer or cer)
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level
        
    Returns:
        Dictionary with confidence intervals
    """
    import random
    
    n_samples = len(references)
    differences = []
    
    if metric == "wer":
        metric_fn = calculate_wer
    elif metric == "cer":
        metric_fn = calculate_cer
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    # Bootstrap
    for _ in range(n_bootstrap):
        indices = random.choices(range(n_samples), k=n_samples)
        
        refs_sample = [references[i] for i in indices]
        baseline_sample = [hypotheses_baseline[i] for i in indices]
        finetuned_sample = [hypotheses_finetuned[i] for i in indices]
        
        baseline_metric = metric_fn(refs_sample, baseline_sample)
        finetuned_metric = metric_fn(refs_sample, finetuned_sample)
        
        diff = baseline_metric - finetuned_metric
        differences.append(diff)
    
    # Calculate confidence interval
    differences.sort()
    lower_idx = int((1 - confidence_level) / 2 * n_bootstrap)
    upper_idx = int((1 + confidence_level) / 2 * n_bootstrap)
    
    ci_lower = differences[lower_idx]
    ci_upper = differences[upper_idx]
    mean_diff = np.mean(differences)
    
    is_significant = ci_lower > 0
    
    result = {
        "metric": metric,
        "mean_difference": mean_diff,
        "confidence_interval": (ci_lower, ci_upper),
        "confidence_level": confidence_level,
        "is_significant": is_significant,
    }
    
    logger.info(f"\nBootstrap Analysis ({metric.upper()}):")
    logger.info(f"  Mean difference: {mean_diff:.2f}%")
    logger.info(f"  {confidence_level*100:.0f}% CI: [{ci_lower:.2f}%, {ci_upper:.2f}%]")
    logger.info(f"  Statistically significant: {is_significant}")
    
    return result


def save_predictions(
    predictions: List[Dict[str, str]],
    output_path: str,
    format: str = "json",
) -> None:
    """
    Save predictions to file.
    
    Args:
        predictions: List of prediction dictionaries
        output_path: Path to save predictions
        format: Output format (json, csv, or txt)
    """
    import json
    import csv
    
    if format == "json":
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(predictions, f, ensure_ascii=False, indent=2)
    
    elif format == "csv":
        if not predictions:
            return
        
        keys = predictions[0].keys()
        with open(output_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(predictions)
    
    elif format == "txt":
        with open(output_path, "w", encoding="utf-8") as f:
            for pred in predictions:
                f.write(f"Reference: {pred.get('reference', '')}\n")
                f.write(f"Hypothesis: {pred.get('hypothesis', '')}\n")
                f.write("-" * 50 + "\n")
    
    else:
        raise ValueError(f"Unknown format: {format}")
    
    logger.info(f"Saved predictions to {output_path}")

