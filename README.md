# Whisper Decoder Unsupervised Training

Research project for training Whisper Large v3 decoder on Azerbaijani text data.

## Project Overview

This project extracts the decoder component from OpenAI's Whisper Large v3, trains it on Azerbaijani text data (DOLLMA dataset) using causal language modeling, and re-integrates it back into Whisper to improve ASR performance.

## Setup

This project uses [UV](https://github.com/astral-sh/uv) for dependency management.

```bash
# Install UV (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e ".[dev]"
```

## Project Structure

```
whisper_research/
├── pyproject.toml          # Project dependencies
├── README.md               # This file
├── configs/                # Configuration files
│   ├── model_config.yaml
│   ├── training_config.yaml
│   └── evaluation_config.yaml
├── src/                    # Source code
│   ├── model/              # Model components
│   ├── data/               # Data loading and preprocessing
│   └── evaluation/         # Evaluation and metrics
├── scripts/                # Executable scripts
├── notebooks/              # Jupyter notebooks
└── experiments/            # Training outputs and checkpoints
```

## Workflow

1. **Extract Decoder**: Extract decoder from Whisper Large v3
2. **Prepare Data**: Load and preprocess DOLLMA dataset
3. **Train**: Unsupervised causal LM training
4. **Integrate**: Re-integrate fine-tuned decoder into Whisper
5. **Evaluate**: ASR benchmarking on Azerbaijani test sets

## Dataset

[DOLLMA](https://huggingface.co/datasets/allmalab/DOLLMA) - Large Azerbaijani text corpus

## License

MIT
