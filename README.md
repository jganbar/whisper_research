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

### 1. Extract Decoder
```bash
python scripts/01_extract_decoder.py --device cuda
```

### 2. Prepare Data
```bash
python scripts/02_prepare_data.py
```

### 3. Train Decoder
```bash
python scripts/03_train_decoder.py --device cuda
```

**Monitor training with TensorBoard:**
```bash
tensorboard --logdir ./experiments/runs
```
Then open http://localhost:6006 in your browser.

### 4. Integrate Decoder
```bash
python scripts/04_integrate_decoder.py \
    --checkpoint ./experiments/decoder_training/best_model.pt
```

### 5. Evaluate
```bash
python scripts/05_evaluate.py \
    --finetuned_model ./experiments/whisper_integrated
```

## Monitoring Training

This project uses **TensorBoard** for experiment tracking:
- Training and validation loss
- Perplexity metrics
- Learning rate schedules
- Real-time training progress

Start TensorBoard before or during training:
```bash
tensorboard --logdir ./experiments/runs
```

For W&B (optional), install separately:
```bash
uv pip install ".[wandb]"
wandb login
```
And enable in `configs/training_config.yaml`.

## Dataset

[DOLLMA](https://huggingface.co/datasets/allmalab/DOLLMA) - Large Azerbaijani text corpus

## License

MIT
