# Training Scripts

End-to-end executable scripts for the Whisper decoder training pipeline.

## Workflow

### 1. Extract Decoder
Extract the decoder component from Whisper Large v3:
```bash
python scripts/01_extract_decoder.py --config configs/model_config.yaml --device cuda
```

### 2. Train Decoder
Train the decoder on your HuggingFace text dataset:
```bash
python scripts/03_train_decoder.py \
    --config configs/training_config.yaml \
    --decoder_path ./experiments/decoder_extracted \
    --device cuda
```

**Note:** Make sure to update `configs/training_config.yaml` with your dataset name first.

### 3. Integrate Decoder
Re-integrate the trained decoder back into Whisper:
```bash
python scripts/04_integrate_decoder.py \
    --checkpoint ./experiments/decoder_training/best_model.pt \
    --output_dir ./experiments/whisper_integrated \
    --device cuda
```

### 4. Evaluate
Evaluate the improved model on ASR tasks:
```bash
python scripts/05_evaluate.py \
    --config configs/evaluation_config.yaml \
    --finetuned_model ./experiments/whisper_integrated \
    --device cuda
```

## Dataset Configuration

Before training, update `configs/training_config.yaml`:

```yaml
dataset:
  name: "YOUR_USERNAME/YOUR_DATASET_NAME"  # Your HuggingFace dataset
  text_column: "text"                       # Column name with text data
```

Your dataset should:
- Be hosted on HuggingFace Hub
- Have a column with text data (default: "text")
- Be in any standard format (parquet, CSV, JSON, etc.)

## Monitoring

Start TensorBoard to monitor training:
```bash
tensorboard --logdir ./experiments/runs
```

Then open http://localhost:6006 in your browser.

## Notes

- All scripts support `--help` flag for detailed options
- Ensure UV environment is activated before running
- GPU recommended for training and evaluation
- W&B login required if using wandb logging (optional)
