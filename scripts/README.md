# Training Scripts

End-to-end executable scripts for the Whisper decoder training pipeline.

## Workflow

### 1. Extract Decoder
```bash
python scripts/01_extract_decoder.py --config configs/model_config.yaml --device cuda
```

### 2. Prepare Data
```bash
python scripts/02_prepare_data.py --config configs/training_config.yaml
```

### 3. Train Decoder
```bash
python scripts/03_train_decoder.py \
    --config configs/training_config.yaml \
    --decoder_path ./experiments/decoder_extracted \
    --texts_path ./cache/processed_texts.pkl \
    --device cuda
```

### 4. Integrate Decoder
```bash
python scripts/04_integrate_decoder.py \
    --checkpoint ./experiments/decoder_training/best_model.pt \
    --output_dir ./experiments/whisper_integrated \
    --device cuda
```

### 5. Evaluate
```bash
python scripts/05_evaluate.py \
    --config configs/evaluation_config.yaml \
    --finetuned_model ./experiments/whisper_integrated \
    --device cuda
```

## Notes

- All scripts support `--help` flag for detailed options
- Ensure UV environment is activated before running
- GPU recommended for training and evaluation
- W&B login required if using wandb logging

