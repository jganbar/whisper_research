# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Alternative decoder extraction script using openai-whisper library (`01_extract_decoder_openai.py`)
  - Addresses HuggingFace Hub download issues (500 Internal Server errors)
  - Uses OpenAI's direct download mechanism instead of HuggingFace Hub
  - Fully compatible with existing training pipeline
- Project setup and dependency management
  - Virtual environment setup with UV
  - All dependencies installed (PyTorch 2.9.0 + CUDA 12.8)
  - openai-whisper library integration
- Support for any HuggingFace text dataset with configurable column names

### Changed
- **BREAKING**: Simplified dataset loading workflow
  - Removed `02_prepare_data.py` script (no longer needed)
  - Direct loading from HuggingFace Hub without preprocessing/caching
  - Dataset must be pre-prepared and hosted on HuggingFace
- Streamlined `dataset_loader.py` to work directly with HuggingFace datasets
  - Removed DOLLMA-specific multi-config loading
  - Removed complex preprocessing and chunking logic
  - Added `load_text_dataset()` for simple dataset loading
  - Added `prepare_texts()` for basic text extraction
- Simplified `preprocessing.py` to minimal text cleaning
  - Removed language-specific cleaning (Azerbaijani)
  - Removed URL/email removal
  - Kept only unicode normalization and whitespace cleaning
- Updated `03_train_decoder.py` to load dataset directly (no pickle cache)
- Updated training configuration schema in `training_config.yaml`
  - Removed preprocessing options (lowercase, remove_punctuation, min/max length)
  - Added `text_column` parameter
  - Added `max_seq_length` parameter
- Updated documentation to reflect simplified workflow
  - README.md with streamlined 4-step process
  - scripts/README.md with updated commands
  - Dataset requirements and configuration guide
- Updated `pyproject.toml` to include package configuration for hatchling

### Removed
- `02_prepare_data.py` script (dataset preparation step)
- Complex DOLLMA-specific loading logic
- Pickle-based dataset caching
- Advanced text preprocessing options
- `load_dollma_dataset()` function
- `prepare_dataset()` function (replaced with `prepare_texts()`)
- Azerbaijani-specific text cleaning functions

### Fixed
- Build system configuration to properly recognize src package structure

## [0.1.0] - 2025-10-20

### Initial Release
- Project structure and basic implementation
- Whisper Large v3 decoder extraction
- DOLLMA dataset loading and preprocessing
- Decoder training with TensorBoard logging
- Model integration and evaluation
- Comprehensive documentation

