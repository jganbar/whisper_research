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

### Changed
- Updated `pyproject.toml` to include package configuration for hatchling

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

