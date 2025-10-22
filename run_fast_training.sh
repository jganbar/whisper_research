#!/bin/bash
# Fast Training Script for Whisper Decoder
# Optimized for RTX 4090 with CUDA

# Stop any existing training for this script only
echo "🛑 Stopping any running training..."
PIDS=$(pgrep -f "python3 scripts/03_train_decoder.py")
if [ -n "$PIDS" ]; then
  echo "   Active PIDs: $PIDS"
  kill -INT $PIDS 2>/dev/null || true
  sleep 2
else
  echo "   No existing training processes found."
fi

# Setup
cd /home/javidan/whisper_research
rm -f training.log

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source .venv/bin/activate

# Set CUDA device (GPU 1 - RTX 4090)
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1

# Verify GPU
echo ""
echo "🚀 GPU Information:"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader | grep "1,"
echo ""

# Run training with optimized settings
echo "🏋️ Starting FAST training (100K samples on RTX 4090)..."
echo "   - Batch size: 96"
echo "   - Gradient accumulation: 4 (effective batch: 384)"
echo "   - Workers: 12"
echo "   - Mixed precision: BF16"
echo "   - Expected time: ~2-3 hours"
echo ""

python3 scripts/03_train_decoder.py \
  --config configs/training_config.yaml \
  --decoder_path ./experiments/decoder_extracted \
  --device cuda 2>&1 | tee training.log

echo ""
echo "✅ Training finished! Check training.log for details."
