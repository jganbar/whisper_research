#!/bin/bash
# GPU Selection and Training Script
# Lists GPUs by PCI bus ID and allows selection

cd /home/javidan/whisper_research

echo "======================================================================"
echo "🎮 Available GPUs (sorted by PCI Bus ID):"
echo "======================================================================"
nvidia-smi --query-gpu=index,pci.bus_id,name,memory.total,memory.free,memory.used,utilization.gpu --format=csv,noheader,nounits | \
  awk -F', ' '{printf "GPU %s | PCI: %s | %s\n        Memory: %s MB total, %s MB free, %s MB used | Utilization: %s%%\n\n", $1, $2, $3, $4, $5, $6, $7}'

echo "======================================================================"
echo ""

# Check if GPU index is provided as argument
if [ -z "$1" ]; then
    echo "Usage: $0 <gpu_index>"
    echo ""
    echo "Example: $0 2    # Use GPU 2 (PCI: 00000000:42:00.0)"
    echo ""
    exit 1
fi

GPU_INDEX=$1

# Verify GPU index is valid
GPU_COUNT=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
if [ "$GPU_INDEX" -ge "$GPU_COUNT" ] || [ "$GPU_INDEX" -lt 0 ]; then
    echo "❌ Invalid GPU index: $GPU_INDEX"
    echo "   Available GPUs: 0-$((GPU_COUNT-1))"
    exit 1
fi

# Get GPU info
GPU_INFO=$(nvidia-smi --query-gpu=name,pci.bus_id,memory.free --format=csv,noheader,nounits -i $GPU_INDEX)
GPU_NAME=$(echo $GPU_INFO | cut -d',' -f1)
GPU_PCI=$(echo $GPU_INFO | cut -d',' -f2 | xargs)
GPU_FREE=$(echo $GPU_INFO | cut -d',' -f3 | xargs)

echo "======================================================================"
echo "✅ Selected GPU $GPU_INDEX:"
echo "   Name: $GPU_NAME"
echo "   PCI Bus ID: $GPU_PCI"
echo "   Free Memory: $GPU_FREE MB"
echo "======================================================================"
echo ""

# Stop any existing training
echo "🛑 Stopping any running training..."
pkill -9 python3
sleep 2

# Clean up old logs
rm -f training.log

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source .venv/bin/activate

# Set CUDA device
export CUDA_VISIBLE_DEVICES=$GPU_INDEX

echo ""
echo "======================================================================"
echo "🏋️ Starting FAST training on GPU $GPU_INDEX ($GPU_NAME)..."
echo "======================================================================"
echo "   - PCI Bus ID: $GPU_PCI"
echo "   - Batch size: 96"
echo "   - Gradient accumulation: 4 (effective batch: 384)"
echo "   - Workers: 12"
echo "   - Mixed precision: BF16"
echo "   - Dataset: 100K training + 10K validation samples"
echo "   - Expected time: ~2-3 hours"
echo ""
echo "💡 Monitor progress:"
echo "   - Log file: tail -f training.log"
echo "   - TensorBoard: tensorboard --logdir ./experiments/runs"
echo "   - GPU usage: watch -n 1 nvidia-smi"
echo ""

# Run training
python3 scripts/03_train_decoder.py \
  --config configs/training_config.yaml \
  --decoder_path ./experiments/decoder_extracted \
  --device cuda 2>&1 | tee training.log

echo ""
echo "======================================================================"
echo "✅ Training finished!"
echo "======================================================================"
echo "Check training.log for full details"
echo ""

