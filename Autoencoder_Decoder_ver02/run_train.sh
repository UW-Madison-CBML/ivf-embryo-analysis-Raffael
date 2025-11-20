#!/usr/bin/env bash
set -euo pipefail

echo "=== [run_train.sh] Starting job ==="
echo "CWD: $(pwd)"
echo "Contents in CWD:"
ls

# Ensure data symlink points to project (only available on GPU node)
if [ ! -L data ]; then
  echo "[run_train] Creating data symlink -> /project/bhaskar_group/ivf"
  ln -s /project/bhaskar_group/ivf data
fi
echo "[run_train] data symlink:"
ls -ld data || echo "data symlink missing"

# Build index.csv on GPU node
echo "[run_train] Building index.csv on GPU node..."
python -u build_index.py

echo "[run_train] After build_index, check index.csv:"
ls -lh index.csv || { echo "✗ index.csv NOT FOUND after build_index.py"; exit 1; }

# Set PYTHONPATH
export PYTHONPATH="$PWD:$PYTHONPATH"

# Install dependencies into local directory (container may not have all packages)
echo "[run_train] Installing dependencies..."
PYDEPS="$PWD/pydeps"
mkdir -p "$PYDEPS"
python3 -m pip install --no-cache-dir --upgrade pip || echo "Warning: pip upgrade failed"

# Install all required packages (PyTorch should already be in container)
python3 -m pip install --no-cache-dir \
    pillow \
    pandas \
    numpy \
    tqdm \
    scikit-learn \
    matplotlib \
    ripser \
    persim \
    -t "$PYDEPS" || echo "Warning: Some packages failed to install"

export PYTHONPATH="$PYDEPS:$PYTHONPATH"

# Start training
echo "[run_train] Starting training..."
python -u train.py \
    --index_csv index.csv \
    --batch_size 8 \
    --seq_len 20 \
    --num_epochs 50 \
    --learning_rate 3e-4 \
    --save_dir checkpoints \
    --log_dir logs

echo "=== [run_train.sh] Training finished ==="
