#!/bin/bash
# Training wrapper script for CHTC
# Uses shared dataset and new ConvLSTM Autoencoder model

# Don't exit on error
set +e

# Use shared dataset
echo "Linking dataset..."
ln -sfn /project/bhaskar_group/ivf data || echo "Warning: Could not link dataset"

# Add current directory to PYTHONPATH FIRST (so train.py can find dataset_ivf.py)
export PYTHONPATH="$PWD:$PYTHONPATH"
echo "PYTHONPATH set to: $PYTHONPATH"

# Install dependencies into local directory (container may not have all packages)
echo "Installing dependencies..."
PYDEPS="$PWD/pydeps"
mkdir -p "$PYDEPS"
python3 -m pip install --no-cache-dir --upgrade pip || echo "Warning: pip upgrade failed"
python3 -m pip install --no-cache-dir opencv-python pandas numpy tqdm scikit-learn matplotlib ripser persim -t "$PYDEPS" || echo "Warning: Some packages failed to install"
export PYTHONPATH="$PYDEPS:$PWD:$PYTHONPATH"

# Build index if needed
if [ ! -f index.csv ]; then
    echo "Building index.csv..."
    if [ -f "build_index.py" ]; then
        python3 build_index.py || echo "Index build failed, continuing..."
    else
        echo "build_index.py not found, will try to continue without index.csv"
    fi
fi

# Verify dataset_ivf.py exists
echo "=== Checking for dataset_ivf.py ==="
if [ -f "dataset_ivf.py" ]; then
    echo "✓ dataset_ivf.py found in current directory"
else
    echo "⚠ ERROR: dataset_ivf.py not found!"
    ls -la *.py 2>/dev/null || echo "No .py files found"
fi

# Test import
echo "=== Testing dataset_ivf import ==="
python3 -c "import sys; sys.path.insert(0, '.'); from dataset_ivf import IVFSequenceDataset; print('✓ Successfully imported dataset_ivf')" || echo "⚠ Import test failed!"

# Run training
echo "Starting training..."
python3 train.py \
    --index_csv index.csv \
    --batch_size 8 \
    --seq_len 20 \
    --num_epochs 50 \
    --learning_rate 3e-4 \
    --save_dir checkpoints \
    --log_dir logs || echo "Training failed!"

echo "Training script completed!"
