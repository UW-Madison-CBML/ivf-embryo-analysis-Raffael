#!/bin/bash
# Training wrapper script for CHTC
# Uses shared dataset and new ConvLSTM Autoencoder model

# CRITICAL: Create results.tgz IMMEDIATELY at script start (BEFORE anything else)
# Use absolute path and multiple fallbacks to ensure it exists
cd "$(pwd)" || cd "$HOME" || true
echo "Initializing results.tgz in: $(pwd)"
echo "Job started at $(date)" > job_info.txt 2>/dev/null || echo "Job started" > job_info.txt
tar -czf results.tgz job_info.txt 2>/dev/null || \
    (echo "empty" > empty.txt && tar -czf results.tgz empty.txt 2>/dev/null && rm -f empty.txt) || \
    touch results.tgz || \
    (echo "fallback" > results.tgz && chmod 644 results.tgz)
rm -f job_info.txt empty.txt
ls -lh results.tgz 2>/dev/null || echo "WARNING: results.tgz creation may have failed"

# Function to update results.tgz (called at end and on exit)
update_results_tgz() {
    echo "Updating results.tgz..."
    if [ -d "checkpoints" ] && [ -d "logs" ]; then
        tar -czf results.tgz checkpoints/ logs/ 2>/dev/null || tar -czf results.tgz logs/ 2>/dev/null || true
    elif [ -d "logs" ]; then
        tar -czf results.tgz logs/ 2>/dev/null || true
    elif [ -d "checkpoints" ]; then
        tar -czf results.tgz checkpoints/ 2>/dev/null || true
    fi
    # Ensure file exists (final fallback)
    if [ ! -f results.tgz ]; then
        echo "empty" > empty.txt && tar -czf results.tgz empty.txt 2>/dev/null && rm -f empty.txt || touch results.tgz
    fi
    echo "Results packaged: $(ls -lh results.tgz 2>/dev/null || echo 'results.tgz exists')"
}

# Trap to ensure results.tgz is updated even if script exits unexpectedly
trap update_results_tgz EXIT ERR

# Don't exit on error
set +e

# Use shared dataset
echo "Linking dataset..."
ln -sfn /project/bhaskar_group/ivf data || echo "Warning: Could not link dataset"

# Add current directory to PYTHONPATH FIRST (so train.py can find dataset_ivf.py)
export PYTHONPATH="$PWD:$PYTHONPATH"
echo "PYTHONPATH set to: $PYTHONPATH"
echo "Files in current directory:"
ls -lh *.py 2>/dev/null | head -10 || echo "No .py files found"

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
    elif [ -f "../../Code/build_index.py" ]; then
        python3 ../../Code/build_index.py || echo "Index build failed, continuing..."
    elif [ -f "../../../Code/build_index.py" ]; then
        python3 ../../../Code/build_index.py || echo "Index build failed, continuing..."
    else
        echo "build_index.py not found, will try to continue without index.csv"
    fi
fi

# PYTHONPATH already includes $PWD, verify dataset_ivf.py exists
echo "=== Checking for dataset_ivf.py ==="
echo "Current directory: $(pwd)"
echo "PYTHONPATH: $PYTHONPATH"
echo "Python files in current directory:"
ls -lh *.py 2>/dev/null | head -10 || echo "No .py files found!"

if [ -f "dataset_ivf.py" ]; then
    echo "✓ dataset_ivf.py found in current directory"
    echo "File size: $(ls -lh dataset_ivf.py | awk '{print $5}')"
else
    echo "⚠ ERROR: dataset_ivf.py not found in current directory!"
    echo "All files in current directory:"
    ls -la
    echo "This is a critical error - training will fail!"
fi

# Ensure PYTHONPATH includes current directory explicitly
export PYTHONPATH="$PWD:$PYTHONPATH"
echo "Final PYTHONPATH: $PYTHONPATH"

# Test import before running training
echo "=== Testing dataset_ivf import ==="
python3 -c "import sys; sys.path.insert(0, '.'); from dataset_ivf import IVFSequenceDataset; print('✓ Successfully imported dataset_ivf')" || echo "⚠ Import test failed!"

# Run training (don't exit on error)
echo "Starting training..."
python3 train.py \
    --index_csv index.csv \
    --batch_size 8 \
    --seq_len 20 \
    --num_epochs 50 \
    --learning_rate 3e-4 \
    --save_dir checkpoints \
    --log_dir logs || echo "Training failed, but continuing to package results..."

# Update results.tgz with actual results
update_results_tgz

echo "Training script completed!"
