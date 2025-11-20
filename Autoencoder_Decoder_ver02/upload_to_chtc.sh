#!/bin/bash
# Upload files directly to CHTC (bypasses GitHub auth issue)

set -e

CHTC_USER="rho9"
CHTC_HOST="ap2001.chtc.wisc.edu"
REMOTE_DIR="~/ivf_train"
LOCAL_DIR="/Users/grnho/Desktop/Project IVF/Code/Autoencoder_Decoder_ver02"

echo "=== Uploading ConvLSTM Autoencoder to CHTC ==="
echo ""

# Files to upload
FILES=(
    "conv_lstm.py"
    "model.py"
    "losses.py"
    "train.py"
    "test_model.py"
    "verify_data_connection.py"
    "run_train.sh"
    "train_h200_lab.sub"
    "README.md"
)

echo "Uploading files to ${CHTC_USER}@${CHTC_HOST}:${REMOTE_DIR}"
echo ""

# Create remote directory
ssh ${CHTC_USER}@${CHTC_HOST} "mkdir -p ${REMOTE_DIR}"

# Upload each file
for file in "${FILES[@]}"; do
    if [ -f "${LOCAL_DIR}/${file}" ]; then
        echo "Uploading ${file}..."
        scp "${LOCAL_DIR}/${file}" ${CHTC_USER}@${CHTC_HOST}:${REMOTE_DIR}/
    else
        echo "Warning: ${file} not found, skipping..."
    fi
done

echo ""
echo "=== Upload Complete ==="
echo ""
echo "Next steps on CHTC:"
echo "  ssh ${CHTC_USER}@${CHTC_HOST}"
echo "  cd ${REMOTE_DIR}"
echo "  chmod +x run_train.sh"
echo "  mkdir -p logs"
echo "  condor_submit train_h200_lab.sub"
