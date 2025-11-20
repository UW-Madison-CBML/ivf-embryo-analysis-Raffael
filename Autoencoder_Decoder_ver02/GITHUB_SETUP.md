# GitHub Repository Setup

## Target Repository Structure

Upload these files to:
```
https://github.com/UW-Madison-CBML/ivf
├── Raffael/
│   └── 2025-11-19/  (current date)
│       ├── conv_lstm.py
│       ├── model.py
│       ├── losses.py
│       ├── train.py
│       ├── test_model.py
│       ├── verify_data_connection.py
│       └── README.md
```

## Steps to Upload

### Option 1: Using GitHub Web Interface

1. Go to https://github.com/UW-Madison-CBML/ivf
2. Click "Add file" → "Create new file"
3. Create folder structure: `Raffael/2025-11-16/`
4. Upload each file one by one

### Option 2: Using Git Command Line

```bash
# Clone the repository (if not already cloned)
git clone https://github.com/UW-Madison-CBML/ivf.git
cd ivf

# Create directory structure
mkdir -p Raffael/2025-11-19

# Copy files from local
cp -r /path/to/Autoencoder_Decoder_ver02/* Raffael/2025-11-19/

# Remove unnecessary files (keep only code and essential docs)
cd Raffael/2025-11-19
rm -f QUALITY_ASSESSMENT.md DATA_CONNECTION_VERIFIED.md GITHUB_SETUP.md

# Commit and push
cd ../..
git add Raffael/
git commit -m "Add ConvLSTM Autoencoder implementation (Raffael/2025-11-19)"
git push
```

## Files to Include

**Essential files only:**
- `conv_lstm.py` - ConvLSTM implementation
- `model.py` - Complete model
- `losses.py` - Loss functions
- `train.py` - Training script
- `test_model.py` - Test script
- `verify_data_connection.py` - Data verification
- `README.md` - Documentation

**Do NOT include:**
- `QUALITY_ASSESSMENT.md` (removed)
- `DATA_CONNECTION_VERIFIED.md` (removed)
- `GITHUB_SETUP.md` (this file, remove after setup)
- Checkpoints/logs (generated during training)

## Date Folder Naming

Use format: `YYYY-MM-DD` (e.g., `2025-11-19`)

Current date folder: `2025-11-19`

