# IVF Embryo Timelapse Analysis

Deep learning framework for analyzing IVF embryo development from timelapse microscopy images.

## Overview

This project implements a ConvLSTM Autoencoder to extract and analyze temporal features from embryo timelapse sequences. The model learns developmental patterns and can identify anomalies in embryo development.

## Key Results

After training on 13,298 sequences from 704 embryos over 20 epochs, the model achieved:

- Reconstruction loss: 0.0662
- Temporal smoothness loss: 0.0002
- Successfully identified 3 embryos with abnormally fast development (3-4x the normal speed)

The analysis revealed that most embryos follow similar developmental trajectories, with a small subset showing significantly different patterns that warrant further investigation.

## Core Components

**build_index.py** - Scans embryo image directories and creates an index of temporal sequences with configurable window size and overlap.

**dataset_ivf.py** - PyTorch Dataset class that loads image sequences, applies preprocessing (resize, grayscale conversion, normalization), and returns batches for training.

**model_conv_lstm_ae.py** - ConvLSTM Autoencoder architecture with frame-level encoding/decoding and LSTM layers for temporal modeling. Contains 1.6M parameters.

**train_ae.py** - Training script with reconstruction loss and temporal smoothness regularization.

**export_latents_unique.py** - Extracts latent features from trained models and generates trajectory visualizations using PCA projection.

**analyze_all_embryos.py** - Computes statistical features (development speed, trajectory length, variability) and performs anomaly detection.

## Installation

```bash
pip install torch opencv-python pandas numpy tqdm scikit-learn matplotlib ripser persim
```

## Usage

Build the dataset index:
```bash
python3 build_index.py
```

Train the model:
```bash
python3 train_ae.py
```

Extract and analyze features:
```bash
python3 export_latents_unique.py
python3 analyze_all_embryos.py
```

## Analysis Output

The framework generates several types of analysis:

- **Latent trajectories**: 2D projections showing developmental paths
- **Development speed curves**: Quantification of temporal change rates
- **Topological features**: Persistence diagrams identifying structural patterns
- **Statistical summaries**: Population-level distributions and outlier detection

## Results Summary

From analyzing 50 unique embryos, we observed:

- Mean development speed: 0.087 ± 0.063
- Mean trajectory length: 1.31 ± 0.94
- Three outliers detected with speeds of 0.26, 0.30, and 0.37 (significantly higher than the population mean)

These outliers may indicate different developmental mechanisms or quality issues worth investigating.

## Model Architecture

The ConvLSTM Autoencoder processes sequences of 16 frames (128x128 grayscale) and compresses each frame to a 128-dimensional latent representation. LSTM layers model temporal dependencies, and the decoder reconstructs the original sequence. This design allows the model to learn both spatial features within frames and temporal patterns across the sequence.

## License

MIT
