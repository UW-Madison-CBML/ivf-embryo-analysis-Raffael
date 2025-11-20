# dataset_ivf.py
# PyTorch Dataset for IVF embryo timelapse sequences
from PIL import Image
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class IVFSequenceDataset(Dataset):
    """
    Dataset for loading IVF embryo timelapse sequences.
    
    Args:
        index_csv: Path to CSV file with columns: cell_id, start_idx, paths
        resize: Target image size (default: 128)
        norm: Normalization method - "minmax01" or "zscore" (default: "minmax01")
    """
    
    def __init__(self, index_csv, resize=128, norm="minmax01"):
        self.df = pd.read_csv(index_csv)
        self.resize = resize
        self.norm = norm

    def _read_gray(self, path):
        """Read and preprocess a single grayscale image using Pillow"""
        try:
            img = Image.open(path).convert("L")  # Convert to grayscale
            img = img.resize((self.resize, self.resize), Image.BILINEAR)
            arr = np.array(img, dtype=np.float32)
            # Light denoising using simple box filter (alternative to GaussianBlur)
            # For simplicity, we skip denoising here - can add if needed
            return arr
        except Exception as e:
            raise FileNotFoundError(f"Could not read image: {path}, error: {e}")

    def _normalize_video(self, vol):  # vol: [T, H, W]
        """Normalize video sequence per-sequence"""
        if self.norm == "zscore":
            m, s = vol.mean(), vol.std() + 1e-6
            vol = (vol - m) / s
        elif self.norm == "minmax01":
            lo, hi = np.percentile(vol, 1), np.percentile(vol, 99)
            vol = (vol - lo) / (hi - lo + 1e-6)
            vol = np.clip(vol, 0, 1)
        return vol

    def __getitem__(self, idx):
        """Get a single sequence"""
        paths = self.df.iloc[idx]["paths"].split("|")
        frames = [self._read_gray(p) for p in paths]
        vol = np.stack(frames, axis=0)  # [T, H, W]
        vol = self._normalize_video(vol)
        vol = vol[:, None, :, :]  # [T, 1, H, W] - add channel dimension
        return torch.from_numpy(vol), self.df.iloc[idx]["cell_id"]

    def __len__(self):
        return len(self.df)

