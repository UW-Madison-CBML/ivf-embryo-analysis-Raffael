"""
Build index.csv for IVF dataset
Works on CHTC with data symlink pointing to /project/bhaskar_group/ivf
"""
import re
import glob
import csv
from pathlib import Path
from tqdm import tqdm

# Use relative path - run_train.sh creates symlink: data -> /project/bhaskar_group/ivf
DATASET_ROOT = Path("data")
OUT_CSV = "index.csv"
T = 16                 # Sequence length (frames)
SUBSAMPLE = 3          # Take every 3rd frame
WINDOW_STRIDE = T // 2   # 50% overlap

run_pat = re.compile(r'RUN[_\- ]?(\d+)', re.I)
num_pat = re.compile(r'(\d+)')

def parse_sort_key(p: Path):
    """Parse sorting key from path name"""
    name = p.name
    # Extract RUN number
    run_m = run_pat.search(name)
    run_idx = int(run_m.group(1)) if run_m else 10**9  # No RUN number goes last
    # Extract all numbers from filename
    nums = [int(x) for x in num_pat.findall(name)]
    nums = tuple(nums) if nums else ()
    # File modification time (nanoseconds)
    mtime = p.stat().st_mtime_ns
    return (run_idx, nums, mtime)

def list_frames(cell_dir: Path):
    """List all image frames in a cell directory"""
    exts = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
    frames = []
    for ext in exts:
        frames += [Path(p) for p in glob.glob(str(cell_dir / ext))]
    frames = [p for p in frames if p.exists() and p.stat().st_size > 0]
    frames.sort(key=parse_sort_key)
    return frames

def main():
    root = Path(DATASET_ROOT)
    if not root.exists():
        print(f"ERROR: Dataset root '{root}' does not exist!")
        print(f"Current directory: {Path.cwd()}")
        print(f"Available files: {list(Path('.').iterdir())[:10]}")
        return
    
    cell_dirs = [p for p in root.iterdir() if p.is_dir()]
    if not cell_dirs:
        print(f"ERROR: No cell directories found in '{root}'!")
        return
    
    print(f"Found {len(cell_dirs)} cell directories in {root}")
    
    rows = []
    for cell in tqdm(sorted(cell_dirs, key=lambda x: x.name), desc="Processing cells"):
        frames = list_frames(cell)
        if not frames: 
            continue
        # Temporal subsampling
        frames = frames[::SUBSAMPLE]
        if len(frames) < T: 
            continue
        # Sliding window
        for start in range(0, len(frames) - T + 1, WINDOW_STRIDE):
            seq = frames[start:start+T]
            rows.append({
                "cell_id": cell.name,
                "start_idx": start,
                "paths": "|".join(str(p) for p in seq)
            })
    
    # Write CSV
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["cell_id", "start_idx", "paths"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    
    print(f"✓ Wrote {OUT_CSV} with {len(rows)} sequences")

if __name__ == "__main__":
    main()

