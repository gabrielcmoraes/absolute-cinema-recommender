import numpy as np
import pandas as pd
from pathlib import Path

def load_csv(path: Path, **kwargs):
    return pd.read_csv(path, **kwargs)

def save_npz(path: Path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
