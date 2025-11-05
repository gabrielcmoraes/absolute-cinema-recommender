from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def data_raw_dir():
    a = ROOT / "data" / "raw" / "ml-latest"
    b = ROOT / "data" / "raw"
    return a if a.exists() else b
