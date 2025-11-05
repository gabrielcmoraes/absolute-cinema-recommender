import argparse, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable

def main(args):
    # 1) Exports runtime + copies CSVs to layer folder
    cmd = [PY, str(ROOT/"src"/"models"/"collaborative"/"export_runtime.py"),
           "--factors_npz", args.factors_npz]
    print("[RUN]", " ".join(cmd))
    subprocess.check_call(cmd)

    # 2) Shows what is in the layer
    layer_dir = ROOT/"deployment"/"layers"/"recommender-data"/"python"
    print("\n[INFO] Arquivos na layer:")
    for p in sorted(layer_dir.glob("*")):
        print(" -", p.relative_to(ROOT))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--factors_npz", required=True)
    args = ap.parse_args()
    main(args)
