import argparse
import os
import sys

# Ensure the ``src`` package is importable when executing the script directly.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.postprocess import run as run_postprocess

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="path to output/segmentation.npy")
    parser.add_argument("--search_radius", type=float, default=3.0, help="XY radius to attach walls")
    args = parser.parse_args()
    run_postprocess(args.input, search_radius=args.search_radius)
