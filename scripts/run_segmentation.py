import argparse
import os
import sys

# Ensure the ``src`` package is importable when running this file directly.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.segmentation import run as run_segmentation

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="path to output/preprocessed.npy")
    parser.add_argument("--ridge_thresh", type=float, default=0.25, help="edge cut threshold")
    parser.add_argument("--merge_thresh", type=float, default=0.15, help="weak-edge merge threshold")
    args = parser.parse_args()
    run_segmentation(args.input, ridge_thresh=args.ridge_thresh, merge_thresh=args.merge_thresh)
