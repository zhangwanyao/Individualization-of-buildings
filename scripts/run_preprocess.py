import argparse
import os
import sys

# Allow running as a script from the project root by ensuring the parent
# directory (which contains the ``src`` package) is on ``sys.path``.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocess import run as run_preprocess

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="path to .laz/.las")
    parser.add_argument("--grid", type=float, default=0.5, help="DSM grid size in meters")
    parser.add_argument("--min_points", type=int, default=100, help="min roof points per region")
    args = parser.parse_args()
    run_preprocess(args.input, grid_size=args.grid, min_region_points=args.min_points)