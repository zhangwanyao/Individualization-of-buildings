import argparse
from src.postprocess import run as run_postprocess

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="path to output/segmentation.npy")
    parser.add_argument("--search_radius", type=float, default=3.0, help="XY radius to attach walls")
    args = parser.parse_args()
    run_postprocess(args.input, search_radius=args.search_radius)
