from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from guided_filter import guided_filter
from utils import ensure_dir, load_image, save_image, save_parameters


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Guided filter detail enhancement demo.")
    parser.add_argument("--input", required=True, help="Path to the input image.")
    parser.add_argument("--alpha", type=float, default=1.5, help="Detail amplification factor.")
    parser.add_argument("--radius", type=int, default=8, help="Guided filter radius.")
    parser.add_argument("--eps", type=float, default=1e-3, help="Regularization parameter.")
    parser.add_argument("--output-dir", default="data/results/enhance", help="Directory to store results.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    result_dir = ensure_dir(args.output_dir)

    image = load_image(input_path)
    base = guided_filter(image, image, args.radius, args.eps)
    detail = image - base
    enhanced = np.clip(image + args.alpha * detail, 0.0, 1.0)

    comparison = np.concatenate([image, enhanced], axis=1)

    suffix = f"alpha{args.alpha:g}_r{args.radius}_eps{args.eps:.0e}"
    enhanced_path = result_dir / f"{input_path.stem}_{suffix}.png"
    comparison_path = result_dir / f"{input_path.stem}_{suffix}_comparison.png"
    params_path = result_dir / f"{input_path.stem}_{suffix}.json"

    save_image(enhanced_path, enhanced)
    save_image(comparison_path, comparison)
    save_parameters(
        params_path,
        {
            "input": str(input_path),
            "alpha": args.alpha,
            "radius": args.radius,
            "eps": args.eps,
        },
    )

    print(f"Enhanced image saved to {enhanced_path}")
    print(f"Comparison saved to {comparison_path}")
    print(f"Parameters recorded at {params_path}")


if __name__ == "__main__":
    main()
