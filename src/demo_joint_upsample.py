from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from guided_filter import guided_filter
from utils import (
    ensure_dir,
    load_image,
    save_image,
    save_parameters,
    upsample_to_size,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Joint upsampling with guided filter demo.")
    parser.add_argument("--lowres", required=True, help="Low-resolution input path.")
    parser.add_argument("--guide", required=True, help="High-resolution guidance image path.")
    parser.add_argument("--radius", type=int, default=4, help="Guided filter radius.")
    parser.add_argument("--eps", type=float, default=1e-4, help="Regularization parameter.")
    parser.add_argument(
        "--upsample-method",
        default="nearest",
        choices=("nearest", "bilinear", "bicubic"),
        help="Interpolation used before guided filtering.",
    )
    parser.add_argument(
        "--baseline-method",
        default="bilinear",
        choices=("nearest", "bilinear", "bicubic"),
        help="Baseline interpolation for comparison.",
    )
    parser.add_argument("--output-dir", default="data/results/joint_upsample", help="Directory to store results.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lowres_path = Path(args.lowres)
    guide_path = Path(args.guide)
    result_dir = ensure_dir(args.output_dir)

    lowres = load_image(lowres_path)
    guide = load_image(guide_path)

    target_hw = guide.shape[:2]
    upsampled = upsample_to_size(lowres, target_hw, method=args.upsample_method)
    baseline = upsample_to_size(lowres, target_hw, method=args.baseline_method)
    filtered = guided_filter(guide, upsampled, args.radius, args.eps)

    suffix = (
        f"{lowres_path.stem}_to_{guide_path.stem}_"
        f"{args.upsample_method}_r{args.radius}_eps{args.eps:.0e}"
    )
    filtered_path = result_dir / f"{suffix}.png"
    baseline_path = result_dir / f"{suffix}_baseline.png"
    comparison_path = result_dir / f"{suffix}_comparison.png"
    params_path = result_dir / f"{suffix}.json"

    save_image(filtered_path, filtered)
    save_image(baseline_path, baseline)
    comparison = np.concatenate([baseline, filtered], axis=1)
    save_image(comparison_path, comparison)
    save_parameters(
        params_path,
        {
            "lowres": str(lowres_path),
            "guide": str(guide_path),
            "radius": args.radius,
            "eps": args.eps,
            "upsample_method": args.upsample_method,
            "baseline_method": args.baseline_method,
        },
    )

    print(f"Filtered output saved to {filtered_path}")
    print(f"Baseline saved to {baseline_path}")
    print(f"Comparison saved to {comparison_path}")
    print(f"Parameters recorded at {params_path}")


if __name__ == "__main__":
    main()
