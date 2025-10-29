from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from guided_filter import guided_filter
from utils import ensure_dir, load_image, save_image, save_parameters, upsample_to_size


def list_images(root: Path, patterns: Sequence[str]) -> list[Path]:
    files: list[Path] = []
    for pattern in patterns:
        files.extend(sorted(root.glob(pattern)))
    return files


def run_smoothing(image_path: Path, radii: Sequence[int], eps_values: Sequence[float], output_dir: Path) -> None:
    image = load_image(image_path)
    for radius, eps in itertools.product(radii, eps_values):
        smoothed = guided_filter(image, image, radius, eps)
        comparison = np.concatenate([image, smoothed], axis=1)
        suffix = f"r{radius}_eps{eps:.0e}"
        stem = image_path.stem
        save_image(output_dir / f"{stem}_{suffix}.png", smoothed)
        save_image(output_dir / f"{stem}_{suffix}_comparison.png", comparison)
        save_parameters(
            output_dir / f"{stem}_{suffix}.json",
            {
                "input": str(image_path),
                "radius": radius,
                "eps": eps,
                "mode": "smoothing",
            },
        )


def run_enhancement(
    image_path: Path,
    alphas: Sequence[float],
    radius: int,
    eps: float,
    output_dir: Path,
) -> None:
    image = load_image(image_path)
    base = guided_filter(image, image, radius, eps)
    detail = image - base

    for alpha in alphas:
        enhanced = np.clip(image + alpha * detail, 0.0, 1.0)
        comparison = np.concatenate([image, enhanced], axis=1)
        suffix = f"alpha{alpha:g}_r{radius}_eps{eps:.0e}"
        stem = image_path.stem
        save_image(output_dir / f"{stem}_{suffix}.png", enhanced)
        save_image(output_dir / f"{stem}_{suffix}_comparison.png", comparison)
        save_parameters(
            output_dir / f"{stem}_{suffix}.json",
            {
                "input": str(image_path),
                "alpha": alpha,
                "radius": radius,
                "eps": eps,
                "mode": "enhance",
            },
        )


def run_joint_upsample(
    image_path: Path,
    radius: int,
    eps: float,
    scale: float,
    upsample_method: str,
    baseline_method: str,
    output_dir: Path,
) -> None:
    guide = load_image(image_path)
    height, width = guide.shape[:2]
    low_h = max(1, int(height * scale))
    low_w = max(1, int(width * scale))

    lowres = upsample_to_size(guide, (low_h, low_w), method="bilinear")
    upsampled = upsample_to_size(lowres, (height, width), method=upsample_method)
    baseline = upsample_to_size(lowres, (height, width), method=baseline_method)
    filtered = guided_filter(guide, upsampled, radius, eps)

    stem = image_path.stem
    suffix = f"{stem}_scale{scale:.2f}_{upsample_method}_r{radius}_eps{eps:.0e}"

    save_image(output_dir / f"{suffix}_filtered.png", filtered)
    save_image(output_dir / f"{suffix}_baseline.png", baseline)
    comparison = np.concatenate([baseline, filtered], axis=1)
    save_image(output_dir / f"{suffix}_comparison.png", comparison)
    save_parameters(
        output_dir / f"{suffix}.json",
        {
            "guide": str(image_path),
            "scale": scale,
            "radius": radius,
            "eps": eps,
            "upsample_method": upsample_method,
            "baseline_method": baseline_method,
            "mode": "joint_upsample",
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch runner for guided filter demos.")
    parser.add_argument("--input-root", default="data/input", help="Root directory containing image categories.")
    parser.add_argument("--results-root", default="data/results", help="Directory to store generated results.")
    parser.add_argument(
        "--portrait-patterns",
        nargs="+",
        default=("portrait/*.png", "portrait/*.jpg"),
        help="Glob patterns (relative to input-root) for portrait images.",
    )
    parser.add_argument(
        "--landscape-patterns",
        nargs="+",
        default=("landscape/*.png", "landscape/*.jpg"),
        help="Glob patterns (relative to input-root) for landscape images.",
    )
    parser.add_argument("--smoothing-radii", nargs="+", type=int, default=(4, 8, 16), help="Radii for smoothing demo.")
    parser.add_argument(
        "--smoothing-eps",
        nargs="+",
        type=float,
        default=(1e-4, 1e-3, 1e-2),
        help="Epsilon values for smoothing demo.",
    )
    parser.add_argument(
        "--enhance-alphas",
        nargs="+",
        type=float,
        default=(1.0, 1.5, 2.0),
        help="Alpha values for detail enhancement.",
    )
    parser.add_argument("--enhance-radius", type=int, default=8, help="Radius for detail enhancement.")
    parser.add_argument("--enhance-eps", type=float, default=1e-3, help="Epsilon for detail enhancement.")
    parser.add_argument("--upsample-radius", type=int, default=4, help="Radius for joint upsampling.")
    parser.add_argument("--upsample-eps", type=float, default=1e-4, help="Epsilon for joint upsampling.")
    parser.add_argument("--upsample-scale", type=float, default=0.25, help="Scale factor to create low-res inputs.")
    parser.add_argument(
        "--upsample-method",
        default="nearest",
        choices=("nearest", "bilinear", "bicubic"),
        help="Interpolation before guided filtering.",
    )
    parser.add_argument(
        "--baseline-method",
        default="bilinear",
        choices=("nearest", "bilinear", "bicubic"),
        help="Baseline interpolation for comparison.",
    )
    parser.add_argument(
        "--skip",
        nargs="+",
        choices=("smoothing", "enhance", "joint"),
        default=(),
        help="Skip specified stages.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_root = Path(args.input_root)
    results_root = Path(args.results_root)

    portraits = list_images(input_root, args.portrait_patterns)
    landscapes = list_images(input_root, args.landscape_patterns)
    if not portraits and "enhance" not in args.skip:
        print("Warning: no portrait images found for enhancement.")
    if not landscapes and "smoothing" not in args.skip:
        print("Warning: no landscape images found for smoothing/joint upsampling.")

    if "smoothing" not in args.skip:
        smoothing_dir = ensure_dir(results_root / "smoothing")
        for image_path in landscapes + portraits:
            run_smoothing(image_path, args.smoothing_radii, args.smoothing_eps, smoothing_dir)

    if "enhance" not in args.skip and portraits:
        enhance_dir = ensure_dir(results_root / "enhance")
        for image_path in portraits:
            run_enhancement(
                image_path,
                args.enhance_alphas,
                args.enhance_radius,
                args.enhance_eps,
                enhance_dir,
            )

    if "joint" not in args.skip and landscapes:
        joint_dir = ensure_dir(results_root / "joint_upsample")
        for image_path in landscapes:
            run_joint_upsample(
                image_path,
                radius=args.upsample_radius,
                eps=args.upsample_eps,
                scale=args.upsample_scale,
                upsample_method=args.upsample_method,
                baseline_method=args.baseline_method,
                output_dir=joint_dir,
            )


if __name__ == "__main__":
    main()
