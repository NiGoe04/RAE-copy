"""Create a gFID reference NPZ using the same path as generated samples.

This intentionally builds one NHWC uint8 image array and calls the existing
repository helper once. The helper itself performs feature extraction in
batches, exactly as it does for generated samples.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from pack_images import center_crop_arr
from src.eval.fid import _compute_inception_moments_from_arr


def find_images(image_dir: Path):
    suffixes = {".jpg", ".jpeg", ".png", ".webp"}
    return sorted(
        path for path in image_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in suffixes
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", choices=("cuda", "cpu"), default=None)
    args = parser.parse_args()

    if not args.image_dir.is_dir():
        raise FileNotFoundError(args.image_dir)

    paths = find_images(args.image_dir)
    if not paths:
        raise ValueError(f"No supported images found in {args.image_dir}")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Found {len(paths)} images")
    print(f"Preprocessing images on CPU; Inception extraction on {device}")

    images = []
    for path in tqdm(paths, desc="Preprocessing images"):
        with Image.open(path) as image:
            image = center_crop_arr(image.convert("RGB"), args.image_size)
            images.append(np.asarray(image, dtype=np.uint8))

    # Same NHWC uint8 input convention used by generated-sample gFID.
    image_array = np.stack(images, axis=0)
    print(f"Image array shape: {image_array.shape}")

    # This is the exact helper used by calculate_gfid for generated images.
    mu, sigma = _compute_inception_moments_from_arr(
        image_array,
        batch_size=args.batch_size,
        device=device,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, mu=mu, sigma=sigma)
    print(f"Saved FID statistics to {args.output}")
    print(f"mu shape: {mu.shape}; sigma shape: {sigma.shape}")


if __name__ == "__main__":
    main()
