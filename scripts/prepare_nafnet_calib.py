import argparse
import random
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(image_dir):
    paths = [p for p in Path(image_dir).rglob("*") if p.suffix.lower() in IMAGE_EXTS]
    if not paths:
        raise FileNotFoundError(f"No images found under {image_dir}")
    return sorted(paths)


def crop_or_resize(rgb, size, rng):
    h, w = rgb.shape[:2]
    if h < size or w < size:
        return cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)

    y = rng.randint(0, h - size)
    x = rng.randint(0, w - size)
    return rgb[y : y + size, x : x + size]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-dir", required=True)
    parser.add_argument("--dst-dir", default="rdk_x5/calibration_rgb_chw_256_f32")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=10)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    images = list_images(args.src_dir)
    dst_dir = Path(args.dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    for idx in range(args.count):
        image_path = images[idx % len(images)]
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"Failed to read image: {image_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        patch = crop_or_resize(rgb, args.size, rng)
        chw = patch.transpose(2, 0, 1).astype(np.float32)
        chw.tofile(dst_dir / f"{idx:04d}.rgbchw")

    print(f"Wrote {args.count} calibration files to {dst_dir}")


if __name__ == "__main__":
    main()
