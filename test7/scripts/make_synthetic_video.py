#!/usr/bin/env python
"""Create a small deterministic smoke video and first-frame object mask."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--frames", type=int, default=3)
    parser.add_argument("--height", type=int, default=192)
    parser.add_argument("--width", type=int, default=320)
    args = parser.parse_args()
    if args.frames < 2 or args.height < 32 or args.width < 32:
        raise ValueError("Use at least 2 frames and dimensions of at least 32 pixels")
    frames_dir = args.output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    box_height = max(16, args.height // 4)
    box_width = max(16, args.width // 5)
    y0 = (args.height - box_height) // 2
    first_mask = np.zeros((args.height, args.width), dtype=np.uint8)
    for frame_idx in range(args.frames):
        image = np.zeros((args.height, args.width, 3), dtype=np.uint8)
        image[..., 0] = np.linspace(20, 100, args.width, dtype=np.uint8)
        image[..., 1] = np.linspace(10, 70, args.height, dtype=np.uint8)[:, None]
        x0 = min(12 + frame_idx * 8, args.width - box_width)
        image[y0 : y0 + box_height, x0 : x0 + box_width] = (220, 180, 40)
        Image.fromarray(image).save(frames_dir / f"{frame_idx:05d}.jpg", quality=95)
        if frame_idx == 0:
            first_mask[y0 : y0 + box_height, x0 : x0 + box_width] = 1
    prompt = args.output_dir / "00000.png"
    Image.fromarray(first_mask).save(prompt)
    print(f"frames={frames_dir.resolve()}")
    print(f"prompt_mask={prompt.resolve()}")


if __name__ == "__main__":
    main()

