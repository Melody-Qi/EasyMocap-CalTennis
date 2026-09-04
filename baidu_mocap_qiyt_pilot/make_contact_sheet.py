#!/usr/bin/env python3
"""Create a camera-by-time contact sheet from extracted EasyMocap frames."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, help="directory containing images/<camera>")
    parser.add_argument("--cameras", nargs="+", required=True)
    parser.add_argument("--frames", nargs="+", type=int, required=True)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--cell-width", type=int, default=480)
    parser.add_argument("--cell-height", type=int, default=270)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for camera in args.cameras:
        cells = []
        for frame_index in args.frames:
            path = args.root / "images" / camera / f"{frame_index:06d}.jpg"
            image = cv2.imread(str(path))
            if image is None:
                raise FileNotFoundError(path)
            image = cv2.resize(image, (args.cell_width, args.cell_height), interpolation=cv2.INTER_AREA)
            cv2.rectangle(image, (0, 0), (args.cell_width, 38), (0, 0, 0), -1)
            label = f"{camera}  t={frame_index / args.fps:.1f}s  frame={frame_index}"
            cv2.putText(image, label, (8, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (30, 255, 255), 2)
            cells.append(image)
        rows.append(np.concatenate(cells, axis=1))
    sheet = np.concatenate(rows, axis=0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), sheet):
        raise RuntimeError(f"Cannot write {args.output}")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
