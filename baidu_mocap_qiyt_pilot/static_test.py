#!/usr/bin/env python3
"""判断三路相机是否全程静止。

做法：比较同一相机不同时刻的两帧。若相机静止，背景像素几乎不变，
差异只出现在运动员等运动物体上 —— 因此「差异的中位数」应该很小，
而「差异很大的像素占比」等于运动物体在画面中的面积占比（几个百分点）。

若相机移动过（或被机身电子防抖抖过），背景整体错位，
中位数会显著抬高。

用法: python static_test.py <dir> cam1_t05s.jpg cam1_t380s.jpg ...
"""
from __future__ import annotations

import sys
import numpy as np
from PIL import Image


def load(path: str) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32)


def compare(a: np.ndarray, b: np.ndarray) -> dict:
    d = np.abs(a - b)
    h, w = d.shape
    return {
        "median": float(np.median(d)),
        "mean": float(d.mean()),
        "p90": float(np.percentile(d, 90)),
        "frac_gt10": float((d > 10).mean()),
        "frac_gt30": float((d > 30).mean()),
        # 分块统计：若只是运动员在动，只有少数块差异大；
        # 若相机移动，所有块都会差异大。
        "block_frac_moving": float(
            np.mean(
                d.reshape(d.shape[0] // 12, 12, d.shape[1] // 12, 12)
                .mean(axis=(1, 3))
                > 10
            )
        ),
    }


def main() -> None:
    d = sys.argv[1]
    paths = sys.argv[2:]
    imgs = {p: load(f"{d}/{p}") for p in paths}
    print(f"{'pair':<34}{'median':>8}{'p90':>8}{'>10':>8}{'blocks moving':>16}")
    for i in range(0, len(paths), 2):
        a, b = paths[i], paths[i + 1]
        r = compare(imgs[a], imgs[b])
        print(
            f"{a[:-4]+' vs '+b[:-4]:<34}"
            f"{r['median']:>8.2f}{r['p90']:>8.1f}"
            f"{r['frac_gt10']*100:>7.1f}%{r['block_frac_moving']*100:>15.1f}%"
        )


if __name__ == "__main__":
    main()
