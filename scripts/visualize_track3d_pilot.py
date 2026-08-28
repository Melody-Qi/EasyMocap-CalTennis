"""Create visual QA artifacts for a CalTennis EasyMocap Track3D run.

Outputs one projected-skeleton frame sequence per camera plus a top-down court
trajectory plot. The script is deliberately visualization-only: it never edits
the reconstruction JSON files.
"""

from __future__ import annotations

import argparse
import colorsys
import glob
import json
import os
from collections import defaultdict

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


BODY25_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7),
    (1, 8), (8, 9), (9, 10), (10, 11), (8, 12), (12, 13),
    (13, 14), (0, 15), (15, 17), (0, 16), (16, 18),
]


def id_color(pid: int) -> tuple[int, int, int]:
    hue = (pid * 0.61803398875) % 1.0
    rgb = colorsys.hsv_to_rgb(hue, 0.85, 1.0)
    return tuple(int(255 * channel) for channel in rgb[::-1])  # BGR


def load_cameras(root: str) -> tuple[list[str], dict[str, np.ndarray]]:
    fs = cv2.FileStorage(os.path.join(root, "intri.yml"), cv2.FILE_STORAGE_READ)
    names = [fs.getNode("names").at(i).string() for i in range(fs.getNode("names").size())]
    intrinsics = {name: fs.getNode("K_" + name).mat() for name in names}
    fs.release()

    fs = cv2.FileStorage(os.path.join(root, "extri.yml"), cv2.FILE_STORAGE_READ)
    rotations = {name: fs.getNode("R_" + name).mat().reshape(-1) for name in names}
    translations = {name: fs.getNode("T_" + name).mat().reshape(-1) for name in names}
    fs.release()

    projections = {}
    for name in names:
        rotation, _ = cv2.Rodrigues(rotations[name])
        extrinsic = np.hstack([rotation, translations[name].reshape(3, 1)])
        projections[name] = intrinsics[name] @ extrinsic
    return names, projections


def read_people(filename: str) -> list[dict]:
    with open(filename, "r", encoding="utf-8") as handle:
        value = json.load(handle)
    return value if isinstance(value, list) else value.get("annots", [])


def root_xyz(points: np.ndarray) -> np.ndarray | None:
    valid = np.isfinite(points[:, :3]).all(axis=1)
    if points.shape[1] >= 4:
        valid &= points[:, 3] > 0
    if len(points) > 8 and valid[8]:
        return points[8, :3]
    return points[valid, :3].mean(axis=0) if valid.any() else None


def project_skeleton(points: np.ndarray, projection: np.ndarray) -> list[tuple[int, int] | None]:
    projected = []
    for point in points:
        if len(point) >= 4 and point[3] <= 0:
            projected.append(None)
            continue
        homogeneous = np.r_[point[:3], 1.0]
        pixel = projection @ homogeneous
        if not np.isfinite(pixel).all() or pixel[2] <= 0:
            projected.append(None)
            continue
        projected.append((int(round(pixel[0] / pixel[2])), int(round(pixel[1] / pixel[2]))))
    return projected


def draw_camera_sequences(
    root: str,
    cameras: list[str],
    projections: dict[str, np.ndarray],
    tracked_dir: str,
    out_dir: str,
    scale: float,
) -> None:
    files = sorted(glob.glob(os.path.join(tracked_dir, "*.json")))
    frame_ids = [int(os.path.splitext(os.path.basename(path))[0]) for path in files]
    for camera in cameras:
        camera_out = os.path.join(out_dir, "frames", camera)
        os.makedirs(camera_out, exist_ok=True)
        for frame_id, keypoint_file in zip(frame_ids, files):
            image_file = os.path.join(root, "images", camera, f"{frame_id:06d}.jpg")
            image = cv2.imread(image_file)
            if image is None:
                continue
            active_ids = []
            for order, person in enumerate(read_people(keypoint_file)):
                pid = int(person.get("id", order))
                points = np.asarray(person["keypoints3d"], dtype=float)
                pixels = project_skeleton(points, projections[camera])
                color = id_color(pid)
                active_ids.append(pid)
                for start, end in BODY25_EDGES:
                    if start < len(pixels) and end < len(pixels) and pixels[start] and pixels[end]:
                        cv2.line(image, pixels[start], pixels[end], color, 4, cv2.LINE_AA)
                for pixel in pixels:
                    if pixel is not None:
                        cv2.circle(image, pixel, 5, color, -1, cv2.LINE_AA)
                root_pixel = pixels[8] if len(pixels) > 8 else next((p for p in pixels if p), None)
                if root_pixel is not None:
                    cv2.putText(
                        image, f"ID {pid}", (root_pixel[0] + 8, root_pixel[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 3, cv2.LINE_AA,
                    )
            text = f"{camera} | frame {frame_id:06d} | active track IDs: {active_ids}"
            cv2.rectangle(image, (0, 0), (image.shape[1], 58), (0, 0, 0), -1)
            cv2.putText(image, text, (18, 39), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (255, 255, 255), 2, cv2.LINE_AA)
            if scale != 1.0:
                image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            cv2.imwrite(os.path.join(camera_out, f"{frame_id:06d}.jpg"), image)
        print(f"wrote projected skeleton frames: {camera_out}")


def draw_topdown(tracked_dir: str, out_file: str, subtitle: str) -> None:
    tracks: dict[int, list[tuple[int, np.ndarray]]] = defaultdict(list)
    for frame_index, filename in enumerate(sorted(glob.glob(os.path.join(tracked_dir, "*.json")))):
        for order, person in enumerate(read_people(filename)):
            pid = int(person.get("id", order))
            points = np.asarray(person["keypoints3d"], dtype=float)
            root = root_xyz(points)
            if root is not None:
                tracks[pid].append((frame_index, root))

    fig, axis = plt.subplots(figsize=(13, 7.5), dpi=160)
    for pid, samples in sorted(tracks.items()):
        if len(samples) < 2:
            continue
        values = np.stack([sample[1] for sample in samples])
        color_bgr = id_color(pid)
        color_rgb = tuple(channel / 255.0 for channel in color_bgr[::-1])
        axis.plot(values[:, 0], values[:, 1], linewidth=1.8, alpha=0.9,
                  color=color_rgb, label=f"ID {pid} ({len(samples)} frames)")
        axis.scatter(values[0, 0], values[0, 1], marker="o", s=24, color=color_rgb)
        axis.text(values[0, 0], values[0, 1], str(pid), fontsize=7, color=color_rgb)
    axis.set_title("EasyMocap Track3D — top-down root trajectories\n" + subtitle)
    axis.set_xlabel("Calibrated court X (m)")
    axis.set_ylabel("Calibrated court Y (m)")
    axis.grid(True, alpha=0.25)
    axis.set_aspect("equal", adjustable="datalim")
    axis.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=7, ncol=1)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote top-down plot: {out_file}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="CalTennis EasyMocap data root")
    parser.add_argument("--tracked", required=True, help="directory containing per-frame keypoints3d JSON")
    parser.add_argument("--out", required=True)
    parser.add_argument("--cameras", nargs="+", default=None)
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument(
        "--topdown-subtitle",
        default="Many short IDs indicate track fragmentation",
        help="second line of the top-down plot title",
    )
    args = parser.parse_args()

    camera_names, projections = load_cameras(args.root)
    cameras = args.cameras or camera_names
    os.makedirs(args.out, exist_ok=True)
    draw_camera_sequences(args.root, cameras, projections, args.tracked, args.out, args.scale)
    draw_topdown(
        args.tracked,
        os.path.join(args.out, "track3d_topdown_trajectories.png"),
        args.topdown_subtitle,
    )


if __name__ == "__main__":
    main()
