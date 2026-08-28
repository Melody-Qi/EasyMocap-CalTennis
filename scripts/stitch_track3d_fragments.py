"""Merge fragmented EasyMocap Track3D tracks into four court-global IDs.

This is a CalTennis 2v2 pilot post-process.  It exploits the fact that the four
players occupy four well-separated regions in calibrated court XY coordinates.
It is not intended as a general multi-camera person re-identification model.
The input JSON files are read-only; all rewritten frames and diagnostics go to
a separate output directory.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import Counter, defaultdict
from copy import deepcopy

import numpy as np


def read_people(filename: str) -> list[dict]:
    with open(filename, "r", encoding="utf-8") as handle:
        value = json.load(handle)
    return value if isinstance(value, list) else value.get("annots", [])


def root_xyz(person: dict) -> np.ndarray | None:
    points = np.asarray(person.get("keypoints3d", []), dtype=float)
    if points.ndim != 2 or points.shape[0] == 0 or points.shape[1] < 3:
        return None
    valid = np.isfinite(points[:, :3]).all(axis=1)
    if points.shape[1] >= 4:
        valid &= points[:, 3] > 0
    if len(points) > 8 and valid[8]:
        return points[8, :3]
    return points[valid, :3].mean(axis=0) if valid.any() else None


def weighted_kmeans_1d(values: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split one court end into its two players using robust weighted Y centers."""
    centers = np.array([np.percentile(values, 20), np.percentile(values, 80)], dtype=float)
    for _ in range(100):
        labels = np.argmin(np.abs(values[:, None] - centers[None]), axis=1)
        updated = centers.copy()
        for cluster in range(2):
            mask = labels == cluster
            if mask.any():
                updated[cluster] = np.average(values[mask], weights=weights[mask])
        if np.allclose(updated, centers, atol=1e-6):
            break
        centers = updated
    order = np.argsort(centers)
    remap = np.zeros(2, dtype=int)
    remap[order] = np.arange(2)
    return centers[order], remap[labels]


def hierarchical_court_centers(
    roots_by_track: dict[int, list[np.ndarray]],
) -> tuple[np.ndarray, dict[int, int]]:
    """Use the largest court-X gap, then split each end into two Y regions.

    Unlike frame-level four-means, every original tracklet contributes one
    median weighted by its duration. This prevents a short bad triangulation
    from claiming its own cluster center.
    """
    track_ids = sorted(roots_by_track)
    medians = np.stack([np.median(np.stack(roots_by_track[tid])[:, :2], axis=0) for tid in track_ids])
    weights = np.asarray([len(roots_by_track[tid]) for tid in track_ids], dtype=float)

    unique_x = np.sort(np.unique(medians[:, 0]))
    if len(unique_x) < 2:
        raise RuntimeError("cannot split players by court X")
    split_index = int(np.argmax(np.diff(unique_x)))
    x_threshold = float((unique_x[split_index] + unique_x[split_index + 1]) * 0.5)
    side_labels = (medians[:, 0] > x_threshold).astype(int)

    centers = []
    initial_mapping = {}
    for side in (0, 1):
        mask = side_labels == side
        if mask.sum() < 2:
            raise RuntimeError(f"court side {side} has fewer than two tracklets")
        _, y_labels = weighted_kmeans_1d(medians[mask, 1], weights[mask])
        side_indices = np.where(mask)[0]
        for local_id in (0, 1):
            local_indices = side_indices[y_labels == local_id]
            center = np.average(medians[local_indices], axis=0, weights=weights[local_indices])
            global_id = side * 2 + local_id
            centers.append(center)
            for index in local_indices:
                initial_mapping[track_ids[index]] = global_id
    return np.stack(centers), initial_mapping


def interpolate_person(left: dict, right: dict, alpha: float, global_id: int) -> dict:
    left_points = np.asarray(left["keypoints3d"], dtype=float)
    right_points = np.asarray(right["keypoints3d"], dtype=float)
    points = left_points * (1.0 - alpha) + right_points * alpha
    # Confidence should never become larger merely because a frame was synthesized.
    if points.shape[1] >= 4:
        points[:, 3] = np.minimum(left_points[:, 3], right_points[:, 3])
    return {
        "id": int(global_id),
        "type": left.get("type", right.get("type", "body25")),
        "keypoints3d": points.tolist(),
        "source_track_id": None,
        "interpolated": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Track3D keypoints3d directory")
    parser.add_argument("--output", required=True, help="new output root")
    parser.add_argument("--clusters", type=int, default=4)
    parser.add_argument("--max-interp-gap", type=int, default=15)
    parser.add_argument("--max-track-center-distance", type=float, default=2.5)
    parser.add_argument("--court-x", nargs=2, type=float, default=[-5.0, 30.0])
    parser.add_argument("--court-y", nargs=2, type=float, default=[-5.0, 20.0])
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.input, "*.json")))
    if not files:
        raise FileNotFoundError(f"no keypoints3d JSON files in {args.input}")

    frames: list[list[dict]] = []
    roots_by_track: dict[int, list[np.ndarray]] = defaultdict(list)
    all_roots = []
    for filename in files:
        people = read_people(filename)
        frames.append(people)
        for order, person in enumerate(people):
            track_id = int(person.get("id", order))
            root = root_xyz(person)
            if root is None:
                continue
            if not (args.court_x[0] <= root[0] <= args.court_x[1]):
                continue
            if not (args.court_y[0] <= root[1] <= args.court_y[1]):
                continue
            roots_by_track[track_id].append(root)
            all_roots.append(root[:2])

    if len(all_roots) < args.clusters:
        raise RuntimeError("not enough valid roots to form global tracks")
    if args.clusters != 4:
        raise ValueError("this pilot stitcher currently supports exactly four court-region identities")
    centers, initial_mapping = hierarchical_court_centers(roots_by_track)

    track_to_global = {}
    track_medians = {}
    rejected_tracks = {}
    for track_id, roots in roots_by_track.items():
        median = np.median(np.stack(roots)[:, :2], axis=0)
        global_id = int(initial_mapping[track_id])
        distance = float(np.linalg.norm(centers[global_id] - median))
        if distance > args.max_track_center_distance:
            rejected_tracks[track_id] = {
                "candidate_global_id": global_id,
                "distance_m": distance,
                "median_xy": median.tolist(),
                "frames": len(roots),
            }
            continue
        track_to_global[track_id] = global_id
        track_medians[track_id] = median

    selected: list[dict[int, dict]] = [dict() for _ in frames]
    conflict_count = 0
    dropped_candidates = 0
    out_of_court = 0
    for frame_index, people in enumerate(frames):
        candidates: dict[int, list[tuple[float, int, dict]]] = defaultdict(list)
        for order, person in enumerate(people):
            track_id = int(person.get("id", order))
            root = root_xyz(person)
            if root is None or track_id not in track_to_global:
                out_of_court += 1
                continue
            global_id = track_to_global[track_id]
            distance = float(np.linalg.norm(root[:2] - centers[global_id]))
            candidates[global_id].append((distance, track_id, person))
        for global_id, options in candidates.items():
            options.sort(key=lambda item: item[0])
            if len(options) > 1:
                conflict_count += 1
                dropped_candidates += len(options) - 1
            _, track_id, person = options[0]
            rewritten = deepcopy(person)
            rewritten["id"] = int(global_id)
            rewritten["source_track_id"] = int(track_id)
            rewritten["interpolated"] = False
            selected[frame_index][global_id] = rewritten

    interpolated_frames = Counter()
    for global_id in range(args.clusters):
        known = [index for index, values in enumerate(selected) if global_id in values]
        for left_index, right_index in zip(known, known[1:]):
            gap = right_index - left_index - 1
            if gap <= 0 or gap > args.max_interp_gap:
                continue
            left = selected[left_index][global_id]
            right = selected[right_index][global_id]
            for frame_index in range(left_index + 1, right_index):
                if global_id in selected[frame_index]:
                    continue
                alpha = (frame_index - left_index) / (right_index - left_index)
                selected[frame_index][global_id] = interpolate_person(left, right, alpha, global_id)
                interpolated_frames[global_id] += 1

    keypoints_out = os.path.join(args.output, "keypoints3d")
    os.makedirs(keypoints_out, exist_ok=True)
    for filename, frame_values in zip(files, selected):
        basename = os.path.basename(filename)
        people = [frame_values[global_id] for global_id in sorted(frame_values)]
        with open(os.path.join(keypoints_out, basename), "w", encoding="utf-8") as handle:
            json.dump(people, handle, indent=2)

    frame_count_histogram = Counter(len(values) for values in selected)
    global_frames = {
        str(global_id): sum(global_id in values for values in selected)
        for global_id in range(args.clusters)
    }
    mapping = {
        str(track_id): {
            "global_id": int(track_to_global[track_id]),
            "frames": len(roots_by_track[track_id]),
            "median_xy": track_medians[track_id].tolist(),
        }
        for track_id in sorted(track_to_global)
    }
    report = {
        "method": "court-XY four-cluster Track3D tracklet stitching",
        "warning": "CalTennis 2v2 pilot heuristic; not general person ReID",
        "input": args.input,
        "output": args.output,
        "frames": len(frames),
        "global_centers_xy": centers.tolist(),
        "original_track_count": len(track_to_global),
        "rejected_track_count": len(rejected_tracks),
        "rejected_tracks": {str(key): value for key, value in sorted(rejected_tracks.items())},
        "global_track_count": args.clusters,
        "original_to_global": mapping,
        "people_per_frame_after": dict(sorted(frame_count_histogram.items())),
        "frames_present_per_global_id": global_frames,
        "conflict_frames": conflict_count,
        "dropped_duplicate_candidates": dropped_candidates,
        "ignored_out_of_court_or_invalid": out_of_court,
        "interpolated_frames_per_global_id": dict(interpolated_frames),
        "max_interpolation_gap": args.max_interp_gap,
    }
    with open(os.path.join(args.output, "stitch_report.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
