"""Evaluate whether EasyMocap Track3D yields stable two-player court tracks.

The script intentionally reports continuity rather than claiming ground-truth
accuracy.  It treats BODY25 pelvis joint 8 as the player root, falling back to
the centroid of valid joints when pelvis confidence is unavailable.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import Counter, defaultdict

import numpy as np


def read_people(path: str) -> list[dict]:
    data = json.load(open(path, "r", encoding="utf-8"))
    return data if isinstance(data, list) else data.get("annots", [])


def root_xyz(person: dict) -> np.ndarray | None:
    points = np.asarray(person.get("keypoints3d", []), dtype=float)
    if points.ndim != 2 or points.shape[0] == 0 or points.shape[1] < 3:
        return None
    valid = np.isfinite(points[:, :3]).all(axis=1)
    if points.shape[1] >= 4:
        valid &= points[:, 3] > 0
    if points.shape[0] > 8 and valid[8]:
        return points[8, :3]
    if valid.any():
        return points[valid, :3].mean(axis=0)
    return None


def summarize(folder: str) -> dict:
    files = sorted(glob.glob(os.path.join(folder, "*.json")))
    people_per_frame = Counter()
    roots: dict[str, list[tuple[int, np.ndarray]]] = defaultdict(list)
    for frame_index, filename in enumerate(files):
        people = read_people(filename)
        people_per_frame[len(people)] += 1
        for order, person in enumerate(people):
            pid = str(person.get("id", person.get("personID", order)))
            xyz = root_xyz(person)
            if xyz is not None:
                roots[pid].append((frame_index, xyz))

    track_stats = {}
    all_step_distances = []
    for pid, samples in roots.items():
        consecutive = []
        gaps = []
        for (f0, p0), (f1, p1) in zip(samples, samples[1:]):
            gaps.append(f1 - f0)
            if f1 == f0 + 1:
                consecutive.append(float(np.linalg.norm(p1 - p0)))
        all_step_distances.extend(consecutive)
        track_stats[pid] = {
            "frames_present": len(samples),
            "first_frame": samples[0][0] if samples else None,
            "last_frame": samples[-1][0] if samples else None,
            "max_gap_frames": max(gaps, default=0),
            "step_median_m": float(np.median(consecutive)) if consecutive else None,
            "step_p95_m": float(np.percentile(consecutive, 95)) if consecutive else None,
            "step_max_m": max(consecutive, default=None),
            "steps_gt_0_5m": sum(d > 0.5 for d in consecutive),
            "steps_gt_1m": sum(d > 1.0 for d in consecutive),
        }

    n_frames = len(files)
    return {
        "folder": folder,
        "frames": n_frames,
        "people_per_frame": dict(sorted(people_per_frame.items())),
        "two_person_frame_fraction": people_per_frame[2] / n_frames if n_frames else 0.0,
        "track_count": len(track_stats),
        "tracks": track_stats,
        "all_tracks_step_median_m": (
            float(np.median(all_step_distances)) if all_step_distances else None
        ),
        "all_tracks_step_max_m": max(all_step_distances, default=None),
        "all_tracks_steps_gt_0_5m": sum(d > 0.5 for d in all_step_distances),
        "all_tracks_steps_gt_1m": sum(d > 1.0 for d in all_step_distances),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True)
    parser.add_argument("--tracked", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    report = {"raw": summarize(args.raw), "tracked": summarize(args.tracked)}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
