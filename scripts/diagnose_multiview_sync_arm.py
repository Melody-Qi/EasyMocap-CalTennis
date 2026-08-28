"""Diagnose residual multi-view sync and fast-arm 2D keypoint quality.

The script uses calibrated epipolar geometry, so it does not need stable
cross-camera person IDs.  For each camera pair it searches a small integer
frame offset and matches detected people with the Hungarian algorithm.  It
also reports BODY25 shoulder/elbow/wrist confidence and temporal jumps.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path

import cv2
import matplotlib
import numpy as np
from scipy.optimize import linear_sum_assignment

matplotlib.use("Agg")
import matplotlib.pyplot as plt


BODY = np.arange(19)
ARMS = np.asarray([2, 3, 4, 5, 6, 7])
ELBOWS = np.asarray([3, 6])
WRISTS = np.asarray([4, 7])
JOINT_NAMES = {2: "RShoulder", 3: "RElbow", 4: "RWrist", 5: "LShoulder", 6: "LElbow", 7: "LWrist"}


def load_opencv_matrix(path: Path, key: str) -> np.ndarray:
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    value = fs.getNode(key).mat()
    fs.release()
    if value is None:
        raise KeyError(f"missing {key} in {path}")
    return np.asarray(value, dtype=np.float64)


def load_camera(root: Path, name: str) -> dict[str, np.ndarray]:
    camera = {
        "K": load_opencv_matrix(root / "intri.yml", f"K_{name}"),
        "R": load_opencv_matrix(root / "extri.yml", f"Rot_{name}"),
        "T": load_opencv_matrix(root / "extri.yml", f"T_{name}").reshape(3, 1),
    }
    try:
        camera["dist"] = load_opencv_matrix(root / "intri.yml", f"dist_{name}").reshape(-1)
    except KeyError:
        camera["dist"] = np.zeros(5, dtype=np.float64)
    return camera


def skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = vector.reshape(3)
    return np.asarray([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=np.float64)


def fundamental(first: dict[str, np.ndarray], second: dict[str, np.ndarray]) -> np.ndarray:
    relative_rotation = second["R"] @ first["R"].T
    relative_translation = second["T"] - relative_rotation @ first["T"]
    essential = skew(relative_translation) @ relative_rotation
    return np.linalg.inv(second["K"]).T @ essential @ np.linalg.inv(first["K"])


def undistort(keypoints: np.ndarray, camera: dict[str, np.ndarray]) -> np.ndarray:
    result = keypoints.copy()
    valid = result[:, 2] > 0
    if np.any(valid):
        points = result[valid, :2].reshape(-1, 1, 2).astype(np.float64)
        result[valid, :2] = cv2.undistortPoints(
            points, camera["K"], camera["dist"], P=camera["K"]
        ).reshape(-1, 2)
    return result


def read_annotations(folder: Path, camera: dict[str, np.ndarray]) -> list[list[dict]]:
    files = sorted(folder.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"no annotations in {folder}")
    output: list[list[dict]] = []
    for filename in files:
        source = json.loads(filename.read_text(encoding="utf-8"))
        people = []
        for index, person in enumerate(source.get("annots", source if isinstance(source, list) else [])):
            points = np.asarray(person.get("keypoints", person.get("keypoints2d")), dtype=np.float64)
            if points.ndim != 2 or points.shape[1] < 3:
                continue
            bbox = np.asarray(person.get("bbox", [0, 0, 0, 0, 0]), dtype=np.float64)
            people.append({
                "id": int(person.get("personID", person.get("id", index))),
                "points": undistort(points[:, :3], camera),
                "bbox": bbox,
            })
        output.append(people)
    return output


def symmetric_epipolar(points1: np.ndarray, points2: np.ndarray, matrix: np.ndarray, joints: np.ndarray,
                       confidence: float, minimum: int) -> float:
    valid = (points1[joints, 2] >= confidence) & (points2[joints, 2] >= confidence)
    selected = joints[valid]
    if len(selected) < minimum:
        return float("nan")
    x1 = np.column_stack([points1[selected, :2], np.ones(len(selected))])
    x2 = np.column_stack([points2[selected, :2], np.ones(len(selected))])
    lines2 = (matrix @ x1.T).T
    lines1 = (matrix.T @ x2.T).T
    numerator = np.abs(np.sum(x2 * lines2, axis=1))
    distance2 = numerator / np.maximum(np.linalg.norm(lines2[:, :2], axis=1), 1e-9)
    distance1 = numerator / np.maximum(np.linalg.norm(lines1[:, :2], axis=1), 1e-9)
    return float(np.median(0.5 * (distance1 + distance2)))


def camera_motion_quality(frames: list[list[dict]], confidence: float) -> tuple[dict, list[set[int]]]:
    confidence_by_joint = {int(joint): [] for joint in ARMS}
    speeds = []
    speed_records = []
    previous = {}
    fast_by_frame = [set() for _ in frames]
    for frame_index, people in enumerate(frames):
        current = {}
        for person in people:
            pid, points = person["id"], person["points"]
            height = max(float(person["bbox"][3] - person["bbox"][1]), 1.0)
            for joint in ARMS:
                confidence_by_joint[int(joint)].append(float(points[joint, 2]))
            current[pid] = (points, height)
            if pid not in previous:
                continue
            old_points, old_height = previous[pid]
            valid = (points[ARMS, 2] >= confidence) & (old_points[ARMS, 2] >= confidence)
            if not np.any(valid):
                continue
            displacement = np.linalg.norm(points[ARMS[valid], :2] - old_points[ARMS[valid], :2], axis=1)
            normalized = float(np.max(displacement) / max(0.5 * (height + old_height), 1.0))
            speeds.append(normalized)
            speed_records.append((frame_index, pid, normalized))
        previous = current
    threshold = float(np.percentile(speeds, 80)) if speeds else float("nan")
    for frame_index, pid, speed in speed_records:
        if speed >= threshold:
            fast_by_frame[frame_index].add(pid)
    joint_summary = {}
    for joint, values in confidence_by_joint.items():
        array = np.asarray(values, dtype=np.float64)
        joint_summary[JOINT_NAMES[joint]] = {
            "samples": int(len(array)),
            "median_confidence": float(np.median(array)) if len(array) else None,
            "below_0.3_fraction": float(np.mean(array < 0.3)) if len(array) else None,
            "below_0.5_fraction": float(np.mean(array < 0.5)) if len(array) else None,
        }
    speed_array = np.asarray(speeds, dtype=np.float64)
    return {
        "frames": len(frames),
        "detections": int(sum(len(frame) for frame in frames)),
        "arm_joint_confidence": joint_summary,
        "arm_motion_bbox_per_frame_p50": float(np.percentile(speed_array, 50)) if len(speed_array) else None,
        "arm_motion_bbox_per_frame_p80_fast_threshold": threshold if len(speed_array) else None,
        "arm_motion_bbox_per_frame_p95": float(np.percentile(speed_array, 95)) if len(speed_array) else None,
        "extreme_jump_over_half_bbox_fraction": float(np.mean(speed_array > 0.5)) if len(speed_array) else None,
    }, fast_by_frame


def pair_offset_curve(first_frames, second_frames, fast_first, fast_second, matrix, max_offset, confidence):
    curve = []
    for offset in range(-max_offset, max_offset + 1):
        body_values, arm_values, fast_values = [], [], []
        matched_frames = 0
        start = max(0, -offset)
        stop = min(len(first_frames), len(second_frames) - offset)
        for first_index in range(start, stop):
            second_index = first_index + offset
            people1, people2 = first_frames[first_index], second_frames[second_index]
            if not people1 or not people2:
                continue
            body_cost = np.full((len(people1), len(people2)), 1e6, dtype=np.float64)
            arm_cost = np.full_like(body_cost, np.nan)
            for i, person1 in enumerate(people1):
                for j, person2 in enumerate(people2):
                    body_cost[i, j] = symmetric_epipolar(
                        person1["points"], person2["points"], matrix, BODY, confidence, 6
                    )
                    arm_cost[i, j] = symmetric_epipolar(
                        person1["points"], person2["points"], matrix, ARMS, confidence, 3
                    )
            body_cost[~np.isfinite(body_cost)] = 1e6
            rows, columns = linear_sum_assignment(body_cost)
            used = False
            for row, column in zip(rows, columns):
                value = float(body_cost[row, column])
                # Keep a permissive gate here. A large value is itself useful
                # evidence of calibration/person-association trouble; a tight
                # gate could otherwise make the whole offset curve empty.
                if value > 500.0:
                    continue
                body_values.append(value)
                arm = float(arm_cost[row, column])
                if np.isfinite(arm):
                    arm_values.append(arm)
                    if (people1[row]["id"] in fast_first[first_index] or
                            people2[column]["id"] in fast_second[second_index]):
                        fast_values.append(arm)
                used = True
            matched_frames += int(used)
        curve.append({
            "offset_second_minus_first_frames": offset,
            "matched_frames": matched_frames,
            "matched_people": len(body_values),
            "median_body_epipolar_px": float(np.median(body_values)) if body_values else None,
            "median_arm_epipolar_px": float(np.median(arm_values)) if arm_values else None,
            "p95_arm_epipolar_px": float(np.percentile(arm_values, 95)) if arm_values else None,
            "fast_arm_samples": len(fast_values),
            "median_fast_arm_epipolar_px": float(np.median(fast_values)) if fast_values else None,
        })
    return curve


def choose_best(curve: list[dict]) -> dict:
    valid_fast = [row for row in curve if row["fast_arm_samples"] >= 50]
    if valid_fast:
        return min(valid_fast, key=lambda row: row["median_fast_arm_epipolar_px"])
    valid_arm = [row for row in curve if row["median_arm_epipolar_px"] is not None]
    if valid_arm:
        return min(valid_arm, key=lambda row: row["median_arm_epipolar_px"])
    valid_body = [row for row in curve if row["median_body_epipolar_px"] is not None]
    if valid_body:
        return min(valid_body, key=lambda row: row["median_body_epipolar_px"])
    raise RuntimeError("no valid cross-view person/keypoint matches; check annotation format and calibration")


def joint_detail_at_offset(first_frames, second_frames, fast_first, fast_second, matrix,
                           offset, confidence):
    values = {int(joint): [] for joint in ARMS}
    fast_values = {int(joint): [] for joint in ARMS}
    start = max(0, -offset)
    stop = min(len(first_frames), len(second_frames) - offset)
    for first_index in range(start, stop):
        second_index = first_index + offset
        people1, people2 = first_frames[first_index], second_frames[second_index]
        if not people1 or not people2:
            continue
        costs = np.full((len(people1), len(people2)), 1e6, dtype=np.float64)
        for i, person1 in enumerate(people1):
            for j, person2 in enumerate(people2):
                costs[i, j] = symmetric_epipolar(
                    person1["points"], person2["points"], matrix, BODY, confidence, 6
                )
        costs[~np.isfinite(costs)] = 1e6
        rows, columns = linear_sum_assignment(costs)
        for row, column in zip(rows, columns):
            if costs[row, column] > 500:
                continue
            is_fast = (people1[row]["id"] in fast_first[first_index] or
                       people2[column]["id"] in fast_second[second_index])
            for joint in ARMS:
                error = symmetric_epipolar(
                    people1[row]["points"], people2[column]["points"], matrix,
                    np.asarray([joint]), confidence, 1,
                )
                if np.isfinite(error):
                    values[int(joint)].append(error)
                    if is_fast:
                        fast_values[int(joint)].append(error)
    detail = {}
    for joint in ARMS:
        all_array = np.asarray(values[int(joint)], dtype=np.float64)
        fast_array = np.asarray(fast_values[int(joint)], dtype=np.float64)
        detail[JOINT_NAMES[int(joint)]] = {
            "samples": int(len(all_array)),
            "median_epipolar_px": float(np.median(all_array)) if len(all_array) else None,
            "fast_samples": int(len(fast_array)),
            "median_fast_epipolar_px": float(np.median(fast_array)) if len(fast_array) else None,
            "p95_fast_epipolar_px": float(np.percentile(fast_array, 95)) if len(fast_array) else None,
        }
    return detail


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--annots", default="annots")
    parser.add_argument("--cameras", nargs="+", default=["W_01", "W_02", "E_03", "E_04"])
    parser.add_argument("--max-offset", type=int, default=8)
    parser.add_argument("--confidence", type=float, default=0.3)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    cameras = {name: load_camera(args.data, name) for name in args.cameras}
    frames, quality, fast = {}, {}, {}
    for name in args.cameras:
        frames[name] = read_annotations(args.data / args.annots / name, cameras[name])
        quality[name], fast[name] = camera_motion_quality(frames[name], args.confidence)

    pairs = {}
    for first, second in itertools.combinations(args.cameras, 2):
        curve = pair_offset_curve(
            frames[first], frames[second], fast[first], fast[second],
            fundamental(cameras[first], cameras[second]), args.max_offset, args.confidence,
        )
        best = choose_best(curve)
        zero = next(row for row in curve if row["offset_second_minus_first_frames"] == 0)
        detail = joint_detail_at_offset(
            frames[first], frames[second], fast[first], fast[second],
            fundamental(cameras[first], cameras[second]),
            best["offset_second_minus_first_frames"], args.confidence,
        )
        pairs[f"{first}__{second}"] = {
            "best": best, "zero": zero, "joint_detail_at_best": detail, "curve": curve
        }

    report = {
        "offset_definition": "compare first camera frame t with second camera frame t+offset",
        "camera_quality": quality,
        "pairs": pairs,
    }
    (args.output / "multiview_sync_arm_diagnostics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for axis, (name, values) in zip(axes.flat, pairs.items()):
        offsets = [row["offset_second_minus_first_frames"] for row in values["curve"]]
        arm = [row["median_arm_epipolar_px"] for row in values["curve"]]
        fast_arm = [row["median_fast_arm_epipolar_px"] for row in values["curve"]]
        axis.plot(offsets, arm, marker="o", label="all arm")
        axis.plot(offsets, fast_arm, marker="o", label="fast arm")
        axis.axvline(0, color="black", linewidth=1, linestyle="--")
        axis.axvline(values["best"]["offset_second_minus_first_frames"], color="red", linewidth=1)
        axis.set_title(name)
        axis.set_xlabel("second minus first (frames)")
        axis.set_ylabel("median epipolar error (px)")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    fig.savefig(args.output / "pairwise_offset_curves.png", dpi=180)
    plt.close(fig)

    lines = [
        "# 多机位同步与快速手臂 2D 关键点诊断", "",
        "偏移定义：比较第一台相机第 `t` 帧与第二台相机第 `t + offset` 帧。30 fps 下 1 帧约 33.3 ms。", "",
        "## 帧级同步搜索", "",
        "| 相机对 | 最佳偏移/帧 | 约合毫秒 | offset=0 快速手臂误差/px | 最佳快速手臂误差/px | 样本数 |", "|---|---:|---:|---:|---:|---:|",
    ]
    for name, values in pairs.items():
        best, zero = values["best"], values["zero"]
        lines.append(
            f"| {name} | {best['offset_second_minus_first_frames']} | "
            f"{best['offset_second_minus_first_frames'] * 1000 / 30:.1f} | "
            f"{zero['median_fast_arm_epipolar_px']:.2f} | {best['median_fast_arm_epipolar_px']:.2f} | {best['fast_arm_samples']} |"
        )
    lines += ["", "## 各机位手臂关键点质量", "",
              "| 机位 | 检测人数帧 | 腕点中位置信度 | 腕点低于0.3 | 极端跳点比例 |", "|---|---:|---:|---:|---:|"]
    for name, values in quality.items():
        wrists = [values["arm_joint_confidence"][JOINT_NAMES[int(j)]] for j in WRISTS]
        median_conf = np.median([item["median_confidence"] for item in wrists])
        low = np.mean([item["below_0.3_fraction"] for item in wrists])
        lines.append(
            f"| {name} | {values['detections']} | {median_conf:.3f} | {low:.1%} | "
            f"{values['extreme_jump_over_half_bbox_fraction']:.2%} |"
        )
    lines += ["", "## 最佳偏移下的快速手臂逐关节误差", "",
              "只列出最佳整体手臂误差低于 30 px、确实有共同观测价值的相机对。", "",
              "| 相机对 | 关节 | 快速样本 | 中位误差/px | P95/px |", "|---|---|---:|---:|---:|"]
    for pair_name, pair_values in pairs.items():
        if pair_values["best"]["median_arm_epipolar_px"] is None or pair_values["best"]["median_arm_epipolar_px"] >= 30:
            continue
        for joint_name, joint_values in pair_values["joint_detail_at_best"].items():
            median = joint_values["median_fast_epipolar_px"]
            p95 = joint_values["p95_fast_epipolar_px"]
            lines.append(
                f"| {pair_name} | {joint_name} | {joint_values['fast_samples']} | "
                f"{median:.2f} | {p95:.2f} |"
            )
    lines += ["", "## 如何解释", "",
              "- 若多个相机对都在同一非零偏移处显著更低，说明仍有整机位的残余时间偏移。",
              "- 若最佳偏移接近 0，但快速手臂误差仍很大，则主要问题是挥拍时的 2D 肘/腕关键点，而非 SMPL 时序平滑。",
              "- 若不同相机对给出的偏移互相矛盾，应优先检查标定、人物误匹配或滚动快门，不能直接按某个偏移裁视频。"]
    (args.output / "多机位同步与手臂关键点诊断.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({name: values["best"] for name, values in pairs.items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
