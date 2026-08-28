"""Compare temporally smoothed and frame-independent SMPL fits.

The comparison is evaluated against the non-interpolated Body25 3D skeletons.
It reports same-frame fitting error, the best integer frame offset in [-5, 5],
fast-arm-motion error, and second-difference jitter.
"""

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

from easymocap.mytools.reader import read_smpl
from easymocap.smplmodel import load_model


ARM_JOINTS = np.asarray([2, 3, 4, 5, 6, 7], dtype=np.int64)


def load_skeletons(folder):
    tracks = defaultdict(dict)
    for filename in sorted(glob.glob(os.path.join(folder, "*.json"))):
        frame = int(os.path.splitext(os.path.basename(filename))[0])
        with open(filename, "r", encoding="utf-8") as handle:
            people = json.load(handle)
        for person in people:
            keypoints = np.asarray(person["keypoints3d"], dtype=np.float32)
            tracks[int(person["id"])][frame] = keypoints
    return tracks


def load_params(folder):
    tracks = defaultdict(dict)
    for filename in sorted(glob.glob(os.path.join(folder, "*.json"))):
        frame = int(os.path.splitext(os.path.basename(filename))[0])
        for person in read_smpl(filename):
            pid = int(person.pop("id"))
            tracks[pid][frame] = person
    return tracks


def infer_joints(body_model, params_by_frame, batch_size=128):
    frames = sorted(params_by_frame)
    outputs = {}
    for start in range(0, len(frames), batch_size):
        batch_frames = frames[start : start + batch_size]
        keys = ["Rh", "Th", "poses", "shapes"]
        batch = {
            key: np.concatenate([params_by_frame[frame][key] for frame in batch_frames], axis=0)
            for key in keys
        }
        joints = body_model(return_verts=False, return_tensor=False, **batch)
        for frame, value in zip(batch_frames, joints):
            outputs[frame] = np.asarray(value, dtype=np.float32)
    return outputs


def valid_distance(pred, target, joints=None):
    if joints is None:
        joints = np.arange(min(len(pred), len(target)))
    conf = target[joints, 3] > 0
    if not np.any(conf):
        return []
    diff = pred[joints[conf], :3] - target[joints[conf], :3]
    return np.linalg.norm(diff, axis=-1).tolist()


def summarize_method(skeletons, predictions):
    same_all, same_arm = [], []
    lag_errors = {lag: [] for lag in range(-5, 6)}
    fast_arm_errors = []
    jitter_pred, jitter_skel = [], []

    for pid in sorted(set(skeletons) & set(predictions)):
        skel = skeletons[pid]
        pred = predictions[pid]
        common = sorted(set(skel) & set(pred))
        for frame in common:
            same_all.extend(valid_distance(pred[frame], skel[frame]))
            same_arm.extend(valid_distance(pred[frame], skel[frame], ARM_JOINTS))
            for lag in lag_errors:
                target_frame = frame + lag
                if target_frame in skel:
                    lag_errors[lag].extend(
                        valid_distance(pred[frame], skel[target_frame], ARM_JOINTS)
                    )

        # Fast frames are the top 20% of observed arm-joint frame-to-frame speed.
        speed_by_frame = {}
        for previous, current in zip(common[:-1], common[1:]):
            if current != previous + 1:
                continue
            conf = (skel[previous][ARM_JOINTS, 3] > 0) & (
                skel[current][ARM_JOINTS, 3] > 0
            )
            if np.any(conf):
                delta = skel[current][ARM_JOINTS[conf], :3] - skel[previous][ARM_JOINTS[conf], :3]
                speed_by_frame[current] = float(np.max(np.linalg.norm(delta, axis=-1)))
        if speed_by_frame:
            threshold = float(np.percentile(list(speed_by_frame.values()), 80))
            for frame, speed in speed_by_frame.items():
                if speed >= threshold:
                    fast_arm_errors.extend(valid_distance(pred[frame], skel[frame], ARM_JOINTS))

        # Compare second differences on consecutive triples; high values indicate jitter.
        for f0, f1, f2 in zip(common[:-2], common[1:-1], common[2:]):
            if not (f1 == f0 + 1 and f2 == f1 + 1):
                continue
            conf = (
                (skel[f0][ARM_JOINTS, 3] > 0)
                & (skel[f1][ARM_JOINTS, 3] > 0)
                & (skel[f2][ARM_JOINTS, 3] > 0)
            )
            if not np.any(conf):
                continue
            joints = ARM_JOINTS[conf]
            pred_acc = pred[f2][joints, :3] - 2 * pred[f1][joints, :3] + pred[f0][joints, :3]
            skel_acc = skel[f2][joints, :3] - 2 * skel[f1][joints, :3] + skel[f0][joints, :3]
            jitter_pred.extend(np.linalg.norm(pred_acc, axis=-1).tolist())
            jitter_skel.extend(np.linalg.norm(skel_acc, axis=-1).tolist())

    lag_medians = {
        str(lag): float(np.median(values)) if values else None
        for lag, values in lag_errors.items()
    }
    valid_lags = {int(k): v for k, v in lag_medians.items() if v is not None}
    best_lag = min(valid_lags, key=valid_lags.get) if valid_lags else None

    def stats(values):
        return {
            "count": len(values),
            "median_m": float(np.median(values)) if values else None,
            "p95_m": float(np.percentile(values, 95)) if values else None,
        }

    return {
        "same_frame_all_joints": stats(same_all),
        "same_frame_arm_joints": stats(same_arm),
        "fast_top20pct_arm_joints": stats(fast_arm_errors),
        "arm_error_by_target_frame_offset_median_m": lag_medians,
        "best_target_frame_offset": best_lag,
        "best_offset_interpretation": (
            "positive means the fitted pose resembles a later input frame; negative means it resembles an earlier input frame"
        ),
        "predicted_arm_second_difference": stats(jitter_pred),
        "input_arm_second_difference": stats(jitter_skel),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skeletons", required=True)
    parser.add_argument("--default")
    parser.add_argument("--singleframe")
    parser.add_argument(
        "--method",
        action="append",
        default=[],
        help="Additional or replacement method in NAME=PARAMETER_FOLDER form.",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    skeletons = load_skeletons(args.skeletons)
    method_paths = {}
    if args.default:
        method_paths["default_temporal"] = args.default
    if args.singleframe:
        method_paths["singleframe_zero_temporal"] = args.singleframe
    for item in args.method:
        if "=" not in item:
            parser.error("--method must use NAME=PARAMETER_FOLDER")
        name, path = item.split("=", 1)
        method_paths[name] = path
    if not method_paths:
        parser.error("provide --default/--singleframe or at least one --method")
    params = {name: load_params(path) for name, path in method_paths.items()}
    model = load_model("female", model_type="smpl")
    predictions = {
        method: {pid: infer_joints(model, track) for pid, track in tracks.items()}
        for method, tracks in params.items()
    }
    report = {
        "arm_joint_indices_body25": ARM_JOINTS.tolist(),
        "methods": {
            method: summarize_method(skeletons, method_predictions)
            for method, method_predictions in predictions.items()
        },
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
