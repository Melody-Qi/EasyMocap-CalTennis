#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import cv2
import numpy as np


BONES = [
    (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7),
    (1, 8), (8, 9), (9, 10), (10, 11), (8, 12), (12, 13), (13, 14),
]


def read_node(path, name):
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    value = fs.getNode(name).mat()
    fs.release()
    return value


def load_person(path, field):
    with path.open('r', encoding='utf-8') as f:
        data = json.load(f)
    if isinstance(data, dict) and 'annots' in data:
        data = data['annots']
    return np.asarray(data[0][field], dtype=np.float64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_root', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()

    root = args.data_root
    out = root / 'output' / 'easymocap_smpl'
    frames = sorted((out / 'keypoints3d').glob('*.json'))
    if not frames:
        raise RuntimeError('No keypoints3d JSON files found')

    cameras = ['X1', 'X2']
    projections = {}
    for cam in cameras:
        K = read_node(root / 'intri.yml', f'K_{cam}')
        R = read_node(root / 'extri.yml', f'Rot_{cam}')
        T = read_node(root / 'extri.yml', f'T_{cam}')
        projections[cam] = K @ np.concatenate([R, T], axis=1)

    all_k3d = []
    repro_errors = {cam: [] for cam in cameras}
    joint_counts = []
    for frame_path in frames:
        name = frame_path.name
        k3d = load_person(frame_path, 'keypoints3d')
        all_k3d.append(k3d)
        valid3d = k3d[:, 3] > 0
        joint_counts.append(int(valid3d.sum()))
        xyz1 = np.concatenate([k3d[:, :3], np.ones((len(k3d), 1))], axis=1)
        for cam in cameras:
            ann = load_person(root / 'annots' / cam / name, 'keypoints')
            uvw = (projections[cam] @ xyz1.T).T
            uv = uvw[:, :2] / uvw[:, 2:3]
            valid = valid3d & (ann[:, 2] > 0.2) & (uvw[:, 2] > 0)
            if np.any(valid):
                repro_errors[cam].extend(np.linalg.norm(uv[valid] - ann[valid, :2], axis=1).tolist())

    k3d_seq = np.stack(all_k3d)
    bone_stats = []
    for a, b in BONES:
        valid = (k3d_seq[:, a, 3] > 0) & (k3d_seq[:, b, 3] > 0)
        lengths = np.linalg.norm(k3d_seq[valid, a, :3] - k3d_seq[valid, b, :3], axis=1)
        if len(lengths):
            mean = float(np.mean(lengths))
            bone_stats.append({'joint_pair': [a, b], 'mean_m': mean,
                               'std_m': float(np.std(lengths)),
                               'cv': float(np.std(lengths) / mean) if mean else None})

    pelvis = k3d_seq[:, 8, :3]
    pelvis_valid = k3d_seq[:, 8, 3] > 0
    pelvis_valid_xyz = pelvis[pelvis_valid]
    accel = np.diff(pelvis_valid_xyz, n=2, axis=0) if len(pelvis_valid_xyz) > 2 else np.empty((0, 3))
    heights = []
    for frame in k3d_seq:
        valid = frame[:, 3] > 0
        if np.any(valid):
            heights.append(float(frame[valid, 2].max() - frame[valid, 2].min()))

    shapes = []
    for path in sorted((out / 'smpl').glob('*.json')):
        shapes.append(load_person(path, 'shapes').reshape(-1))
    shapes = np.stack(shapes) if shapes else np.empty((0, 10))

    def error_summary(values):
        arr = np.asarray(values)
        return {'count': int(arr.size), 'median_px': float(np.median(arr)),
                'p90_px': float(np.percentile(arr, 90)), 'mean_px': float(np.mean(arr))}

    summary = {
        'frames': len(frames),
        'duration_seconds_at_30fps': len(frames) / 30.0,
        'valid_3d_joints_per_frame': {
            'median': float(np.median(joint_counts)),
            'min': int(np.min(joint_counts)),
            'max': int(np.max(joint_counts)),
        },
        'reprojection_error': {cam: error_summary(repro_errors[cam]) for cam in cameras},
        'skeleton_height_range_m': {
            'median': float(np.median(heights)),
            'p10': float(np.percentile(heights, 10)),
            'p90': float(np.percentile(heights, 90)),
        },
        'pelvis_world_xyz_range_m': (np.ptp(pelvis_valid_xyz, axis=0).tolist()
                                     if len(pelvis_valid_xyz) else None),
        'pelvis_second_difference_m_per_frame2': {
            'median': float(np.median(np.linalg.norm(accel, axis=1))) if len(accel) else None,
            'p90': float(np.percentile(np.linalg.norm(accel, axis=1), 90)) if len(accel) else None,
        },
        'bone_length_cv': {
            'median': float(np.median([x['cv'] for x in bone_stats])),
            'p90': float(np.percentile([x['cv'] for x in bone_stats], 90)),
            'per_bone': bone_stats,
        },
        'smpl_frames': int(len(shapes)),
        'smpl_shape_std_max': float(np.max(np.std(shapes, axis=0))) if len(shapes) else None,
    }

    output = args.output or (out / 'quality_summary.json')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
