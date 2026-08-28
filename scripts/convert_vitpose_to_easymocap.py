"""Align two CalTennis views and convert GVHMR ViTPose output to EasyMocap.

The temporal offset is selected by minimizing symmetric epipolar distance over
confident COCO-17 joints. X2 keypoints are linearly interpolated at sub-frame
times; images use the nearest source frame for visualization only.
"""

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch

from easymocap.dataset.config import coco17tobody25


def skew(v):
    x, y, z = v.reshape(3)
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=np.float64)


def load_camera(path):
    d = json.loads(Path(path).read_text())
    return (
        np.asarray(d["K"], dtype=np.float64),
        np.asarray(d["R_w2c"], dtype=np.float64),
        np.asarray(d["t_w2c"], dtype=np.float64).reshape(3),
    )


def fundamental(k1, r1, t1, k2, r2, t2):
    r21 = r2 @ r1.T
    t21 = t2 - r21 @ t1
    return np.linalg.inv(k2).T @ skew(t21) @ r21 @ np.linalg.inv(k1)


def interp_pose(seq, times):
    n = len(seq)
    lo = np.floor(times).astype(int)
    hi = np.clip(lo + 1, 0, n - 1)
    lo = np.clip(lo, 0, n - 1)
    alpha = (times - lo)[:, None, None]
    out = (1 - alpha) * seq[lo] + alpha * seq[hi]
    # Confidence interpolation should be conservative.
    out[..., 2] = np.minimum(seq[lo, :, 2], seq[hi, :, 2])
    return out


def symmetric_epipolar_errors(kp1, kp2, fmat, threshold):
    valid = (kp1[..., 2] >= threshold) & (kp2[..., 2] >= threshold)
    x1 = np.concatenate([kp1[..., :2], np.ones((*kp1.shape[:2], 1))], axis=-1)
    x2 = np.concatenate([kp2[..., :2], np.ones((*kp2.shape[:2], 1))], axis=-1)
    l2 = np.einsum("ij,tkj->tki", fmat, x1)
    l1 = np.einsum("ij,tkj->tki", fmat.T, x2)
    numer = np.abs(np.sum(x2 * l2, axis=-1))
    d2 = numer / np.maximum(np.linalg.norm(l2[..., :2], axis=-1), 1e-9)
    d1 = numer / np.maximum(np.linalg.norm(l1[..., :2], axis=-1), 1e-9)
    err = 0.5 * (d1 + d2)
    return err[valid]


def bbox_from_keypoints(kp, width=1920, height=1080):
    valid = kp[:, 2] > 0.05
    if not valid.any():
        return [0.0, 0.0, float(width - 1), float(height - 1), 0.0]
    xy = kp[valid, :2]
    lo, hi = xy.min(axis=0), xy.max(axis=0)
    pad = max(20.0, 0.15 * max(hi - lo))
    x1, y1 = np.maximum(lo - pad, [0, 0])
    x2, y2 = np.minimum(hi + pad, [width - 1, height - 1])
    return [float(x1), float(y1), float(x2), float(y2), float(kp[valid, 2].mean())]


def write_annot(path, image_rel, kp):
    body25 = coco17tobody25(kp[None])[0]
    payload = {
        "filename": image_rel,
        "height": 1080,
        "width": 1920,
        "annots": [{
            "personID": 0,
            "bbox": bbox_from_keypoints(body25),
            "keypoints": body25.tolist(),
            "area": 0.0,
        }],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--gvhmr-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--search", type=float, default=12.0)
    parser.add_argument("--step", type=float, default=0.1)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--sync-only", action="store_true")
    args = parser.parse_args()

    kp1 = torch.load(args.gvhmr_root / "outputs/easymocap_x1x2_X1/X1/preprocess/vitpose.pt",
                     map_location="cpu").cpu().numpy()
    kp2 = torch.load(args.gvhmr_root / "outputs/easymocap_x1x2_X2/X2/preprocess/vitpose.pt",
                     map_location="cpu").cpu().numpy()
    if kp1.shape != kp2.shape or kp1.shape[1:] != (17, 3):
        raise RuntimeError(f"Unexpected keypoint shapes: {kp1.shape}, {kp2.shape}")

    k1, r1, t1 = load_camera(args.source_root / "calib_json/X1.json")
    k2, r2, t2 = load_camera(args.source_root / "calib_json/X2.json")
    fmat = fundamental(k1, r1, t1, k2, r2, t2)

    candidates = np.arange(-args.search, args.search + args.step / 2, args.step)
    scores = []
    base = np.arange(len(kp1), dtype=np.float64)
    for delta in candidates:
        valid_t = (base + delta >= 0) & (base + delta <= len(kp2) - 1)
        p2 = interp_pose(kp2, base[valid_t] + delta)
        errors = symmetric_epipolar_errors(kp1[valid_t], p2, fmat, args.conf)
        score = float(np.median(errors)) if len(errors) else float("inf")
        scores.append(score)
    best_idx = int(np.argmin(scores))
    best_delta = float(candidates[best_idx])

    # Retain only the common interval for the best sub-frame shift.
    valid_indices = base[(base + best_delta >= 0) & (base + best_delta <= len(kp2) - 1)].astype(int)
    kp1_aligned = kp1[valid_indices]
    x2_times = valid_indices.astype(np.float64) + best_delta
    kp2_aligned = interp_pose(kp2, x2_times)

    args.output_root.mkdir(parents=True, exist_ok=True)
    ranking = sorted(zip(candidates.tolist(), scores), key=lambda x: x[1])[:20]
    summary = {
        "definition": "X2 frame (i + delta) corresponds to X1 frame i",
        "best_delta_frames": best_delta,
        "best_delta_ms_at_30fps": best_delta / 30.0 * 1000.0,
        "median_symmetric_epipolar_error_px": scores[best_idx],
        "output_frames": len(valid_indices),
        "top_candidates": [{"delta_frames": d, "median_error_px": s} for d, s in ranking],
        "mean_confidence_X1": float(kp1_aligned[..., 2].mean()),
        "mean_confidence_X2": float(kp2_aligned[..., 2].mean()),
    }
    (args.output_root / "sync_summary.json").write_text(json.dumps(summary, indent=2))
    if args.sync_only:
        print(json.dumps(summary, indent=2))
        return

    shutil.copy2(args.source_root / "intri.yml", args.output_root / "intri.yml")
    shutil.copy2(args.source_root / "extri.yml", args.output_root / "extri.yml")
    for cam in ["X1", "X2"]:
        (args.output_root / "images" / cam).mkdir(parents=True, exist_ok=True)
        (args.output_root / "annots" / cam).mkdir(parents=True, exist_ok=True)

    for out_idx, src1_idx in enumerate(valid_indices):
        src2_idx = int(np.clip(np.rint(x2_times[out_idx]), 0, len(kp2) - 1))
        dst_name = f"{out_idx:06d}.jpg"
        src1 = args.source_root / "images/X1" / f"{src1_idx:06d}.jpg"
        src2 = args.source_root / "images/X2" / f"{src2_idx:06d}.jpg"
        shutil.copy2(src1, args.output_root / "images/X1" / dst_name)
        shutil.copy2(src2, args.output_root / "images/X2" / dst_name)
        write_annot(args.output_root / "annots/X1" / f"{out_idx:06d}.json",
                    f"images/X1/{dst_name}", kp1_aligned[out_idx])
        write_annot(args.output_root / "annots/X2" / f"{out_idx:06d}.json",
                    f"images/X2/{dst_name}", kp2_aligned[out_idx])

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
