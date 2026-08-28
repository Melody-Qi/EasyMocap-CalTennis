"""QA for CalTennis 4-cam EasyMocap pseudo-GT.

For each side (A = W_01+W_02, B = E_03+E_04), for each frame, project the
tracked 3D keypoints to each camera and compare against the 2D detections
(person-matched via pelvis projection). Reports median reprojection error
(px) per camera, per side, overall.
"""
import argparse, json, os
from os.path import join
import numpy as np
import cv2


def read_fs(path, name):
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    v = fs.getNode(name).mat()
    fs.release()
    return v


def load_cameras(data):
    intri = join(data, 'intri.yml')
    extri = join(data, 'extri.yml')
    names = [n for n in read_fs(intri, 'names')]
    cams = {}
    for cam in names:
        K = read_fs(intri, f'K_{cam}')
        dist = read_fs(intri, f'dist_{cam}')
        R = read_fs(extri, f'Rot_{cam}')          # 3x3 R_w2c
        T = read_fs(extri, f'T_{cam}').reshape(3)  # 3x1 t_w2c
        cams[cam] = (K, R, T, dist)
    return cams


def project(X, K, R, T, dist):
    Xh = np.concatenate([X, np.ones((len(X), 1))], axis=1)
    x = (K @ (R @ Xh[:, :3].T + T.reshape(3, 1))).T
    x = x[:, :2] / x[:, 2:3]
    return x


def load_k3d(path):
    if not os.path.exists(path):
        return []
    d = json.load(open(path))
    out = []
    for p in d:
        kp = np.asarray(p['keypoints3d'], dtype=np.float64)
        if kp.shape[1] == 4:
            kp = kp[:, :3]
        out.append({'id': int(p.get('id', 0)), 'kp': kp})
    return out


def load_det(path):
    if not os.path.exists(path):
        return []
    d = json.load(open(path))
    out = []
    for a in d.get('annots', []):
        kp = np.asarray(a['keypoints'], dtype=np.float64)  # (25,3) x,y,conf
        out.append({'pid': int(a.get('personID', 0)), 'kp': kp})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--track_A', required=True)
    ap.add_argument('--track_B', required=True)
    ap.add_argument('--cams_A', nargs='+', default=['W_01', 'W_02'])
    ap.add_argument('--cams_B', nargs='+', default=['E_03', 'E_04'])
    ap.add_argument('--conf', type=float, default=0.3)
    args = ap.parse_args()

    cams = load_cameras(args.data)
    sides = {'A': (args.cams_A, args.track_A), 'B': (args.cams_B, args.track_B)}
    summary = {}
    for side, (cam_list, track_dir) in sides.items():
        k3d_dir = join(track_dir, 'keypoints3d')
        frames = sorted(os.listdir(k3d_dir))
        per_cam_errs = {c: [] for c in cam_list}
        per_person_errs = {}
        all_errs = []
        for fn in frames:
            persons = load_k3d(join(k3d_dir, fn))
            dets = {c: load_det(join(args.data, 'annots', c, fn)) for c in cam_list}
            for person in persons:
                pid = person['id']
                X = person['kp']
                per_person_errs.setdefault(pid, [])
                for c in cam_list:
                    K, R, T, dist = cams[c]
                    x2 = project(X, K, R, T, dist)  # (25,2)
                    # person match via pelvis (body25 idx 8)
                    det = dets[c]
                    if len(det) == 0:
                        continue
                    pel = np.array([d['kp'][8, :2] for d in det])
                    d_pel = np.linalg.norm(x2[8] - pel, axis=1)
                    bi = int(np.argmin(d_pel))
                    dkp = det[bi]['kp']
                    valid = dkp[:, 2] >= args.conf
                    if valid.sum() < 3:
                        continue
                    err = np.linalg.norm(x2[valid] - dkp[valid, :2], axis=1)
                    per_cam_errs[c].extend(err.tolist())
                    per_person_errs[pid].extend(err.tolist())
                    all_errs.extend(err.tolist())
        summary[side] = {
            'n_frames': len(frames),
            'per_camera_median_px': {c: float(np.median(per_cam_errs[c])) if per_cam_errs[c] else None for c in cam_list},
            'per_camera_mean_px': {c: float(np.mean(per_cam_errs[c])) if per_cam_errs[c] else None for c in cam_list},
            'per_person_median_px': {str(pid): float(np.median(v)) for pid, v in per_person_errs.items()},
            'overall_median_px': float(np.median(all_errs)) if all_errs else None,
            'overall_mean_px': float(np.mean(all_errs)) if all_errs else None,
            'n_samples': len(all_errs),
        }
    out = {'summary': summary}
    with open(join(args.data, 'quality_summary.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
