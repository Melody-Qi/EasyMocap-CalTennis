"""Prepare a two-camera CalTennis clip for EasyMocap.

Converts CalTennis calib.json files to EasyMocap OpenCV YAML and extracts
frame-aligned images. The supplied clips are already coarse-synchronized;
sub-frame synchronization is refined later from the 2D pose trajectories.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def write_matrix(fs, name, value):
    fs.write(name, np.asarray(value, dtype=np.float64))


def convert_cameras(root: Path, cameras):
    intri_path = root / "intri.yml"
    extri_path = root / "extri.yml"
    intri = cv2.FileStorage(str(intri_path), cv2.FILE_STORAGE_WRITE)
    extri = cv2.FileStorage(str(extri_path), cv2.FILE_STORAGE_WRITE)
    intri.startWriteStruct("names", cv2.FileNode_SEQ)
    for cam in cameras:
        intri.write("", cam)
    intri.endWriteStruct()
    extri.startWriteStruct("names", cv2.FileNode_SEQ)
    for cam in cameras:
        extri.write("", cam)
    extri.endWriteStruct()

    for cam in cameras:
        data = json.loads((root / "calib_json" / f"{cam}.json").read_text())
        k = np.asarray(data["K"], dtype=np.float64)
        rot = np.asarray(data["R_w2c"], dtype=np.float64)
        trans = np.asarray(data["t_w2c"], dtype=np.float64).reshape(3, 1)
        rvec, _ = cv2.Rodrigues(rot)
        write_matrix(intri, f"K_{cam}", k)
        write_matrix(intri, f"dist_{cam}", np.zeros((1, 5), dtype=np.float64))
        write_matrix(extri, f"R_{cam}", rvec)
        write_matrix(extri, f"Rot_{cam}", rot)
        write_matrix(extri, f"T_{cam}", trans)
    intri.release()
    extri.release()
    return intri_path, extri_path


def extract_frames(root: Path, cameras, target_height: int):
    counts = {}
    for cam in cameras:
        src = root / "videos" / f"{cam}.mp4"
        dst = root / "images" / cam
        dst.mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(str(src))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open {src}")
        count = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame.shape[0] < target_height:
                raise RuntimeError(f"{src}: height {frame.shape[0]} < {target_height}")
            frame = frame[:target_height, :, :]
            out = dst / f"{count:06d}.jpg"
            if not cv2.imwrite(str(out), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                raise RuntimeError(f"Cannot write {out}")
            count += 1
        cap.release()
        counts[cam] = count
    if len(set(counts.values())) != 1:
        raise RuntimeError(f"View frame counts differ: {counts}")
    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--cameras", nargs="+", default=["X1", "X2"])
    parser.add_argument("--height", type=int, default=1080)
    args = parser.parse_args()

    intri, extri = convert_cameras(args.root, args.cameras)
    counts = extract_frames(args.root, args.cameras, args.height)
    print(f"intri={intri}")
    print(f"extri={extri}")
    print(f"frames={counts}")


if __name__ == "__main__":
    main()
