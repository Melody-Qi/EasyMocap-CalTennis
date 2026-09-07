"""Stack three synchronized GVHMR overlay videos into one review video."""

import argparse
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", default="three_camera_gvhmr_montage.mp4")
    args = parser.parse_args()

    root = Path(args.root)
    paths = [root / f"cam{i}_incam.mp4" for i in range(1, 4)]
    captures = [cv2.VideoCapture(str(path)) for path in paths]
    if not all(cap.isOpened() for cap in captures):
        raise RuntimeError(f"Cannot open all inputs: {paths}")
    fps = captures[0].get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) for cap in captures)
    output = root / args.output
    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (1920, 360)
    )
    for frame_index in range(frame_count):
        frames = []
        for camera, cap in enumerate(captures, start=1):
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"camera {camera} ended at frame {frame_index}")
            frame = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA)
            cv2.rectangle(frame, (0, 0), (126, 30), (0, 0, 0), -1)
            cv2.putText(frame, f"camera {camera}", (8, 21),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255),
                        1, cv2.LINE_AA)
            frames.append(frame)
        writer.write(np.hstack(frames))
    writer.release()
    for cap in captures:
        cap.release()
    print(output)


if __name__ == "__main__":
    main()
