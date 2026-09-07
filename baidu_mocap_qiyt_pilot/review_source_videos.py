"""Probe and create contact sheets for the three-camera Baidu tennis clips."""

import json
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(
    "/public/home/CS286/qiyt2023-CS286/EasyMocap/BaiduNetdiskDownload/"
    "2025-07-08 上体采集数据 3相机 4K 运动员"
)
OUT = Path(
    "/public/home/CS286/qiyt2023-CS286/EasyMocap/"
    "baidu_mocap_qiyt_pilot/work/source_review"
)
CLIPS = ("热身", "多球", "比赛")
CAMERAS = ("tennis_camera_1", "tennis_camera_2", "tennis_camera_3")


def read_at(cap, seconds):
    cap.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000)
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f"cannot read frame at {seconds:.2f}s")
    return cv2.resize(frame, (320, 180), interpolation=cv2.INTER_AREA)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    report = {}
    for clip in CLIPS:
        rows = []
        report[clip] = {}
        for camera in CAMERAS:
            video = ROOT / clip / f"{camera}.mp4"
            cap = cv2.VideoCapture(str(video))
            if not cap.isOpened():
                raise RuntimeError(f"cannot open {video}")
            fps = cap.get(cv2.CAP_PROP_FPS)
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frames / fps
            report[clip][camera] = {
                "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "fps": fps,
                "frames": frames,
                "duration_s": duration,
            }
            times = np.linspace(max(5, duration * 0.08), duration * 0.92, 6)
            cells = []
            for t in times:
                frame = read_at(cap, float(t))
                cv2.rectangle(frame, (0, 0), (145, 24), (0, 0, 0), -1)
                cv2.putText(
                    frame,
                    f"{camera[-1]}  {t:.1f}s",
                    (6, 17),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                cells.append(frame)
            cap.release()
            rows.append(np.hstack(cells))
        cv2.imwrite(str(OUT / f"{clip}_contact_sheet.jpg"), np.vstack(rows))
        # A denser camera-3 timeline helps select a short action-heavy pilot.
        video = ROOT / clip / "tennis_camera_3.mp4"
        cap = cv2.VideoCapture(str(video))
        duration = report[clip]["tennis_camera_3"]["duration_s"]
        timeline = []
        for t in np.arange(10.0, duration, 20.0):
            frame = read_at(cap, float(t))
            frame = cv2.resize(frame, (240, 135), interpolation=cv2.INTER_AREA)
            cv2.rectangle(frame, (0, 0), (88, 22), (0, 0, 0), -1)
            cv2.putText(frame, f"{t:.0f}s", (5, 16), cv2.FONT_HERSHEY_SIMPLEX,
                        0.48, (255, 255, 255), 1, cv2.LINE_AA)
            timeline.append(frame)
        cap.release()
        cols = 8
        blank = np.zeros_like(timeline[0])
        timeline += [blank] * ((-len(timeline)) % cols)
        rows_t = [np.hstack(timeline[i:i + cols]) for i in range(0, len(timeline), cols)]
        cv2.imwrite(str(OUT / f"{clip}_cam3_timeline_20s.jpg"), np.vstack(rows_t))
    (OUT / "video_metadata.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
