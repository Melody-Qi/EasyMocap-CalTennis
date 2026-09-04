#!/usr/bin/env python3
"""Prepare a synchronized Baidu three-camera tennis clip for EasyMocap.

The source videos are never modified.  Frames are sampled at a shared source
frame index, resized without cropping, and written to images/<camera>.  The
script also creates a first-frame montage and a machine-readable manifest.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any


def resolve_tool(name: str, explicit: str | None) -> str:
    candidates = [explicit, shutil.which(name)]
    home = Path.home()
    candidates.extend(
        [
            str(home / "miniconda3" / "envs" / "cs224n" / "bin" / name),
            str(home / "miniconda3" / "envs" / "gvhmr" / "bin" / name),
        ]
    )
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return candidate
    raise FileNotFoundError(f"Cannot find {name}; pass --{name}")


def probe_video(ffprobe: str, path: Path) -> dict[str, Any]:
    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,width,height,r_frame_rate,avg_frame_rate,nb_frames,duration",
        "-of",
        "json",
        str(path),
    ]
    payload = json.loads(subprocess.check_output(command, text=True, encoding="utf-8"))
    stream = (payload.get("streams") or [{}])[0]
    return {
        key: stream.get(key)
        for key in ("codec_name", "width", "height", "r_frame_rate", "avg_frame_rate", "nb_frames", "duration")
    }


def frame_count(folder: Path) -> int:
    return sum(1 for _ in folder.glob("*.jpg"))


def extract_camera(
    ffmpeg: str,
    source: Path,
    destination: Path,
    *,
    start_frame: int,
    source_fps: float,
    output_fps: float,
    max_frames: int,
    width: int,
    height: int,
    quality: int,
) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.glob("*.jpg")):
        raise FileExistsError(f"Destination already has JPEGs: {destination}")
    start_seconds = start_frame / source_fps
    # ffmpeg qscale uses 2 as visually high quality and 31 as lowest.  Convert
    # the familiar JPEG quality scale approximately while keeping bounds safe.
    qscale = max(2, min(31, round(31 - quality * 29 / 100)))
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{start_seconds:.9f}",
        "-i",
        str(source),
        "-vf",
        f"fps={output_fps},scale={width}:{height}:flags=lanczos",
        "-frames:v",
        str(max_frames),
        "-q:v",
        str(qscale),
        "-start_number",
        "0",
        str(destination / "%06d.jpg"),
    ]
    subprocess.run(command, check=True)


def make_montage(image_paths: list[Path], output: Path) -> None:
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("OpenCV is required to create the preview montage") from exc
    images = []
    for path in image_paths:
        image = cv2.imread(str(path))
        if image is None:
            raise RuntimeError(f"Cannot read extracted frame: {path}")
        image = cv2.resize(image, (640, 360), interpolation=cv2.INTER_AREA)
        cv2.putText(image, path.parent.name, (18, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (30, 255, 255), 2)
        images.append(image)
    montage = np.concatenate(images, axis=1)
    if not cv2.imwrite(str(output), montage):
        raise RuntimeError(f"Cannot write montage: {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ffmpeg")
    parser.add_argument("--ffprobe")
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    project_root = config_path.parent.parent
    source_root = Path(config["source_root"]).expanduser()
    clip_root = source_root / config["clip"]
    output_root = Path(config["output_root"])
    if not output_root.is_absolute():
        output_root = project_root / output_root
    output_root.mkdir(parents=True, exist_ok=True)

    ffmpeg = resolve_tool("ffmpeg", args.ffmpeg)
    ffprobe = resolve_tool("ffprobe", args.ffprobe)
    cameras = list(config["cameras"])
    sources = {camera: clip_root / f"{camera}.mp4" for camera in cameras}
    missing = [str(path) for path in sources.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing source videos:\n" + "\n".join(missing))

    metadata = {camera: probe_video(ffprobe, path) for camera, path in sources.items()}
    frame_values = {entry.get("nb_frames") for entry in metadata.values()}
    rate_values = {entry.get("avg_frame_rate") for entry in metadata.values()}
    if len(frame_values) != 1 or len(rate_values) != 1:
        raise RuntimeError(f"Source videos are not frame-compatible: {metadata}")

    for camera, source in sources.items():
        extract_camera(
            ffmpeg,
            source,
            output_root / "images" / camera,
            start_frame=int(config["start_frame"]),
            source_fps=float(config.get("source_fps", 60)),
            output_fps=float(config["output_fps"]),
            max_frames=int(config["max_frames"]),
            width=int(config["output_width"]),
            height=int(config["output_height"]),
            quality=int(config["jpeg_quality"]),
        )

    counts = {camera: frame_count(output_root / "images" / camera) for camera in cameras}
    if len(set(counts.values())) != 1 or next(iter(counts.values())) != int(config["max_frames"]):
        raise RuntimeError(f"Extracted frame counts differ or are incomplete: {counts}")
    make_montage([output_root / "images" / camera / "000000.jpg" for camera in cameras], output_root / "preview_first_frame.jpg")

    manifest = {
        "config": config,
        "config_path": str(config_path),
        "clip_root": str(clip_root),
        "output_root": str(output_root),
        "sources": {camera: str(path) for camera, path in sources.items()},
        "source_metadata": metadata,
        "extracted_frame_counts": counts,
        "sampling_note": "All cameras use the same source start_frame and output fps; no per-camera temporal offset was applied.",
    }
    (output_root / "prepare_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"output_root": str(output_root), "frames": counts}, ensure_ascii=False))


if __name__ == "__main__":
    main()
