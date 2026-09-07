#!/bin/bash
set -euo pipefail

FFMPEG=/public/software/ffmpeg/ffmpeg
CLIP="${CLIP:-多球}"
START="${START:-545}"
DURATION="${DURATION:-20}"
TAG="${TAG:-duoqiu_545_565s}"
SOURCE_ROOT="/public/home/CS286/qiyt2023-CS286/EasyMocap/BaiduNetdiskDownload/2025-07-08 上体采集数据 3相机 4K 运动员/${CLIP}"
OUT="/public/home/CS286/qiyt2023-CS286/EasyMocap/baidu_mocap_qiyt_pilot/work/videos_${TAG}"
mkdir -p "$OUT"

for camera in 1 2 3; do
  "$FFMPEG" -y -ss "$START" -t "$DURATION" \
    -i "$SOURCE_ROOT/tennis_camera_${camera}.mp4" \
    -vf "fps=30,scale=1920:1080" -an -c:v mpeg4 -q:v 2 \
    "$OUT/tennis_camera_${camera}.mp4" \
    >"$OUT/tennis_camera_${camera}.encode.log" 2>&1
done

ls -lh "$OUT"
