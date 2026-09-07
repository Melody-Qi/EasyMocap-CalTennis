#!/bin/bash
#SBATCH --job-name=gvhmr_baidu_dynamic
#SBATCH --partition=CS286
#SBATCH --gres=gpu:NVIDIAGeForceRTX2080Ti:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --array=1-3
#SBATCH --output=/public/home/CS286/qiyt2023-CS286/GVHMR/gvhmr-baidu-dynamic-%A_%a.out

set -euo pipefail

ROOT=/public/home/CS286/qiyt2023-CS286/GVHMR
PYTHON=/public/home/CS286/qiyt2023-CS286/miniconda3/envs/gvhmr/bin/python
TAG="${TAG:-duoqiu_545_565s}"
VIDDIR=/public/home/CS286/qiyt2023-CS286/EasyMocap/baidu_mocap_qiyt_pilot/work/videos_${TAG}
CAM="tennis_camera_${SLURM_ARRAY_TASK_ID}"

module load ffmpeg 2>/dev/null || true
cd "$ROOT"
"$PYTHON" tools/demo/demo.py \
  --video="$VIDDIR/${CAM}.mp4" \
  --output_root="$ROOT/outputs/baidu_${TAG}_${CAM}" \
  -s --verbose
