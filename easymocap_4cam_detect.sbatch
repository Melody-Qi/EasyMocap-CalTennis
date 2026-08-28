#!/bin/bash
#SBATCH --job-name=easymocap_4cam_detect
#SBATCH --partition=CS286
#SBATCH --gres=gpu:NVIDIAGeForceRTX2080Ti:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/public/home/CS286/qiyt2023-CS286/EasyMocap/4cam_detect-%j.out

set -euo pipefail

ROOT=/public/home/CS286/qiyt2023-CS286/EasyMocap
GVHMR=/public/home/CS286/qiyt2023-CS286/GVHMR
DATA="${ROOT}/data/caltennis_0224_4cam"
PYTHON=/public/home/CS286/qiyt2023-CS286/miniconda3/envs/gvhmr/bin/python

export PYTHONPATH="${ROOT}:${GVHMR}"
export PYOPENGL_PLATFORM=egl
cd "${ROOT}"

"${PYTHON}" scripts/detect_caltennis_4cam.py "${DATA}"   --cameras W_01 W_02 E_03 E_04   --vitpose "${GVHMR}/inputs/checkpoints/vitpose/vitpose-h-multi-coco.pth"   --yolo "${GVHMR}/inputs/checkpoints/yolo/yolov8x.pt"   --batch 8
echo DETECT_DONE
