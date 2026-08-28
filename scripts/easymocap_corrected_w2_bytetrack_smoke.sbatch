#!/bin/bash
#SBATCH --job-name=em_bt_smoke
#SBATCH --partition=CS286
#SBATCH --gres=gpu:NVIDIAGeForceRTX2080Ti:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/public/home/CS286/qiyt2023-CS286/EasyMocap/corrected_w2_bt_smoke-%j.out

set -euo pipefail

ROOT=/public/home/CS286/qiyt2023-CS286/EasyMocap
GVHMR=/public/home/CS286/qiyt2023-CS286/GVHMR
DATA="${ROOT}/data/caltennis_0224_4cam_corrected_w2_bytetrack"
PYTHON=/public/home/CS286/qiyt2023-CS286/miniconda3/envs/gvhmr/bin/python

export PYTHONPATH="${ROOT}:${GVHMR}"
export PYOPENGL_PLATFORM=egl
cd "${ROOT}"

if [ ! -f "${DATA}/images/W_02/000959.jpg" ]; then
  "${PYTHON}" scripts/prepare_caltennis_easymocap.py "${DATA}" \
    --cameras W_01 W_02 E_03 E_04 --height 1080
fi

"${PYTHON}" scripts/detect_caltennis_4cam_bytetrack.py "${DATA}" \
  --cameras W_01 W_02 E_03 E_04 \
  --limit 20 --bytetrack --tracker bytetrack.yaml --det-conf 0.25 \
  --vitpose "${GVHMR}/inputs/checkpoints/vitpose/vitpose-h-multi-coco.pth" \
  --yolo "${GVHMR}/inputs/checkpoints/yolo/yolov8x.pt" --batch 8

echo CORRECTED_W2_BYTETRACK_SMOKE_DONE

