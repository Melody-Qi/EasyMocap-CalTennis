#!/bin/bash
#SBATCH --job-name=easymocap_x1x2
#SBATCH --partition=CS286
#SBATCH --gres=gpu:NVIDIAGeForceRTX2080Ti:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/public/home/CS286/qiyt2023-CS286/EasyMocap/easymocap-x1x2-%j.out

set -euo pipefail

ROOT=/public/home/CS286/qiyt2023-CS286/EasyMocap
DATA="${ROOT}/data/caltennis_0227_x1x2_aligned"
OUT="${DATA}/output/easymocap_smpl"
PYTHON=/public/home/CS286/qiyt2023-CS286/miniconda3/envs/gvhmr/bin/python

module load ffmpeg
export PYOPENGL_PLATFORM=egl
export PYTHONPATH="${ROOT}"
cd "${ROOT}"

"${PYTHON}" apps/demo/mv1p.py "${DATA}" \
  --out "${OUT}" \
  --sub X1 X2 \
  --sub_vis X1 X2 \
  --body body25 \
  --model smpl \
  --gender female \
  --thres2d 0.20 \
  --MAX_REPRO_ERROR 20 \
  --smooth3d 0 \
  --vis_smpl --vis_repro \
  --write_smpl_full \
  --verbose
