#!/bin/bash
#SBATCH --job-name=easymocap_4cam_mvmp
#SBATCH --partition=CS286
#SBATCH --gres=gpu:NVIDIAGeForceRTX2080Ti:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/public/home/CS286/qiyt2023-CS286/EasyMocap/4cam_mvmp-%j.out

set -euo pipefail

ROOT=/public/home/CS286/qiyt2023-CS286/EasyMocap
DATA="${ROOT}/data/caltennis_0224_4cam"
OUT="${DATA}/output"
PYTHON=/public/home/CS286/qiyt2023-CS286/miniconda3/envs/gvhmr/bin/python

export PYTHONPATH="${ROOT}"
export PYOPENGL_PLATFORM=egl
cd "${ROOT}"

echo '===== mvmp ALL 4 cams (correct cross-camera ray-affinity association) ====='
"${PYTHON}" apps/demo/mvmp.py "${DATA}" \
  --out "${OUT}/mvmp_all" \
  --sub W_01 W_02 E_03 E_04 \
  --body body25 \
  --thres2d 0.2 \
  --MAX_REPRO_ERROR 20 \
  --cfg config/exp/mvmp4cam.yml \
  --vis_repro
echo '===== QA (reproject 3D onto both camera pairs) ====='
"${PYTHON}" scripts/qa_caltennis_4cam.py \
  --data "${DATA}" \
  --track_A "${OUT}/mvmp_all" \
  --track_B "${OUT}/mvmp_all" \
  --cams_A W_01 E_03 \
  --cams_B W_02 E_04
echo MVMP_DONE
