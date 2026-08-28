#!/bin/bash
#SBATCH --job-name=smpl-temporal-vis
#SBATCH --partition=CS286
#SBATCH --gres=gpu:NVIDIAGeForceRTX2080Ti:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --array=0-3
#SBATCH --output=/public/home/CS286/qiyt2023-CS286/EasyMocap/smpl-temporal-vis-%A_%a.out

set -euo pipefail

ROOT=/public/home/CS286/qiyt2023-CS286/EasyMocap
DATA="${ROOT}/data/caltennis_0224_4cam_corrected_w2_bytetrack"
PYTHON=/public/home/CS286/qiyt2023-CS286/miniconda3/envs/gvhmr/bin/python

NAMES=(default weak rootonly zero)
RESULTS=(
  mvmp_bytetrack_stitched4_nointerp_smpl_default
  mvmp_bytetrack_stitched4_nointerp_smpl_weak
  mvmp_bytetrack_stitched4_nointerp_smpl_rootonly
  mvmp_bytetrack_stitched4_nointerp_smpl_singleframe
)

NAME="${NAMES[$SLURM_ARRAY_TASK_ID]}"
RESULT="${DATA}/output/${RESULTS[$SLURM_ARRAY_TASK_ID]}"
OUT="${DATA}/output/smpl_temporal_render_${NAME}"

export PYTHONPATH="${ROOT}:/public/home/CS286/qiyt2023-CS286/GVHMR"
export PYOPENGL_PLATFORM=egl
cd "${ROOT}"

echo "===== render ${NAME}: ${RESULT} ====="
"${PYTHON}" scripts/render_smpl_fourcam_robust.py "${DATA}" \
  --subs W_01 W_02 E_03 E_04 \
  --model scripts/smpl_female_render.yml \
  --result "${RESULT}" \
  --output "${OUT}" \
  --fps 30

echo "SMPL_TEMPORAL_RENDER_DONE ${NAME} ${OUT}.mp4"
