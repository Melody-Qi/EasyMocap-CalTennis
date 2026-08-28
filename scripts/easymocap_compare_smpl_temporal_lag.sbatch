#!/bin/bash
#SBATCH --job-name=em_smpl_lag
#SBATCH --partition=CS286
#SBATCH --gres=gpu:NVIDIAGeForceRTX2080Ti:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH --output=/public/home/CS286/qiyt2023-CS286/EasyMocap/compare_smpl_temporal_lag-%j.out

set -euo pipefail

ROOT=/public/home/CS286/qiyt2023-CS286/EasyMocap
DATA="${ROOT}/data/caltennis_0224_4cam_corrected_w2_bytetrack"
PYTHON=/public/home/CS286/qiyt2023-CS286/miniconda3/envs/gvhmr/bin/python

export PYTHONPATH="${ROOT}:/public/home/CS286/qiyt2023-CS286/GVHMR"
cd "${ROOT}"

"${PYTHON}" scripts/compare_smpl_temporal_lag.py \
  --skeletons "${DATA}/output/mvmp_bytetrack_stitched4_nointerp/keypoints3d" \
  --default "${DATA}/output/mvmp_bytetrack_stitched4_smpl" \
  --singleframe "${DATA}/output/mvmp_bytetrack_stitched4_nointerp_smpl_singleframe" \
  --output "${DATA}/output/smpl_temporal_ablation_metrics.json"

echo SMPL_TEMPORAL_LAG_COMPARISON_DONE
