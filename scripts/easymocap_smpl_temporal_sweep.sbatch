#!/bin/bash
#SBATCH --job-name=em_smpl_sweep
#SBATCH --partition=CS286
#SBATCH --gres=gpu:NVIDIAGeForceRTX2080Ti:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/public/home/CS286/qiyt2023-CS286/EasyMocap/smpl-temporal-sweep-%j.out

set -euo pipefail

ROOT=/public/home/CS286/qiyt2023-CS286/EasyMocap
DATA="${ROOT}/data/caltennis_0224_4cam_corrected_w2_bytetrack"
SKEL="${DATA}/output/mvmp_bytetrack_stitched4_nointerp/keypoints3d"
PYTHON=/public/home/CS286/qiyt2023-CS286/miniconda3/envs/gvhmr/bin/python

export PYTHONPATH="${ROOT}:/public/home/CS286/qiyt2023-CS286/GVHMR"
cd "${ROOT}"

fit() {
  local out="$1"
  shift
  "${PYTHON}" apps/demo/smpl_from_keypoints.py "${DATA}" \
    --skel3d "${SKEL}" --sub W_01 W_02 E_03 E_04 \
    --out "${DATA}/output/${out}" --model smpl --gender female --body body25 \
    --write_smpl_full --opts "$@"
}

# Same non-interpolated input for every configuration.
fit mvmp_bytetrack_stitched4_nointerp_smpl_default \
  smooth_body 5 smooth_poses 1 smooth_Rh 1
fit mvmp_bytetrack_stitched4_nointerp_smpl_weak \
  smooth_body 0.5 smooth_poses 0.1 smooth_Rh 0.2
fit mvmp_bytetrack_stitched4_nointerp_smpl_rootonly \
  smooth_body 0 smooth_poses 0 smooth_Rh 1

"${PYTHON}" scripts/compare_smpl_temporal_lag.py \
  --skeletons "${SKEL}" \
  --method "default_same_input=${DATA}/output/mvmp_bytetrack_stitched4_nointerp_smpl_default" \
  --method "weak_temporal=${DATA}/output/mvmp_bytetrack_stitched4_nointerp_smpl_weak" \
  --method "root_only_temporal=${DATA}/output/mvmp_bytetrack_stitched4_nointerp_smpl_rootonly" \
  --method "zero_temporal=${DATA}/output/mvmp_bytetrack_stitched4_nointerp_smpl_singleframe" \
  --output "${DATA}/output/smpl_temporal_weight_sweep_metrics.json"

echo SMPL_TEMPORAL_WEIGHT_SWEEP_DONE
