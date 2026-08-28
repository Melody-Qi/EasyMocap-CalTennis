#!/bin/bash
#SBATCH --job-name=em_bt_smpl_sf
#SBATCH --partition=CS286
#SBATCH --gres=gpu:NVIDIAGeForceRTX2080Ti:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/public/home/CS286/qiyt2023-CS286/EasyMocap/corrected_w2_bt_smpl_singleframe-%j.out

set -euo pipefail

ROOT=/public/home/CS286/qiyt2023-CS286/EasyMocap
DATA="${ROOT}/data/caltennis_0224_4cam_corrected_w2_bytetrack"
STITCH="${DATA}/output/mvmp_bytetrack_stitched4_nointerp"
SMPL="${DATA}/output/mvmp_bytetrack_stitched4_nointerp_smpl_singleframe"
PYTHON=/public/home/CS286/qiyt2023-CS286/miniconda3/envs/gvhmr/bin/python

export PYTHONPATH="${ROOT}:/public/home/CS286/qiyt2023-CS286/GVHMR"
export PYOPENGL_PLATFORM=egl
cd "${ROOT}"

echo '===== Ablation: no gap interpolation + zero temporal SMPL losses ====='
"${PYTHON}" apps/demo/smpl_from_keypoints.py "${DATA}" \
  --skel3d "${STITCH}/keypoints3d" --sub W_01 W_02 E_03 E_04 \
  --out "${SMPL}" --model smpl --gender female --body body25 \
  --vis_smpl --write_smpl_full \
  --opts smooth_body 0 smooth_poses 0 smooth_Rh 0

echo CORRECTED_W2_BYTETRACK_SMPL_SINGLEFRAME_DONE
