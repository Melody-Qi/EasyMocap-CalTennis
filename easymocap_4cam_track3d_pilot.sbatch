#!/bin/bash
#SBATCH --job-name=em_4cam_track3d
#SBATCH --partition=CS286
#SBATCH --gres=gpu:NVIDIAGeForceRTX2080Ti:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/public/home/CS286/qiyt2023-CS286/EasyMocap/4cam_track3d-%j.out

set -euo pipefail

ROOT=/public/home/CS286/qiyt2023-CS286/EasyMocap
DATA="${ROOT}/data/caltennis_0224_4cam"
OUT="${DATA}/output"
RAW="${OUT}/mvmp_all_trackpilot_raw"
TRACK="${OUT}/mvmp_all_trackpilot_track3d"
SMPL="${OUT}/mvmp_all_trackpilot_smpl"
PYTHON=/public/home/CS286/qiyt2023-CS286/miniconda3/envs/gvhmr/bin/python

export PYTHONPATH="${ROOT}"
export PYOPENGL_PLATFORM=egl
cd "${ROOT}"

# Keep every previous result untouched. This job reconstructs a clean baseline,
# then applies EasyMocap's own temporal 3D tracker to the correct four-camera run.
if [ ! -d "${RAW}/keypoints3d" ]; then
  "${PYTHON}" apps/demo/mvmp.py "${DATA}" \
    --out "${RAW}" \
    --sub W_01 W_02 E_03 E_04 \
    --body body25 \
    --thres2d 0.2 \
    --MAX_REPRO_ERROR 20 \
    --cfg config/exp/mvmp4cam.yml
fi

"${PYTHON}" apps/demo/auto_track.py "${RAW}" "${TRACK}" --track3d

"${PYTHON}" scripts/evaluate_track3d_pilot.py \
  --raw "${RAW}/keypoints3d" \
  --tracked "${TRACK}/keypoints3d" \
  --out "${TRACK}/track3d_quality.json"

"${PYTHON}" apps/demo/smpl_from_keypoints.py "${DATA}" \
  --skel3d "${TRACK}/keypoints3d" \
  --sub W_01 W_02 E_03 E_04 \
  --out "${SMPL}" \
  --model smpl --gender female --body body25 \
  --vis_smpl --write_smpl_full

echo TRACK3D_PILOT_DONE
