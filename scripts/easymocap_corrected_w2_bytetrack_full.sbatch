#!/bin/bash
#SBATCH --job-name=em_bt_full
#SBATCH --partition=CS286
#SBATCH --gres=gpu:NVIDIAGeForceRTX2080Ti:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:30:00
#SBATCH --output=/public/home/CS286/qiyt2023-CS286/EasyMocap/corrected_w2_bt_full-%j.out

set -euo pipefail

ROOT=/public/home/CS286/qiyt2023-CS286/EasyMocap
GVHMR=/public/home/CS286/qiyt2023-CS286/GVHMR
DATA="${ROOT}/data/caltennis_0224_4cam_corrected_w2_bytetrack"
OUT="${DATA}/output"
RAW="${OUT}/mvmp_bytetrack_raw"
TRACK="${OUT}/mvmp_bytetrack_track3d"
STITCH="${OUT}/mvmp_bytetrack_stitched4"
SMPL="${OUT}/mvmp_bytetrack_smpl"
PYTHON=/public/home/CS286/qiyt2023-CS286/miniconda3/envs/gvhmr/bin/python

export PYTHONPATH="${ROOT}:${GVHMR}"
export PYOPENGL_PLATFORM=egl
cd "${ROOT}"

if [ ! -f "${DATA}/images/W_02/000959.jpg" ]; then
  "${PYTHON}" scripts/prepare_caltennis_easymocap.py "${DATA}" \
    --cameras W_01 W_02 E_03 E_04 --height 1080
fi

echo '===== YOLOv8 + ByteTrack + ViTPose-H ====='
"${PYTHON}" scripts/detect_caltennis_4cam_bytetrack.py "${DATA}" \
  --cameras W_01 W_02 E_03 E_04 \
  --bytetrack --tracker bytetrack.yaml --det-conf 0.25 \
  --vitpose "${GVHMR}/inputs/checkpoints/vitpose/vitpose-h-multi-coco.pth" \
  --yolo "${GVHMR}/inputs/checkpoints/yolo/yolov8x.pt" --batch 8

echo '===== EasyMocap multiview association and triangulation ====='
"${PYTHON}" apps/demo/mvmp.py "${DATA}" \
  --out "${RAW}" --sub W_01 W_02 E_03 E_04 \
  --body body25 --thres2d 0.2 --MAX_REPRO_ERROR 20 \
  --cfg config/exp/mvmp4cam.yml

echo '===== EasyMocap temporal Track3D ====='
"${PYTHON}" apps/demo/auto_track.py "${RAW}" "${TRACK}" --track3d
"${PYTHON}" scripts/evaluate_track3d_pilot.py \
  --raw "${RAW}/keypoints3d" --tracked "${TRACK}/keypoints3d" \
  --out "${TRACK}/track3d_quality.json"

echo '===== Court-position four-ID stitching pilot ====='
"${PYTHON}" scripts/stitch_track3d_fragments.py \
  --input "${TRACK}/keypoints3d" --output "${STITCH}" \
  --clusters 4 --max-interp-gap 15 --max-track-center-distance 2.5
"${PYTHON}" scripts/evaluate_track3d_pilot.py \
  --raw "${TRACK}/keypoints3d" --tracked "${STITCH}/keypoints3d" \
  --out "${STITCH}/stitched4_quality.json"

echo '===== SMPL fitting from tracked 3D skeletons ====='
"${PYTHON}" apps/demo/smpl_from_keypoints.py "${DATA}" \
  --skel3d "${TRACK}/keypoints3d" --sub W_01 W_02 E_03 E_04 \
  --out "${SMPL}" --model smpl --gender female --body body25 \
  --vis_smpl --write_smpl_full

echo CORRECTED_W2_BYTETRACK_FULL_DONE

