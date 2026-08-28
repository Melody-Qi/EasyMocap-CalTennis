#!/bin/bash
#SBATCH --job-name=em_bt_vis
#SBATCH --partition=CS286
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --output=/public/home/CS286/qiyt2023-CS286/EasyMocap/corrected_w2_bt_vis-%j.out

set -euo pipefail

ROOT=/public/home/CS286/qiyt2023-CS286/EasyMocap
DATA="${ROOT}/data/caltennis_0224_4cam_corrected_w2_bytetrack"
TRACK="${DATA}/output/mvmp_bytetrack_stitched4/keypoints3d"
OUT="${DATA}/output/mvmp_bytetrack_stitched4_visualization"
PYTHON=/public/home/CS286/qiyt2023-CS286/miniconda3/envs/gvhmr/bin/python

export PYTHONPATH="${ROOT}"
cd "${ROOT}"

"${PYTHON}" scripts/visualize_track3d_pilot.py \
  --root "${DATA}" --tracked "${TRACK}" --out "${OUT}" \
  --cameras W_01 W_02 E_03 E_04 --scale 0.5 \
  --topdown-subtitle "Corrected W2 plus ByteTrack: four stitched court-position IDs"

module load ffmpeg
mkdir -p "${OUT}/videos"
for cam in W_01 W_02 E_03 E_04; do
  ffmpeg -y -framerate 30 -pattern_type glob -i "${OUT}/frames/${cam}/*.jpg" \
    -c:v mpeg4 -q:v 3 -pix_fmt yuv420p "${OUT}/videos/${cam}_overlay.mp4"
done
ffmpeg -y \
  -i "${OUT}/videos/W_01_overlay.mp4" -i "${OUT}/videos/W_02_overlay.mp4" \
  -i "${OUT}/videos/E_03_overlay.mp4" -i "${OUT}/videos/E_04_overlay.mp4" \
  -filter_complex "[0:v][1:v]hstack=inputs=2[top];[2:v][3:v]hstack=inputs=2[bottom];[top][bottom]vstack=inputs=2[v]" \
  -map "[v]" -c:v mpeg4 -q:v 4 -pix_fmt yuv420p \
  "${OUT}/corrected_w2_bytetrack_four_camera_montage.mp4"

echo CORRECTED_W2_BYTETRACK_VIS_DONE
