#!/bin/bash
#SBATCH --job-name=em_bt_nointerp
#SBATCH --partition=CS286
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --output=/public/home/CS286/qiyt2023-CS286/EasyMocap/corrected_w2_bt_nointerp-%j.out

set -euo pipefail

ROOT=/public/home/CS286/qiyt2023-CS286/EasyMocap
DATA="${ROOT}/data/caltennis_0224_4cam_corrected_w2_bytetrack"
TRACK="${DATA}/output/mvmp_bytetrack_track3d/keypoints3d"
NOINTERP="${DATA}/output/mvmp_bytetrack_stitched4_nointerp"
VIS="${DATA}/output/mvmp_bytetrack_stitched4_nointerp_visualization"
PYTHON=/public/home/CS286/qiyt2023-CS286/miniconda3/envs/gvhmr/bin/python

export PYTHONPATH="${ROOT}"
cd "${ROOT}"

# Keep Track3D's observed coordinates, but do not synthesize any missing frame.
"${PYTHON}" scripts/stitch_track3d_fragments.py \
  --input "${TRACK}" --output "${NOINTERP}" \
  --clusters 4 --max-interp-gap 0 --max-track-center-distance 2.5

"${PYTHON}" scripts/evaluate_track3d_pilot.py \
  --raw "${TRACK}" --tracked "${NOINTERP}/keypoints3d" \
  --out "${NOINTERP}/nointerp_quality.json"

"${PYTHON}" scripts/visualize_track3d_pilot.py \
  --root "${DATA}" --tracked "${NOINTERP}/keypoints3d" --out "${VIS}" \
  --cameras W_01 W_02 E_03 E_04 --scale 0.5 \
  --topdown-subtitle "Corrected W2 plus ByteTrack: no gap interpolation"

module load ffmpeg
mkdir -p "${VIS}/videos"
for cam in W_01 W_02 E_03 E_04; do
  ffmpeg -y -framerate 30 -pattern_type glob -i "${VIS}/frames/${cam}/*.jpg" \
    -c:v mpeg4 -q:v 3 -pix_fmt yuv420p "${VIS}/videos/${cam}_overlay.mp4"
done
ffmpeg -y \
  -i "${VIS}/videos/W_01_overlay.mp4" -i "${VIS}/videos/W_02_overlay.mp4" \
  -i "${VIS}/videos/E_03_overlay.mp4" -i "${VIS}/videos/E_04_overlay.mp4" \
  -filter_complex "[0:v][1:v]hstack=inputs=2[top];[2:v][3:v]hstack=inputs=2[bottom];[top][bottom]vstack=inputs=2[v]" \
  -map "[v]" -c:v mpeg4 -q:v 4 -pix_fmt yuv420p \
  "${VIS}/corrected_w2_bytetrack_nointerp_montage.mp4"

echo CORRECTED_W2_BYTETRACK_NOINTERP_DONE
