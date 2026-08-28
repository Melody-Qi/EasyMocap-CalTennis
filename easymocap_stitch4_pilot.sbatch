#!/bin/bash
#SBATCH --job-name=em_stitch4
#SBATCH --partition=CS286
#SBATCH --cpus-per-task=8
#SBATCH --mem=12G
#SBATCH --time=00:20:00
#SBATCH --output=/public/home/CS286/qiyt2023-CS286/EasyMocap/stitch4-%j.out

set -euo pipefail

ROOT=/public/home/CS286/qiyt2023-CS286/EasyMocap
DATA="${ROOT}/data/caltennis_0224_4cam"
INPUT="${DATA}/output/mvmp_all_trackpilot_track3d/keypoints3d"
OUTPUT="${DATA}/output/mvmp_all_trackpilot_stitched4_v2"
PYTHON=/public/home/CS286/qiyt2023-CS286/miniconda3/envs/gvhmr/bin/python

export PYTHONPATH="${ROOT}"
cd "${ROOT}"

"${PYTHON}" scripts/stitch_track3d_fragments.py \
  --input "${INPUT}" \
  --output "${OUTPUT}" \
  --clusters 4 \
  --max-interp-gap 15 \
  --max-track-center-distance 2.5

"${PYTHON}" scripts/evaluate_track3d_pilot.py \
  --raw "${INPUT}" \
  --tracked "${OUTPUT}/keypoints3d" \
  --out "${OUTPUT}/stitched4_quality.json"

echo STITCH4_DONE
