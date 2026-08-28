#!/bin/bash
#SBATCH --job-name=easymocap_4cam_smpl
#SBATCH --partition=CS286
#SBATCH --gres=gpu:NVIDIAGeForceRTX2080Ti:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/public/home/CS286/qiyt2023-CS286/EasyMocap/4cam_smpl-%j.out

set -euo pipefail

ROOT=/public/home/CS286/qiyt2023-CS286/EasyMocap
DATA="${ROOT}/data/caltennis_0224_4cam"
OUT="${DATA}/output"
PYTHON=/public/home/CS286/qiyt2023-CS286/miniconda3/envs/gvhmr/bin/python

export PYTHONPATH="${ROOT}"
export PYOPENGL_PLATFORM=egl
cd "${ROOT}"

echo '===== SMPL fit (2 players, cross W-E pairing, female model) ====='
"${PYTHON}" apps/demo/smpl_from_keypoints.py "${DATA}" \
  --skel3d "${OUT}/mvmp_all/keypoints3d" \
  --out "${OUT}/smpl_all" \
  --model smpl --gender female --body body25 \
  --vis_smpl --write_smpl_full

echo '===== repro overlay video (best-effort) ====='
"${PYTHON}" scripts/vis_repro_4cam.py "${DATA}" || echo "vis_repro failed"

module load ffmpeg
for cam in W_01 W_02 E_03 E_04; do
  ffmpeg -y -framerate 30 -i "${OUT}/repro_vid/${cam}/%06d.png" \
    -c:v libx264 -pix_fmt yuv420p -an "${OUT}/repro_${cam}.mp4" \
    || echo "ffmpeg repro ${cam} skipped"
done
ffmpeg -y -framerate 30 -i "${OUT}/smpl_all/vis_smpl/%06d.png" \
  -c:v libx264 -pix_fmt yuv420p -an "${OUT}/smpl_all/smpl.mp4" \
  || echo "ffmpeg smpl skipped"

echo SMPLPIPE_DONE
