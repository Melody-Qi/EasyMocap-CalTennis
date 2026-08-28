#!/bin/bash
#SBATCH --job-name=easymocap_4cam_mvs
#SBATCH --partition=CS286
#SBATCH --gres=gpu:NVIDIAGeForceRTX2080Ti:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/public/home/CS286/qiyt2023-CS286/EasyMocap/4cam_mvs-%j.out

set -euo pipefail

ROOT=/public/home/CS286/qiyt2023-CS286/EasyMocap
GVHMR=/public/home/CS286/qiyt2023-CS286/GVHMR
DATA="${ROOT}/data/caltennis_0224_4cam"
OUT="${DATA}/output"
PYTHON=/public/home/CS286/qiyt2023-CS286/miniconda3/envs/gvhmr/bin/python

export PYTHONPATH="${ROOT}:${GVHMR}"
export PYOPENGL_PLATFORM=egl
cd "${ROOT}"

echo '===== mvmp side A (W_01 W_02) ====='
"${PYTHON}" apps/demo/mvmp.py "${DATA}" --out "${OUT}/mvmp_A" --sub W_01 W_02 --body body25 --thres2d 0.2 --MAX_REPRO_ERROR 20 --vis_repro
echo '===== mvmp side B (E_03 E_04) ====='
"${PYTHON}" apps/demo/mvmp.py "${DATA}" --out "${OUT}/mvmp_B" --sub E_03 E_04 --body body25 --thres2d 0.2 --MAX_REPRO_ERROR 20 --vis_repro
echo '===== auto_track A ====='
"${PYTHON}" apps/demo/auto_track.py "${OUT}/mvmp_A" "${OUT}/track_A" --track3d
echo '===== auto_track B ====='
"${PYTHON}" apps/demo/auto_track.py "${OUT}/mvmp_B" "${OUT}/track_B" --track3d
echo '===== smpl A ====='
"${PYTHON}" apps/demo/smpl_from_keypoints.py "${OUT}/track_A" --skel3d "${OUT}/track_A/keypoints3d" --sub W_01 W_02 --out "${OUT}/smpl_A" --model smpl --gender female --body body25 --vis_smpl --write_smpl_full
echo '===== smpl B ====='
"${PYTHON}" apps/demo/smpl_from_keypoints.py "${OUT}/track_B" --skel3d "${OUT}/track_B/keypoints3d" --sub E_03 E_04 --out "${OUT}/smpl_B" --model smpl --gender female --body body25 --vis_smpl --write_smpl_full
echo '===== QA ====='
"${PYTHON}" scripts/qa_caltennis_4cam.py --data "${DATA}" --track_A "${OUT}/track_A" --track_B "${OUT}/track_B"
echo '===== ffmpeg vis -> mp4 (best-effort) ====='
module load ffmpeg
for side in A B; do
  for cam in "${OUT}/mvmp_${side}/vis_repro"/*; do
    c=$(basename "$cam")
    ffmpeg -y -framerate 30 -i "${cam}/%06d.png" -c:v libx264 -pix_fmt yuv420p -an "${OUT}/mvmp_${side}/repro_${c}.mp4" || echo "ffmpeg repro ${side} ${c} skipped"
  done
  ffmpeg -y -framerate 30 -i "${OUT}/smpl_${side}/vis_smpl/%06d.png" -c:v libx264 -pix_fmt yuv420p -an "${OUT}/smpl_${side}/smpl.mp4" || echo "ffmpeg smpl ${side} skipped"
done
echo MVSMPL_DONE
