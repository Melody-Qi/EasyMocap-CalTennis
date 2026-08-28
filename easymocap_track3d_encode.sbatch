#!/bin/bash
#SBATCH --job-name=em_track3d_encode
#SBATCH --partition=CS286
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/public/home/CS286/qiyt2023-CS286/EasyMocap/track3d_encode-%j.out

set -euo pipefail

OUT=/public/home/CS286/qiyt2023-CS286/EasyMocap/data/caltennis_0224_4cam/output/mvmp_all_trackpilot_visualization

module load ffmpeg
mkdir -p "${OUT}/videos"

# The cluster FFmpeg build does not expose libx264/-crf.  MPEG-4 is supported
# and produces ordinary MP4 files that VLC/PotPlayer can open.
for cam in W_01 W_02 E_03 E_04; do
  ffmpeg -y -framerate 30 \
    -i "${OUT}/frames/${cam}/%06d.jpg" \
    -c:v mpeg4 -q:v 3 -pix_fmt yuv420p \
    "${OUT}/videos/${cam}_track3d_overlay.mp4"
done

ffmpeg -y \
  -i "${OUT}/videos/W_01_track3d_overlay.mp4" \
  -i "${OUT}/videos/W_02_track3d_overlay.mp4" \
  -i "${OUT}/videos/E_03_track3d_overlay.mp4" \
  -i "${OUT}/videos/E_04_track3d_overlay.mp4" \
  -filter_complex "[0:v][1:v]hstack=inputs=2[top];[2:v][3:v]hstack=inputs=2[bottom];[top][bottom]vstack=inputs=2[v]" \
  -map "[v]" -c:v mpeg4 -q:v 4 -pix_fmt yuv420p \
  "${OUT}/track3d_four_camera_montage.mp4"

echo TRACK3D_ENCODE_DONE
