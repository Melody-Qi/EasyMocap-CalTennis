"""Redraw only the stitched-four Track3D top-down QA plot."""

import os
import sys


ROOT = "/public/home/CS286/qiyt2023-CS286/EasyMocap"
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from visualize_track3d_pilot import draw_topdown  # noqa: E402


if __name__ == "__main__":
    base = os.path.join(
        ROOT,
        "data/caltennis_0224_4cam/output/mvmp_all_trackpilot_stitched4_v2",
    )
    draw_topdown(
        os.path.join(base, "keypoints3d"),
        os.path.join(
            ROOT,
            "data/caltennis_0224_4cam/output/"
            "mvmp_all_trackpilot_stitched4_v2_visualization/"
            "track3d_topdown_trajectories.png",
        ),
        "38 input fragments stitched into four stable court-position IDs",
    )
