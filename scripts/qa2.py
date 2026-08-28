import sys, json, glob, os
from collections import Counter
import numpy as np

root = sys.argv[1] if len(sys.argv) > 1 else "data/caltennis_0224_4cam"
k3dir = os.path.join(root, "output", "mvmp_all", "keypoints3d")

# body25 bone pairs (OpenPose-style indices) and nominal lengths (meters)
BONES = [
    ("thigh_R", 9, 10, 0.45),
    ("shin_R", 10, 11, 0.45),
    ("thigh_L", 12, 13, 0.45),
    ("shin_L", 13, 14, 0.45),
    ("uarm_R", 2, 3, 0.30),
    ("farm_R", 3, 4, 0.27),
    ("uarm_L", 5, 6, 0.30),
    ("farm_L", 6, 7, 0.27),
    ("torso", 1, 8, 0.55),
    ("neck", 1, 0, 0.15),
]

def bone_len(K, i, j):
    a = np.array(K[i][:3]); b = np.array(K[j][:3])
    return float(np.linalg.norm(a - b))

def analyze():
    files = sorted(glob.glob(os.path.join(k3dir, "*.json")))
    n_real_per_frame = []
    n_raw_per_frame = []
    all_real_extent = []
    bad_examples = []
    for f in files:
        persons = json.load(open(f))
        if isinstance(persons, dict):
            persons = persons.get("person", [persons])
        n_raw_per_frame.append(len(persons))
        nf_real = 0
        for p in persons:
            K = p["keypoints3d"]
            # 3D extent = max pairwise distance among joints
            pts = np.array([[k[0], k[1], k[2]] for k in K])
            extent = float(np.ptp(pts, axis=0).max())
            # plausible bones
            ok = 0
            for name, i, j, nom in BONES:
                L = bone_len(K, i, j)
                if 0.5 * nom <= L <= 1.6 * nom:
                    ok += 1
            is_real = (1.2 <= extent <= 3.0) and (ok >= 6)
            if is_real:
                nf_real += 1
                all_real_extent.append(extent)
        n_real_per_frame.append(nf_real)
    print("== RAW persons/frame distribution ==")
    print(dict(sorted(Counter(n_raw_per_frame).items())))
    print("== REAL (plausible) persons/frame distribution ==")
    print(dict(sorted(Counter(n_real_per_frame).items())))
    print("frames with >=2 real:", sum(1 for n in n_real_per_frame if n >= 2))
    print("frames with exactly 2 real:", sum(1 for n in n_real_per_frame if n == 2))
    print("frames with 1 real:", sum(1 for n in n_real_per_frame if n == 1))
    print("frames with 0 real:", sum(1 for n in n_real_per_frame if n == 0))
    if all_real_extent:
        print("real extent: min=%.2f mean=%.2f max=%.2f" % (
            min(all_real_extent), np.mean(all_real_extent), max(all_real_extent)))

if __name__ == "__main__":
    analyze()
