import sys, json, glob, os
import numpy as np

root = sys.argv[1] if len(sys.argv) > 1 else "data/caltennis_0224_4cam"
annot_dir = os.path.join(root, "annots")
intri = os.path.join(root, "intri.yml")
extri = os.path.join(root, "extri.yml")

PAIR1 = ("W_01", "E_03")  # Player 1
PAIR2 = ("W_02", "E_04")  # Player 2

def load_cams():
    import cv2
    fs = cv2.FileStorage(intri, cv2.FILE_STORAGE_READ)
    names = [fs.getNode("names").at(i).string() for i in range(fs.getNode("names").size())]
    K = {n: fs.getNode("K_" + n).mat() for n in names}
    fs.release()
    fs = cv2.FileStorage(extri, cv2.FILE_STORAGE_READ)
    Rv = {n: fs.getNode("R_" + n).mat().ravel() for n in names}  # Rodrigues
    T = {n: fs.getNode("T_" + n).mat().ravel() for n in names}
    fs.release()
    import cv2
    P = {}
    for n in names:
        R, _ = cv2.Rodrigues(Rv[n])
        Rt = np.hstack([R, T[n].reshape(3, 1)])
        P[n] = K[n] @ Rt
    return names, P

def load_annots(cam, frame):
    p = os.path.join(annot_dir, cam, "%06d.json" % frame)
    if not os.path.exists(p):
        return []
    d = json.load(open(p))
    if isinstance(d, dict):
        d = d.get("annots", d.get("person", []))
    out = []
    for a in d:
        kp = a.get("keypoints", a.get("keypoints2d"))
        if kp is None:
            continue
        arr = np.array(kp, dtype=float)
        if arr.shape[1] >= 3:
            out.append(arr)
    return out

def triang(P1, P2, k1, k2):
    # k1,k2: (25,3) x,y,conf
    X = []
    for j in range(k1.shape[0]):
        if k1[j, 2] < 0.1 or k2[j, 2] < 0.1:
            X.append(np.array([0, 0, 0, 0])); continue
        A = np.zeros((4, 4))
        A[0] = k1[j, 0] * P1[2] - P1[0]
        A[1] = k1[j, 1] * P1[2] - P1[1]
        A[2] = k2[j, 0] * P2[2] - P2[0]
        A[3] = k2[j, 1] * P2[2] - P2[1]
        _, _, Vt = np.linalg.svd(A)
        x = Vt[-1]
        if x[3] == 0:
            X.append(np.array([0, 0, 0, 0])); continue
        x = x / x[3]
        X.append(np.array([x[0], x[1], x[2], 1.0]))
    return np.array(X)

def repro(P, X):
    if X[3] == 0:
        return 1e9
    p = P @ X
    return np.array([p[0] / p[2], p[1] / p[2]])

def best_pair(camA, camB, P, frame):
    detsA = load_annots(camA, frame)
    detsB = load_annots(camB, frame)
    if not detsA or not detsB:
        return None
    best = None
    for i, a in enumerate(detsA):
        for j, b in enumerate(detsB):
            X = triang(P[camA], P[camB], a, b)
            # repro error over confident joints
            errs = []
            for k in range(25):
                if X[k, 3] == 0:
                    continue
                r1 = repro(P[camA], X[k]); r2 = repro(P[camB], X[k])
                errs.append(np.linalg.norm(r1 - a[k, :2]))
                errs.append(np.linalg.norm(r2 - b[k, :2]))
            if not errs:
                continue
            score = np.mean(errs)
            pts = X[X[:, 3] > 0][:, :3]
            extent = float(np.ptp(pts, axis=0).max()) if len(pts) else 0
            if extent < 1.2 or extent > 3.0:
                score += 1e3  # penalize non-human extent
            if best is None or score < best[0]:
                best = (score, X, extent)
    if best is None:
        return None
    return best

def main():
    names, P = load_cams()
    frames = sorted(int(os.path.basename(f)[:-5]) for f in glob.glob(os.path.join(annot_dir, names[0], "*.json")))
    n2 = 0; n1 = 0; n0 = 0
    extents = []
    repro_scores = []
    for fr in frames:
        r1 = best_pair(*PAIR1, P, fr)
        r2 = best_pair(*PAIR2, P, fr)
        valid = 0
        for r in (r1, r2):
            if r is not None and r[0] < 30 and 1.2 <= r[2] <= 3.0:
                valid += 1
                extents.append(r[2]); repro_scores.append(r[0])
        if valid == 2: n2 += 1
        elif valid == 1: n1 += 1
        else: n0 += 1
    print("frames total:", len(frames))
    print("2 valid players:", n2, "(%.1f%%)" % (100 * n2 / len(frames)))
    print("1 valid:", n1, " 0 valid:", n0)
    if extents:
        print("extent min/mean/max: %.2f / %.2f / %.2f" % (min(extents), np.mean(extents), max(extents)))
    if repro_scores:
        print("repro min/mean/max: %.2f / %.2f / %.2f" % (min(repro_scores), np.mean(repro_scores), max(repro_scores)))

if __name__ == "__main__":
    main()
