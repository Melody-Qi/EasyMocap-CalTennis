import sys, json, glob, os
import numpy as np
import cv2

root = sys.argv[1] if len(sys.argv) > 1 else "data/caltennis_0224_4cam"
out_root = os.path.join(root, "output", "mvmp_all")
k3dir = os.path.join(out_root, "keypoints3d")
annot_dir = os.path.join(root, "annots")
intri = os.path.join(root, "intri.yml")
extri = os.path.join(root, "extri.yml")

# Each player is seen by exactly ONE W + ONE E camera (verified diagnostic):
#   Player1 (court x~24) -> (W_01, E_03) ; Player2 (court x~0-4) -> (W_02, E_04)
PAIR1 = ("W_01", "E_03")
PAIR2 = ("W_02", "E_04")

def load_cams():
    fs = cv2.FileStorage(intri, cv2.FILE_STORAGE_READ)
    names = [fs.getNode("names").at(i).string() for i in range(fs.getNode("names").size())]
    K = {n: fs.getNode("K_" + n).mat() for n in names}
    fs.release()
    fs = cv2.FileStorage(extri, cv2.FILE_STORAGE_READ)
    Rv = {n: fs.getNode("R_" + n).mat().ravel() for n in names}
    T = {n: fs.getNode("T_" + n).mat().ravel() for n in names}
    fs.release()
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
        return None
    p = P @ X
    return np.array([p[0] / p[2], p[1] / p[2]])

def best_pair(camA, camB, P, frame):
    """Return the single best (lowest repro) person match between two cameras."""
    detsA = load_annots(camA, frame)
    detsB = load_annots(camB, frame)
    if not detsA or not detsB:
        return None
    best = None
    for a in detsA:
        for b in detsB:
            X = triang(P[camA], P[camB], a, b)
            errs = []
            for k in range(25):
                if X[k, 3] == 0:
                    continue
                r1 = repro(P[camA], X[k]); r2 = repro(P[camB], X[k])
                if r1 is None or r2 is None:
                    continue
                errs.append(np.linalg.norm(r1 - a[k, :2]))
                errs.append(np.linalg.norm(r2 - b[k, :2]))
            if not errs:
                continue
            score = np.mean(errs)
            pts = X[X[:, 3] > 0][:, :3]
            extent = float(np.ptp(pts, axis=0).max()) if len(pts) else 0
            if extent < 1.2 or extent > 3.0:
                score += 1e3
            if best is None or score < best[0]:
                conf = np.minimum(a[:, 2], b[:, 2])
                best = (score, X, extent, conf)
    if best is None:
        return None
    return best

def centroid_of(X):
    pts = X[X[:, 3] > 0][:, :3]
    return pts.mean(0) if len(pts) else np.zeros(3)

def main():
    names, P = load_cams()
    os.makedirs(k3dir, exist_ok=True)
    frames = sorted(int(os.path.basename(f)[:-5]) for f in glob.glob(os.path.join(annot_dir, names[0], "*.json")))
    summary = {"per_frame": [], "overall": {}}
    n2 = 0
    all_repro = []; all_extent = []
    prev_c = {"0": None, "1": None}
    init = False
    disp = []          # inter-frame centroid displacement per id (proves no jump)
    for fr in frames:
        # gather up to 2 candidate skeletons (one per cross-camera pair)
        cands = []
        for pair in (PAIR1, PAIR2):
            r = best_pair(*pair, P, fr)
            if r is None:
                continue
            score, X, extent, conf = r
            if score >= 30 or not (1.2 <= extent <= 3.0):
                continue
            kp3 = [[float(X[k, 0]), float(X[k, 1]), float(X[k, 2]), float(conf[k])] for k in range(25)]
            cands.append({"c": centroid_of(X), "kp3": kp3, "score": score, "extent": extent})
        # dedupe: two candidates within 5 m are the SAME person -> keep the better one
        if len(cands) == 2 and np.linalg.norm(cands[0]["c"] - cands[1]["c"]) < 5.0:
            cands = [cands[0]] if cands[0]["score"] <= cands[1]["score"] else [cands[1]]

        # ---- temporal ID assignment (the fix for skeleton jumping) ----
        assigned = {}
        if not init and len(cands) >= 1:
            # anchor: larger court-x = id "0" (Player1), smaller = id "1" (Player2)
            order = sorted(range(len(cands)), key=lambda i: cands[i]["c"][0], reverse=True)
            for rank, i in enumerate(order):
                assigned[str(rank)] = cands[i]
            init = True
        else:
            used = set()
            for pid in ("0", "1"):
                if prev_c[pid] is None:
                    continue
                best_i, best_d = None, 1e9
                for i, c in enumerate(cands):
                    if i in used:
                        continue
                    d = np.linalg.norm(c["c"] - prev_c[pid])
                    if d < best_d:
                        best_d, best_i = d, i
                if best_i is not None and best_d < 15.0:   # plausible motion step
                    assigned[pid] = cands[best_i]; used.add(best_i)
            free = [pid for pid in ("0", "1") if pid not in assigned]
            for i, c in enumerate(cands):
                if i not in used and free:
                    assigned[free.pop(0)] = c; used.add(i)

        players_out = [{"id": pid, "keypoints3d": assigned[pid]["kp3"]} for pid in sorted(assigned)]
        with open(os.path.join(k3dir, "%06d.json" % fr), "w") as f:
            json.dump(players_out, f)

        for pid in ("0", "1"):
            if pid in assigned:
                if prev_c[pid] is not None:
                    disp.append(float(np.linalg.norm(assigned[pid]["c"] - prev_c[pid])))
                prev_c[pid] = assigned[pid]["c"]
                all_repro.append(assigned[pid]["score"]); all_extent.append(assigned[pid]["extent"])
        n_valid = len(players_out)
        summary["per_frame"].append({"frame": fr, "n_valid": n_valid, "ids": sorted(assigned.keys())})
        if n_valid == 2:
            n2 += 1

    summary["overall"] = {
        "frames_total": len(frames),
        "frames_2players": n2,
        "frac_2players": n2 / len(frames),
        "repro_mean_px": float(np.mean(all_repro)) if all_repro else None,
        "repro_max_px": float(np.max(all_repro)) if all_repro else None,
        "extent_mean_m": float(np.mean(all_extent)) if all_extent else None,
        "extent_min_m": float(np.min(all_extent)) if all_extent else None,
        "extent_max_m": float(np.max(all_extent)) if all_extent else None,
        "id_jump_max_disp_m": float(np.max(disp)) if disp else None,
        "id_jump_mean_disp_m": float(np.mean(disp)) if disp else None,
    }
    with open(os.path.join(out_root, "quality_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("WROTE", len(frames), "frames ->", k3dir)
    print(json.dumps(summary["overall"], indent=2))

if __name__ == "__main__":
    main()
