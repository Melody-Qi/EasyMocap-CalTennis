import sys, json, glob, os
import numpy as np
import cv2

root = sys.argv[1] if len(sys.argv) > 1 else "data/caltennis_0224_4cam"
out_root = os.path.join(root, "output", "mvmp_all")
k3dir = os.path.join(out_root, "keypoints3d")
annot_dir = os.path.join(root, "annots")
intri = os.path.join(root, "intri.yml")
extri = os.path.join(root, "extri.yml")

# Verified: W_01 & E_03 see the HIGH-x player (~x24); W_02 & E_04 see the LOW-x player (~x0).
CAMS_HIGH = ("W_01", "E_03")   # player seen at large court-x
CAMS_LOW  = ("W_02", "E_04")   # player seen at small court-x
COURT = dict(x=(-4, 28), y=(-4, 16), z=(0.0, 2.6))

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

def det_center_scale(det):
    vis = det[:, 2] > 0.1
    if vis.sum() < 3:
        return None, None
    pts = det[vis][:, :2]
    return pts.mean(0), pts.ptp(0).max()

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
    detsA = load_annots(camA, frame); detsB = load_annots(camB, frame)
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
                errs.append(np.linalg.norm(r1 - a[k, :2])); errs.append(np.linalg.norm(r2 - b[k, :2]))
            if not errs:
                continue
            score = np.mean(errs)
            pts = X[X[:, 3] > 0][:, :3]
            extent = float(np.ptp(pts, axis=0).max()) if len(pts) else 0
            if extent < 1.0 or extent > 3.0:
                score += 1e3
            if best is None or score < best[0]:
                best = (score, X, extent)
    return best

def on_court_pt(xyz):
    x, y, z = xyz[0], xyz[1], xyz[2]
    return (COURT["x"][0] <= x <= COURT["x"][1] and COURT["y"][0] <= y <= COURT["y"][1]
            and COURT["z"][0] <= z <= COURT["z"][1])

def centroid_of(X):
    pts = X[X[:, 3] > 0][:, :3]
    return pts.mean(0) if len(pts) else None

def project(P, xyz):
    x = P @ np.array([xyz[0], xyz[1], xyz[2], 1.0])
    if x[2] == 0:
        return None
    return np.array([x[0] / x[2], x[1] / x[2]])

def select_det(cam, fr_idx, home, P, dets_cache, radius=1e9):
    """Pick the detection in `cam` at frame `fr_idx` nearest to the projection of `home`.
    radius is intentionally huge: each camera sees essentially one player, so the
    nearest-to-home detection is the right one even as the player moves across the court;
    the other player (if visible) projects far away and is never selected."""
    proj = project(P[cam], home)
    if proj is None:
        return None
    dets = dets_cache[cam][fr_idx]
    best, best_d = None, 1e9
    for d in dets:
        c, s = det_center_scale(d)
        if c is None:
            continue
        dd = np.linalg.norm(c - proj)
        if dd < best_d:
            best_d, best = dd, d
    if best is not None and best_d < radius:
        return best
    return None

def main():
    names, P = load_cams()
    frames = sorted(int(os.path.basename(f)[:-5]) for f in glob.glob(os.path.join(annot_dir, names[0], "*.json")))
    dets_cache = {cam: [load_annots(cam, fr) for fr in frames] for cam in names}

    # ---- bootstrap: 2-means on on-court centroid candidates -> two player homes ----
    cand = []
    for fr in frames:
        for (ca, cb) in (CAMS_HIGH, CAMS_LOW):
            r = best_pair(ca, cb, P, fr)
            if r is None:
                continue
            score, X, extent = r
            if score >= 30 or not (1.0 <= extent <= 3.0):
                continue
            c = centroid_of(X)
            if c is not None and on_court_pt(c):
                cand.append(c)
    cand = np.array(cand)
    if len(cand) >= 2:
        a = cand[cand[:, 0].argmin()].copy(); b = cand[cand[:, 0].argmax()].copy()
        for _ in range(25):
            da = np.linalg.norm(cand - a, axis=1); db = np.linalg.norm(cand - b, axis=1)
            na, nb = cand[da < db], cand[db <= da]
            if len(na) == 0 or len(nb) == 0:
                break
            m1, m2 = na.mean(0), nb.mean(0)
            if np.allclose(m1, a) and np.allclose(m2, b):
                a, b = m1, m2; break
            a, b = m1, m2
        # high-x home -> CAMS_HIGH player; low-x home -> CAMS_LOW player
        if a[0] > b[0]:
            home_high, home_low = a, b
        else:
            home_high, home_low = b, a
    else:
        home_high = np.array([23.0, 5.0, 1.0]); home_low = np.array([2.0, 5.0, 1.0])
    print("HOME_HIGH(x~24):", home_high, " HOME_LOW(x~0):", home_low)

    os.makedirs(k3dir, exist_ok=True)
    summary = {"per_frame": [], "overall": {}}
    n2 = 0; all_repro = []; all_extent = []; disp = []
    # Identity is FIXED by which camera-pair sees the player (HIGH->"0", LOW->"1"),
    # so no cross-frame swapping is possible. The selection anchor follows the
    # player's own previous 3D position (projected), so it tracks the player as
    # they move across the court instead of snapping to a fixed point.
    last3d = {"0": home_high.copy(), "1": home_low.copy()}
    prev_c = {"0": None, "1": None}

    def try_player(pid, pair_cams, anchor):
        d = [select_det(c, idx, anchor, P, dets_cache) for c in pair_cams]
        if not all(x is not None for x in d):
            return None
        X = triang(P[pair_cams[0]], P[pair_cams[1]], d[0], d[1])
        pts = X[X[:, 3] > 0]
        if len(pts) < 10:
            return None
        extent = float(np.ptp(pts[:, :3], axis=0).max())
        c = centroid_of(X)
        if not (1.0 <= extent <= 3.0) or c is None or not on_court_pt(c):
            return None
        errs = []
        for k in range(25):
            if X[k, 3] == 0:
                continue
            r1 = repro(P[pair_cams[0]], X[k]); r2 = repro(P[pair_cams[1]], X[k])
            if r1 is None or r2 is None:
                continue
            errs.append(np.linalg.norm(r1 - d[0][k, :2])); errs.append(np.linalg.norm(r2 - d[1][k, :2]))
        score = float(np.mean(errs)) if errs else 99
        if score >= 30:
            return None
        kp3 = [[float(X[k, 0]), float(X[k, 1]), float(X[k, 2]), float(min(d[0][k, 2], d[1][k, 2]))] for k in range(25)]
        return {"pid": pid, "c": c, "kp3": kp3, "score": score, "extent": extent}

    for idx, fr in enumerate(frames):
        assigned = {}
        for pid, pair_cams in (("0", CAMS_HIGH), ("1", CAMS_LOW)):
            res = try_player(pid, pair_cams, last3d[pid])
            if res is not None:
                assigned[pid] = res
                last3d[pid] = res["c"]   # follow the player's motion next frame
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
        "frames_total": len(frames), "frames_2players": n2, "frac_2players": n2 / len(frames),
        "repro_mean_px": float(np.mean(all_repro)) if all_repro else None,
        "repro_max_px": float(np.max(all_repro)) if all_repro else None,
        "extent_mean_m": float(np.mean(all_extent)) if all_extent else None,
        "extent_min_m": float(np.min(all_extent)) if all_extent else None,
        "extent_max_m": float(np.max(all_extent)) if all_extent else None,
        "id_jump_max_disp_m": float(np.max(disp)) if disp else None,
        "id_jump_mean_disp_m": float(np.mean(disp)) if disp else None,
        "homes": {"high": home_high.tolist(), "low": home_low.tolist()},
    }
    with open(os.path.join(out_root, "quality_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("WROTE", len(frames), "frames ->", k3dir)
    print(json.dumps(summary["overall"], indent=2))

if __name__ == "__main__":
    main()
