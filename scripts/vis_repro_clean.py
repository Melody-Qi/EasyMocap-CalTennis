import sys, json, glob, os
import numpy as np
import cv2

root = sys.argv[1] if len(sys.argv) > 1 else "data/caltennis_0224_4cam"
img_dir = os.path.join(root, "images")
k3dir = os.path.join(root, "output", "mvmp_all", "keypoints3d")
out_dir = os.path.join(root, "output", "repro_clean")
intri = os.path.join(root, "intri.yml")
extri = os.path.join(root, "extri.yml")

EDGES = [(0,1),(1,2),(2,3),(3,4),(1,5),(5,6),(6,7),(1,8),(8,9),(9,10),
         (10,11),(8,12),(12,13),(13,14),(0,15),(15,17),(0,16),(16,18),(0,1)]
CAM_PLAYER = {"W_01": 0, "E_03": 0, "W_02": 1, "E_04": 1}  # each cam draws only its player
COLORS = {0: (0,255,0), 1: (0,0,255)}   # P0 green, P1 red
LABELS = {0: "P0 (HIGH-side)", 1: "P1 (LOW-side)"}

def load_cams():
    fs = cv2.FileStorage(intri, cv2.FILE_STORAGE_READ)
    names = [fs.getNode("names").at(i).string() for i in range(fs.getNode("names").size())]
    K = {n: fs.getNode("K_" + n).mat() for n in names}; fs.release()
    fs = cv2.FileStorage(extri, cv2.FILE_STORAGE_READ)
    Rv = {n: fs.getNode("R_" + n).mat().ravel() for n in names}
    T = {n: fs.getNode("T_" + n).mat().ravel() for n in names}; fs.release()
    P = {}
    for n in names:
        R, _ = cv2.Rodrigues(Rv[n]); Rt = np.hstack([R, T[n].reshape(3,1)]); P[n] = K[n] @ Rt
    return names, P

def smooth_k3(arr):
    # arr: [N,25,3] with NaN for missing; remove spikes per joint
    N, J, _ = arr.shape
    out = arr.copy()
    for j in range(J):
        for c in range(3):
            v = arr[:, j, c]
            valid = ~np.isnan(v)
            if valid.sum() < 3:
                continue
            # linear interpolation across gaps first
            idx = np.where(valid)[0]
            f = interp1d_safe(idx, v[idx])
            for t in range(N):
                if not valid[t]:
                    out[t, j, c] = f(t)
            # spike removal on the now-filled series
            filled = out[:, j, c].copy()
            for t in range(1, N-1):
                if abs(filled[t] - 0.5*(filled[t-1]+filled[t+1])) > 0.7:
                    out[t, j, c] = 0.5*(filled[t-1]+filled[t+1])
    return out

def interp1d_safe(x, y):
    # simple linear interpolation callable
    def f(t):
        if t <= x[0]: return y[0]
        if t >= x[-1]: return y[-1]
        i = np.searchsorted(x, t) - 1
        x0, x1, y0, y1 = x[i], x[i+1], y[i], y[i+1]
        return y0 + (y1-y0)*(t-x0)/(x1-x0)
    return f

def main():
    names, P = load_cams()
    frames = sorted(int(os.path.basename(f)[:-5]) for f in glob.glob(os.path.join(k3dir, "*.json")))
    # load all into arrays per player
    players = {0: np.full((len(frames),25,3), np.nan), 1: np.full((len(frames),25,3), np.nan)}
    for fi, fr in enumerate(frames):
        for pl in json.load(open(os.path.join(k3dir, "%06d.json" % fr))):
            pid = int(pl["id"])
            if pid in players:
                players[pid][fi] = np.array(pl["keypoints3d"])[:, :3]
    for pid in players:
        players[pid] = smooth_k3(players[pid])
    for cam in names:
        pid = CAM_PLAYER.get(cam, 0)
        od = os.path.join(out_dir, cam); os.makedirs(od, exist_ok=True)
        color = COLORS[pid]; label = LABELS[pid]
        for fi, fr in enumerate(frames):
            imgp = os.path.join(img_dir, cam, "%06d.jpg" % fr)
            if not os.path.exists(imgp):
                continue
            img = cv2.imread(imgp)
            if img is None:
                continue
            k3 = players[pid][fi]
            if np.isnan(k3).all():
                cv2.imwrite(os.path.join(od, "%06d.png" % fr), img); continue
            pts2d = []
            for k in range(25):
                if np.isnan(k3[k]).any():
                    pts2d.append(None); continue
                X = np.array([k3[k,0], k3[k,1], k3[k,2], 1.0])
                p = P[cam] @ X
                if p[2] <= 0:
                    pts2d.append(None); continue
                pts2d.append((int(p[0]/p[2]), int(p[1]/p[2])))
            for (a,b) in EDGES:
                if pts2d[a] is None or pts2d[b] is None: continue
                cv2.line(img, pts2d[a], pts2d[b], color, 2)
            for p2 in pts2d:
                if p2 is None: continue
                cv2.circle(img, p2, 3, color, -1)
            cv2.putText(img, label, (30,40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
            cv2.imwrite(os.path.join(od, "%06d.png" % fr), img)
        print("cam", cam, "done")
    print("REPRO_CLEAN_DONE ->", out_dir)

if __name__ == "__main__":
    main()
