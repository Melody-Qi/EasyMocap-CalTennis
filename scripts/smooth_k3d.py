import sys, json, glob, os
import numpy as np

k3dir = sys.argv[1]
files = sorted(glob.glob(os.path.join(k3dir, "*.json")))
N = len(files)

def interp1d_safe(x, y):
    def f(t):
        if t <= x[0]: return y[0]
        if t >= x[-1]: return y[-1]
        i = np.searchsorted(x, t) - 1
        x0, x1, y0, y1 = x[i], x[i+1], y[i], y[i+1]
        return y0 + (y1-y0)*(t-x0)/(x1-x0)
    return f

def smooth_xyz(arr):  # arr [N,25,3] with NaN for missing
    Nn, J, _ = arr.shape
    out = arr.copy()
    for j in range(J):
        for c in range(3):
            v = arr[:, j, c]; valid = ~np.isnan(v)
            if valid.sum() < 3:
                continue
            idx = np.where(valid)[0]; f = interp1d_safe(idx, v[idx])
            for t in range(Nn):
                if not valid[t]:
                    out[t, j, c] = f(t)
            filled = out[:, j, c].copy()
            for t in range(1, Nn-1):
                if abs(filled[t] - 0.5*(filled[t-1]+filled[t+1])) > 0.7:
                    out[t, j, c] = 0.5*(filled[t-1]+filled[t+1])
    return out

full = {0: np.full((N,25,3), np.nan), 1: np.full((N,25,3), np.nan)}
order = []
for i, f in enumerate(files):
    d = json.load(open(f)); order.append(d)
    for pl in d:
        pid = int(pl["id"])
        k3 = np.array(pl["keypoints3d"])
        full[pid][i] = k3[:, :3]
for pid in full:
    full[pid] = smooth_xyz(full[pid])
for i, f in enumerate(files):
    d = order[i]
    for pl in d:
        pid = int(pl["id"])
        k3 = np.array(pl["keypoints3d"])
        k3[:, :3] = full[pid][i]      # overwrite xyz, keep confidence col
        pl["keypoints3d"] = k3.tolist()
    json.dump(d, open(f, "w"))
print("SMOOTHED", N, "frames")
