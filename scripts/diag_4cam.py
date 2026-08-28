import sys, json, numpy as np, cv2

root = sys.argv[1]
control = sys.argv[2] if len(sys.argv) > 2 else None
cams = ["W_01", "W_02", "E_03", "E_04"]

def load_cam(base, cam):
    fs = cv2.FileStorage(f'{base}/intri.yml', cv2.FILE_STORAGE_READ)
    K = fs.getNode(f'K_{cam}').mat(); fs.release()
    fs = cv2.FileStorage(f'{base}/extri.yml', cv2.FILE_STORAGE_READ)
    R = fs.getNode(f'Rot_{cam}').mat(); T = fs.getNode(f'T_{cam}').mat().reshape(3); fs.release()
    return K, R, T

def Pmat(K, R, T):
    return K @ np.hstack([R, T.reshape(3, 1)])

def triang(uv1, uv2, P1, P2):
    A = np.vstack([uv1[0]*P1[2]-P1[0], uv1[1]*P1[2]-P1[1],
                   uv2[0]*P2[2]-P2[0], uv2[1]*P2[2]-P2[1]])
    _, _, Vt = np.linalg.svd(A); X = Vt[-1]
    return X[:3]/X[3]

def repro(X, P):
    xh = P @ np.append(X, 1.0)
    return xh[:2]/xh[2]

def run(base, clist, frames, tag):
    print(f"\n##### {tag}  root={base}")
    cd = {c: load_cam(base, c) for c in clist}
    Ps = {c: Pmat(*cd[c]) for c in clist}
    for a in clist:
        for b in clist:
            if a < b:
                print(f"  baseline {a}-{b}: {np.linalg.norm(cd[a][2]-cd[b][2]):.2f} m")
    for c1 in clist:
        for c2 in clist:
            if c1 < c2:
                rows = []
                for nf in frames:
                    try:
                        ann = [json.load(open(f'{base}/annots/{c}/{nf:06d}.json'))['annots']
                               for c in (c1, c2)]
                    except Exception as e:
                        rows.append(f"{c1}x{c2} f{nf}: NOFILE"); continue
                    best = None
                    for i, a in enumerate(ann[0]):
                        for j, b in enumerate(ann[1]):
                            ka = np.array(a['keypoints']); kb = np.array(b['keypoints'])
                            errs = []; Xs = []; e1 = []; e2 = []
                            for k in range(25):
                                if ka[k, 2] > 0.3 and kb[k, 2] > 0.3:
                                    X = triang(ka[k, :2], kb[k, :2], Ps[c1], Ps[c2])
                                    Xs.append(X)
                                    r1 = np.linalg.norm(repro(X, Ps[c1]) - ka[k, :2])
                                    r2 = np.linalg.norm(repro(X, Ps[c2]) - kb[k, :2])
                                    e1.append(r1); e2.append(r2); errs.append(r1+r2)
                            if len(Xs) >= 10:
                                e = np.mean(errs)
                                if best is None or e < best[0]:
                                    best = (e, np.array(Xs), i, j, np.mean(e1), np.mean(e2), ka, kb)
                if best is None:
                    rows.append(f"{c1}x{c2}: no valid pair"); continue
                e, Xs, i, j, m1, m2, ka, kb = best
                ext = np.array([Xs[:, 0].ptp(), Xs[:, 1].ptp(), Xs[:, 2].ptp()])
                c1c = np.median(ka[ka[:, 2] > 0.3, :2], axis=0) if (ka[:, 2] > 0.3).any() else [-1, -1]
                c2c = np.median(kb[kb[:, 2] > 0.3, :2], axis=0) if (kb[:, 2] > 0.3).any() else [-1, -1]
                rows.append(f"{c1}x{c2}: best {c1}#{i}x{c2}#{j} repro={e:5.1f}px "
                            f"(cam1={m1:4.1f}/cam2={m2:4.1f}) "
                            f"Z={ext[2]:.2f}m XY=({ext[0]:.2f},{ext[1]:.2f}) "
                            f"cen=({Xs[:,0].mean():5.1f},{Xs[:,1].mean():5.1f},{Xs[:,2].mean():.1f}) "
                            f"kptC1=({c1c[0]:.0f},{c1c[1]:.0f}) kptC2=({c2c[0]:.0f},{c2c[1]:.0f})")
                print("  " + "\n  ".join(rows))

frames = [50, 200, 400, 600, 800]
run(root, cams, frames, "02_24 4-cam")
if control:
    run(control, ["X1", "X2"], frames, "02_27 CONTROL (X1 x X2)")
