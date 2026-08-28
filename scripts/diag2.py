import sys, json, numpy as np, cv2
root = sys.argv[1]
cams = ["W_01", "W_02", "E_03", "E_04"]
frames = [int(x) for x in sys.argv[2:]] if len(sys.argv) > 2 else [50, 200, 400, 600, 800]

def load_cam(base, cam):
    fs = cv2.FileStorage(f'{base}/intri.yml', cv2.FILE_STORAGE_READ)
    K = fs.getNode(f'K_{cam}').mat(); fs.release()
    fs = cv2.FileStorage(f'{base}/extri.yml', cv2.FILE_STORAGE_READ)
    R = fs.getNode(f'Rot_{cam}').mat(); T = fs.getNode(f'T_{cam}').mat().reshape(3); fs.release()
    return K, R, T

def Pmat(K, R, T): return K @ np.hstack([R, T.reshape(3, 1)])
def triang(uv1, uv2, P1, P2):
    A = np.vstack([uv1[0]*P1[2]-P1[0], uv1[1]*P1[2]-P1[1], uv2[0]*P2[2]-P2[0], uv2[1]*P2[2]-P2[1]])
    _, _, Vt = np.linalg.svd(A); X = Vt[-1]; return X[:3]/X[3]
def repro(X, P):
    xh = P @ np.append(X, 1.0); return xh[:2]/xh[2]

cd = {c: load_cam(root, c) for c in cams}
Ps = {c: Pmat(*cd[c]) for c in cams}

def good_pairs(nf, thr=10.0):
    ann = {}
    for c in cams:
        try: ann[c] = json.load(open(f'{root}/annots/{c}/{nf:06d}.json'))['annots']
        except: ann[c] = []
    edges = []
    for a in range(len(cams)):
        for b in range(a+1, len(cams)):
            c1, c2 = cams[a], cams[b]
            for i, pa in enumerate(ann[c1]):
                for j, pb in enumerate(ann[c2]):
                    ka = np.array(pa['keypoints']); kb = np.array(pb['keypoints'])
                    Xs = []; e1 = []; e2 = []
                    for k in range(25):
                        if ka[k, 2] > 0.3 and kb[k, 2] > 0.3:
                            X = triang(ka[k, :2], kb[k, :2], Ps[c1], Ps[c2])
                            Xs.append(X)
                            e1.append(np.linalg.norm(repro(X, Ps[c1]) - ka[k, :2]))
                            e2.append(np.linalg.norm(repro(X, Ps[c2]) - kb[k, :2]))
                    if len(Xs) >= 10:
                        m1, m2 = np.mean(e1), np.mean(e2)
                        if m1 < thr and m2 < thr:
                            X = np.mean(Xs, axis=0)
                            if -5 < X[0] < 35 and -15 < X[1] < 25 and -1 < X[2] < 3:
                                edges.append((c1, i, c2, j, m1+m2, X))
    return ann, edges

for nf in frames:
    ann, edges = good_pairs(nf)
    print(f"\n=== frame {nf} ===")
    # union-find over (cam,idx)
    parent = {}
    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb: parent[ra] = rb
    nodes = set()
    for c in cams:
        for i in range(len(ann[c])):
            nodes.add((c, i))
    for (c1, i, c2, j, e, X) in edges:
        union((c1, i), (c2, j))
    comps = {}
    for n in nodes:
        comps.setdefault(find(n), []).append(n)
    players = [v for v in comps.values() if len(v) >= 2]
    print(f"  detections per cam: " + ", ".join(f"{c}={len(ann[c])}" for c in cams))
    print(f"  good cross-cam pairs (repro<10px): {len(edges)}")
    for (c1, i, c2, j, e, X) in sorted(edges, key=lambda t: t[4]):
        print(f"    {c1}#{i}--{c2}#{j} err={e:4.1f}px cen=({X[0]:5.1f},{X[1]:5.1f},{X[2]:.1f})")
    print(f"  ==> {len(players)} player(s) reconstructed:")
    for p in players:
        # average centroid
        cens = [e[5] for e in edges if (e[0], e[1]) in [(x[0], x[1]) for x in p] or (e[2], e[3]) in [(x[0], x[1]) for x in p]]
        camset = sorted(set(c for (c, i) in p))
        print(f"    player seen by {camset}: members={p}")
