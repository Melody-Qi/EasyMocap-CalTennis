import sys, json, glob, os
import numpy as np
import cv2

root = sys.argv[1] if len(sys.argv) > 1 else "data/caltennis_0224_4cam"
img_dir = os.path.join(root, "images")
k3dir = os.path.join(root, "output", "mvmp_all", "keypoints3d")
out_dir = os.path.join(root, "output", "repro_vid")
intri = os.path.join(root, "intri.yml")
extri = os.path.join(root, "extri.yml")

# body25 skeleton edges (OpenPose-style)
EDGES = [(0,1),(1,2),(2,3),(3,4),(1,5),(5,6),(6,7),(1,8),(8,9),(9,10),
         (10,11),(8,12),(12,13),(13,14),(0,15),(15,17),(0,16),(16,18),(0,1)]
COLORS = [(0,255,0), (0,0,255)]  # player0 green, player1 red

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

def main():
    names, P = load_cams()
    frames = sorted(int(os.path.basename(f)[:-5]) for f in glob.glob(os.path.join(k3dir, "*.json")))
    for cam in names:
        od = os.path.join(out_dir, cam)
        os.makedirs(od, exist_ok=True)
        for fr in frames:
            imgp = os.path.join(img_dir, cam, "%06d.jpg" % fr)
            if not os.path.exists(imgp):
                continue
            img = cv2.imread(imgp)
            if img is None:
                continue
            persons = json.load(open(os.path.join(k3dir, "%06d.json" % fr)))
            for pl in persons:
                pid = int(pl["id"])
                k3 = np.array(pl["keypoints3d"])
                color = COLORS[pid % 2]
                pts2d = []
                for k in range(25):
                    X = np.array([k3[k, 0], k3[k, 1], k3[k, 2], 1.0])
                    if k3[k, 3] < 0.1:
                        pts2d.append(None); continue
                    p = P[cam] @ X
                    if p[2] <= 0:
                        pts2d.append(None); continue
                    pts2d.append((int(p[0]/p[2]), int(p[1]/p[2])))
                for (a, b) in EDGES:
                    if pts2d[a] is None or pts2d[b] is None:
                        continue
                    cv2.line(img, pts2d[a], pts2d[b], color, 2)
                for p2 in pts2d:
                    if p2 is None:
                        continue
                    cv2.circle(img, p2, 3, color, -1)
            cv2.imwrite(os.path.join(od, "%06d.png" % fr), img)
        print("cam", cam, "done", len(frames), "frames")
    print("REPRO VID DONE ->", out_dir)

if __name__ == "__main__":
    main()
