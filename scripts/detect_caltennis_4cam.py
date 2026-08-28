"""Multi-person 2D detection for CalTennis 4-cam clip, reusing GVHMR's proven
ViTPose-H (vitpose-h-multi-coco.pth) + YOLOv8x, writing EasyMocap annots.

For each camera, for each frame:
  1. YOLOv8 detects ALL persons -> xyxy boxes (original image coords)
  2. For each person: crop_and_resize -> 256x256, take 32:224 width -> 256x192
  3. ViTPose-H infers COCO-17 heatmaps; mmpose keypoints_from_heatmaps (UDP)
     maps back to original image coordinates
  4. COCO-17 -> body25; write annots/<cam>/%06d.json (multi-person)

Run with the gvhmr conda env and PYTHONPATH containing GVHMR and EasyMocap roots.
"""
import argparse
import json
import os
from os.path import join

import cv2
import numpy as np
import torch
from tqdm import tqdm
from ultralytics import YOLO

from hmr4d.network.hmr2.utils.preproc import crop_and_resize, IMAGE_MEAN, IMAGE_STD
from hmr4d.utils.preproc.vitpose_pytorch import build_model
from hmr4d.utils.preproc.vitpose import flip_heatmap_coco17
from hmr4d.utils.kpts.kp2d_utils import keypoints_from_heatmaps
from easymocap.dataset.config import coco17tobody25

IMG_DS = 0.5
IMG_DST = 256
HEIGHT, WIDTH = 1080, 1920
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def detect_cam(root, cam, yolo, pose, limit=None, batch_size=8):
    img_dir = join(root, "images", cam)
    out_dir = join(root, "annots", cam)
    os.makedirs(out_dir, exist_ok=True)
    names = sorted(os.listdir(img_dir))
    if limit is not None:
        names = names[:limit]
    n = len(names)

    # per-frame person count summary
    counts = []

    for fi, name in enumerate(tqdm(names, desc=f"detect {cam}")):
        img = cv2.imread(join(img_dir, name))[..., ::-1]  # RGB
        if img is None:
            counts.append(0)
            continue
        # 1) YOLOv8 person detection (class 0 = person)
        res = yolo(img, classes=[0], conf=0.3, verbose=False)[0]
        boxes = res.boxes.xyxy.cpu().numpy()  # (P,4) xyxy
        confs = res.boxes.conf.cpu().numpy()  # (P,)
        # keep reasonable sizes (avoid far/edge false positives)
        wh = boxes[:, 2:] - boxes[:, :2]
        aspect = wh[:, 1] / np.maximum(wh[:, 0], 1.0)
        area = wh[:, 0] * wh[:, 1]
        # keep standing persons: tall boxes, reasonably confident. Drops courtside
        # near-square / low-conf false positives (rackets, chairs, ball kids).
        keep = (wh[:, 0] > 20) & (wh[:, 1] > 40) & (area > 1500) & (aspect > 1.4) & (confs > 0.45)
        boxes, confs = boxes[keep], confs[keep]
        P = len(boxes)
        counts.append(P)

        if P == 0:
            payload = {"filename": f"images/{cam}/{name}", "height": HEIGHT, "width": WIDTH, "annots": []}
            open(join(out_dir, name.replace(".jpg", ".json")), "w").write(json.dumps(payload))
            continue

        # 2) build crops (downscaled) for every detected person
        img_ds = cv2.resize(img, (0, 0), fx=IMG_DS, fy=IMG_DS)
        crops, centers, scales = [], [], []
        for (x1, y1, x2, y2) in boxes:
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            size = float(max(x2 - x1, y2 - y1))
            crop, _ = crop_and_resize(img_ds, np.array([cx * IMG_DS, cy * IMG_DS]),
                                      np.array(size * IMG_DS), IMG_DST, enlarge_ratio=1.0)
            crop192 = crop[:, 32:224]  # 256x192
            crop_t = torch.from_numpy(crop192).permute(2, 0, 1).float() / 255.0
            crop_t = (crop_t - IMAGE_MEAN.reshape(3, 1, 1)) / IMAGE_STD.reshape(3, 1, 1)
            crops.append(crop_t)
            centers.append([cx, cy])           # original coords
            scales.append([size * 24.0 / 32.0, size])  # original coords

        crops_t = torch.stack(crops).to(DEVICE)
        # 3) ViTPose-H inference (+ horizontal flip test)
        # ViTPose-H inference + horizontal flip test (matches GVHMR VitPoseExtractor)
        hm_cat = pose(torch.cat([crops_t, crops_t.flip(3)], dim=0))
        hm_ori, hm_flip = hm_cat.chunk(2)
        hm = (hm_ori + flip_heatmap_coco17(hm_flip)) * 0.5
        preds, maxvals = keypoints_from_heatmaps(
            hm.detach().cpu().numpy(),
            center=np.array(centers, dtype=np.float32),
            scale=np.array(scales, dtype=np.float32) / 200.0,
            use_udp=True,
        )
        kp2d = np.concatenate([preds, maxvals], axis=-1)  # (P,17,3)
        body25 = coco17tobody25(kp2d)  # (P,25,3)

        annots = []
        for p in range(P):
            annots.append({
                "personID": int(p),
                "bbox": [float(v) for v in [boxes[p][0], boxes[p][1], boxes[p][2], boxes[p][3], confs[p]]],
                "keypoints": body25[p].tolist(),
                "area": 0.0,
            })
        payload = {"filename": f"images/{cam}/{name}", "height": HEIGHT, "width": WIDTH, "annots": annots}
        open(join(out_dir, name.replace(".jpg", ".json")), "w").write(json.dumps(payload))

    summary = {"cam": cam, "n_frames": n, "person_counts": counts,
               "mean_persons": float(np.mean(counts)) if counts else 0.0}
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="data dir with images/<cam>/ and calib_json/")
    ap.add_argument("--cameras", nargs="+", required=True)
    ap.add_argument("--limit", type=int, default=None, help="process only first N frames (test)")
    ap.add_argument("--vitpose", default="inputs/checkpoints/vitpose/vitpose-h-multi-coco.pth")
    ap.add_argument("--yolo", default="inputs/checkpoints/yolo/yolov8x.pt")
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()

    print(f"[detect] loading ViTPose-H from {args.vitpose}")
    pose = build_model("ViTPose_huge_coco_256x192", args.vitpose).to(DEVICE).eval()
    print(f"[detect] loading YOLOv8 from {args.yolo}")
    yolo = YOLO(args.yolo)

    summaries = []
    for cam in args.cameras:
        s = detect_cam(args.root, cam, yolo, pose, limit=args.limit, batch_size=args.batch)
        summaries.append(s)
        print(f"[detect] {cam}: mean persons/frame = {s['mean_persons']:.2f} over {s['n_frames']} frames")
    with open(join(args.root, "detect_summary.json"), "w") as f:
        json.dump(summaries, f, indent=2)
    print("[detect] wrote detect_summary.json")


if __name__ == "__main__":
    main()
