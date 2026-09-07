"""GVHMR pilot 结果量化评估：这批三机位网球视频到底能不能做 mocap。

对每路输出检查三组指标：
  1. 检测与 2D 关键点质量（人有没有检到、ViTPose 置信度、人在画面里多大）
  2. SMPL 结果质量（世界坐标轨迹、速度、抖动、静态接触置信度）
  3. 跨视角同步性诊断（人体速度曲线互相关仅作线索，峰值不突出时拒绝下结论）

用法:
    python eval_gvhmr_pilot.py --root <GVHMR/outputs> \
        --cams tennis_camera_1 tennis_camera_2 tennis_camera_3 \
        --prefix baidu_pilot_15_30s_ --fps 30 --out work/pilot_eval.json
"""

import argparse
import json
import os

import numpy as np
import torch


def load_cam(root, prefix, cam):
    d = os.path.join(root, f"{prefix}{cam}", cam)
    res = torch.load(os.path.join(d, "hmr4d_results.pt"), map_location="cpu")
    vitpose = torch.load(os.path.join(d, "preprocess", "vitpose.pt"), map_location="cpu")
    bbx = torch.load(os.path.join(d, "preprocess", "bbx.pt"), map_location="cpu")
    return res, vitpose, bbx


def as_np(x):
    return x.detach().cpu().numpy().astype(np.float64)


def zscore(x):
    x = np.asarray(x, dtype=np.float64)
    s = x.std()
    return np.zeros_like(x) if s < 1e-12 else (x - x.mean()) / s


def best_lag(a, b, max_lag):
    a, b = zscore(a), zscore(b)
    n = min(len(a), len(b))
    out = []
    for lag in range(-max_lag, max_lag + 1):
        xa, xb = (a[: n - lag], b[lag:n]) if lag >= 0 else (a[-lag:n], b[: n + lag])
        if len(xa) < 30:
            continue
        out.append((lag, float(np.dot(xa, xb) / len(xa))))
    out.sort(key=lambda t: -t[1])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/public/home/CS286/qiyt2023-CS286/GVHMR/outputs")
    ap.add_argument("--prefix", default="baidu_pilot_15_30s_")
    ap.add_argument("--cams", nargs="+", default=["tennis_camera_1", "tennis_camera_2", "tennis_camera_3"])
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--max_lag", type=int, default=30)
    ap.add_argument("--out", default=None, help="评估 JSON 输出路径")
    args = ap.parse_args()

    report = {}
    speeds = {}

    for cam in args.cams:
        res, vitpose, bbx = load_cam(args.root, args.prefix, cam)

        bbx_xys = as_np(bbx["bbx_xys"])      # (F,3) cx, cy, scale
        bbx_xyxy = as_np(bbx["bbx_xyxy"])    # (F,4)
        vp = as_np(vitpose)                  # (F,17,3) x, y, conf
        trans_g = as_np(res["smpl_params_global"]["transl"])   # (F,3) 世界坐标
        trans_i = as_np(res["smpl_params_incam"]["transl"])    # (F,3) 相机坐标
        K = as_np(res["K_fullimg"])

        F = len(bbx_xys)
        scale = bbx_xys[:, 2]
        detected = scale > 1e-6
        bbox_h = (bbx_xyxy[:, 3] - bbx_xyxy[:, 1])[detected]

        conf = vp[:, :, 2]
        conf_det = conf[detected]
        low_conf_frames = detected.copy()
        # 一帧被认为"关键点可用"：至少 10 个点置信度 > 0.3
        good_kp = (conf > 0.3).sum(axis=1) >= 10
        kp_ok = (detected & good_kp).sum()

        vel = np.linalg.norm(np.diff(trans_g, axis=0), axis=1) * args.fps  # m/s
        acc = np.abs(np.diff(vel)) * args.fps                               # m/s^2，抖动代理
        dist_to_cam = trans_i[:, 2]

        static_conf = None
        if "static_conf_logits" in res.get("net_outputs", {}):
            sc = as_np(res["net_outputs"]["static_conf_logits"])
            static_conf = float(torch.sigmoid(torch.from_numpy(sc)).mean()) if sc.ndim == 1 else float(
                torch.sigmoid(torch.from_numpy(sc)).numpy().mean())

        speeds[cam] = vel
        report[cam] = {
            "frames": int(F),
            "detected_frames": int(detected.sum()),
            "detected_ratio": round(float(detected.mean()), 4),
            "bbox_height_px_mean": round(float(bbox_h.mean()), 1) if len(bbox_h) else None,
            "bbox_height_px_min": round(float(bbox_h.min()), 1) if len(bbox_h) else None,
            "bbox_height_px_max": round(float(bbox_h.max()), 1) if len(bbox_h) else None,
            "vitpose_conf_mean": round(float(conf_det.mean()), 3) if conf_det.size else None,
            "kp_ok_frames": int(kp_ok),
            "kp_ok_ratio": round(float(kp_ok / F), 4),
            "dist_to_cam_m_mean": round(float(dist_to_cam.mean()), 2),
            "dist_to_cam_m_range": [round(float(dist_to_cam.min()), 2), round(float(dist_to_cam.max()), 2)],
            "world_path_len_m": round(float(np.linalg.norm(np.diff(trans_g, axis=0), axis=1).sum()), 2),
            "speed_mps_mean": round(float(vel.mean()), 3),
            "speed_mps_p95": round(float(np.percentile(vel, 95)), 3),
            "speed_mps_max": round(float(vel.max()), 3),
            "jitter_mps2_mean": round(float(acc.mean()), 2),
            "jitter_mps2_p95": round(float(np.percentile(acc, 95)), 2),
            "static_conf_mean": None if static_conf is None else round(static_conf, 3),
            "K_est_fx": round(float(K[0, 0, 0]), 1),
            "K_est_fy": round(float(K[0, 1, 1]), 1),
            "K_est_cx": round(float(K[0, 0, 2]), 1),
            "K_est_cy": round(float(K[0, 1, 2]), 1),
        }

    # ---- 用人体速度曲线做跨视角同步性检查（信号比整帧差分强得多）----
    sync = {}
    ref = args.cams[0]
    for cam in args.cams[1:]:
        res = best_lag(speeds[ref], speeds[cam], args.max_lag)
        top = res[0]
        r0 = [r for l, r in res if l == 0][0]
        spread = top[1] - np.mean([r for _, r in res[:5]])
        sync[f"{ref}_vs_{cam}"] = {
            "best_lag": top[0],
            "best_r": round(top[1], 3),
            "r_at_zero": round(r0, 3),
            "peak_prominence": round(float(top[1] - r0), 3),
            "top5_flatness": round(float(spread), 4),
            "reliable": bool(top[1] - r0 > 0.05 and abs(top[0]) < args.max_lag),
        }

    print("=" * 78)
    frame_counts = sorted({r["frames"] for r in report.values()})
    if len(frame_counts) == 1:
        frames_text = f"每路 {frame_counts[0] / args.fps:.1f} 秒 / {frame_counts[0]} 帧"
    else:
        frames_text = "各路帧数不一致: " + ", ".join(map(str, frame_counts))
    print(f"单目 GVHMR 质量（{frames_text}）")
    print("=" * 78)
    hdr = f"{'camera':<18}{'检出率':>8}{'bbox高(px)':>12}{'ViTPose置信':>12}{'关键点可用':>11}{'速度m/s':>9}{'抖动m/s²':>10}{'距相机m':>9}"
    print(hdr)
    for cam in args.cams:
        r = report[cam]
        print(f"{cam:<18}{r['detected_ratio']:>8.3f}"
              f"{str(r['bbox_height_px_mean']):>12}"
              f"{str(r['vitpose_conf_mean']):>12}"
              f"{r['kp_ok_ratio']:>11.3f}"
              f"{r['speed_mps_mean']:>9.2f}"
              f"{r['jitter_mps2_mean']:>10.1f}"
              f"{r['dist_to_cam_m_mean']:>9.1f}")

    print()
    for cam in args.cams:
        r = report[cam]
        print(f"[{cam}] 世界轨迹总长 {r['world_path_len_m']} m, "
              f"速度 p95={r['speed_mps_p95']} m/s, 抖动 p95={r['jitter_mps2_p95']} m/s², "
              f"静态接触置信={r['static_conf_mean']}")
        print(f"         GVHMR 自估内参 fx={r['K_est_fx']} fy={r['K_est_fy']} "
              f"cx={r['K_est_cx']} cy={r['K_est_cy']}")

    print()
    print("=" * 78)
    print("跨视角同步性（基于人体速度曲线互相关）")
    print("=" * 78)
    for k, v in sync.items():
        flag = "可用" if v["reliable"] else "信号不足，不可判定"
        print(f"{k}: 最佳偏移 {v['best_lag']:+d} 帧 (r={v['best_r']}), "
              f"零偏移 r={v['r_at_zero']}, 峰值突出度={v['peak_prominence']} → {flag}")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"per_camera": report, "sync": sync}, f, indent=2, ensure_ascii=False)
        print(f"\n评估已保存: {args.out}")


if __name__ == "__main__":
    main()
