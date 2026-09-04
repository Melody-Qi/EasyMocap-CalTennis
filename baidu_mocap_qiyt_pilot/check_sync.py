"""三路视频时间同步性检查（不依赖相机标定）。

原理：三台相机拍同一场比赛，虽然视角不同、画面内容差异很大，但"画面运动强度"
的时间曲线应当高度一致（运动员跑动、挥拍、停顿都发生在相同时刻）。
对每路计算帧间差分能量序列，再做互相关，峰值位置即为相对帧偏移。

用法:
    python check_sync.py <images_root> <cam1> <cam2> <cam3> [--max_lag 30] [--out png路径]

例如:
    python check_sync.py work/images tennis_camera_1 tennis_camera_2 tennis_camera_3
"""

import argparse
import os
import numpy as np
from PIL import Image


def motion_energy(frames_dir, small=(128, 72), limit=None, crop=None):
    """读一个目录下的有序 JPEG，计算帧间平均绝对差（运动强度）序列。

    crop=(l, t, r, b) 为归一化裁剪范围，用于只保留球场中心区域，
    避免记分牌、观众等无关运动干扰。
    """
    files = sorted(f for f in os.listdir(frames_dir) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    if limit:
        files = files[:limit]
    if len(files) < 2:
        raise RuntimeError(f"{frames_dir} 帧数不足: {len(files)}")

    W, H = small
    box = None
    if crop:
        box = (int(crop[0] * W), int(crop[1] * H), int(crop[2] * W), int(crop[3] * H))

    prev = None
    energy = []
    for fn in files:
        im = Image.open(os.path.join(frames_dir, fn)).convert("L").resize(small, Image.BILINEAR)
        if box:
            im = im.crop(box)
        cur = np.asarray(im, dtype=np.float32) / 255.0
        if prev is not None:
            energy.append(float(np.abs(cur - prev).mean()))
        prev = cur
    return np.asarray(energy, dtype=np.float64), len(files)


def zscore(x):
    x = np.asarray(x, dtype=np.float64)
    s = x.std()
    if s < 1e-12:
        return np.zeros_like(x)
    return (x - x.mean()) / s


def best_lag(a, b, max_lag):
    """在 [-max_lag, max_lag] 上找使 a 与 b 相关最大的位移 lag。
    含义：b 相对 a 延迟 lag 帧（lag>0 表示 b 比 a 晚）。"""
    a = zscore(a)
    b = zscore(b)
    n = min(len(a), len(b))
    results = []
    for lag in range(-max_lag, max_lag + 1):
        # a[t] 与 b[t+lag] 对齐
        if lag >= 0:
            xa, xb = a[: n - lag], b[lag:n]
        else:
            xa, xb = a[-lag:n], b[: n + lag]
        if len(xa) < 30:
            continue
        r = float(np.dot(xa, xb) / len(xa))
        results.append((lag, r))
    results.sort(key=lambda t: -t[1])
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images_root")
    ap.add_argument("cams", nargs="+")
    ap.add_argument("--max_lag", type=int, default=120)
    ap.add_argument("--limit", type=int, default=None, help="只读前 N 帧，调试用")
    ap.add_argument("--crop", type=float, nargs=4, default=None,
                    metavar=("L", "T", "R", "B"), help="归一化裁剪球场中心区域")
    ap.add_argument("--out", default=None, help="互相关曲线 PNG 输出路径")
    args = ap.parse_args()

    if args.max_lag >= 0.4 * (args.limit or 900):
        print("警告: max_lag 相对序列长度偏大，峰值容易落在搜索边界（伪影）。")

    energies = {}
    counts = {}
    for cam in args.cams:
        d = os.path.join(args.images_root, cam)
        e, n = motion_energy(d, limit=args.limit, crop=args.crop)
        energies[cam] = e
        counts[cam] = n
        print(f"{cam}: {n} 帧, 运动能量 mean={e.mean():.5f} std={e.std():.5f}")

    print()
    ref = args.cams[0]
    report = {}
    for cam in args.cams[1:]:
        res = best_lag(energies[ref], energies[cam], args.max_lag)
        top = res[0]
        zero_r = [r for lag, r in res if lag == 0]
        report[cam] = {"best_lag": top[0], "best_r": top[1], "r_at_zero": zero_r[0] if zero_r else float("nan")}
        print(f"[{ref} vs {cam}]  最佳偏移 = {top[0]:+d} 帧 (r={top[1]:.3f}), "
              f"零偏移时 r={report[cam]['r_at_zero']:.3f}")
        print("    top5: " + ", ".join(f"lag={l:+d}(r={r:.3f})" for l, r in res[:5]))

    print()
    worst = max(abs(v["best_lag"]) for v in report.values())
    if worst == 0:
        print("结论：三路在帧级别对齐，未检测到同步偏移。")
    elif worst <= 2:
        print(f"结论：存在 {worst} 帧以内的微小偏移，对 30 fps 相当于 <{worst / 30 * 1000:.0f} ms，可忽略或微调。")
    else:
        print(f"结论：检测到 {worst} 帧偏移（30 fps 下约 {worst / 30 * 1000:.0f} ms），"
              f"三角化前需要按该偏移对齐，否则会引入系统性 3D 误差。")

    if args.out:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(len(args.cams), 1, figsize=(12, 2.4 * len(args.cams)), sharex=True)
            if len(args.cams) == 1:
                axes = [axes]
            for ax, cam in zip(axes, args.cams):
                ax.plot(energies[cam], lw=0.8)
                ax.set_ylabel(cam, fontsize=9)
                ax.grid(alpha=0.3)
            axes[-1].set_xlabel("frame index")
            fig.suptitle("motion energy per camera (should peak at the same frames if synced)")
            fig.tight_layout()
            fig.savefig(args.out, dpi=110)
            print(f"\n曲线图已保存: {args.out}")
        except Exception as e:
            print(f"\n(跳过绘图: {e})")


if __name__ == "__main__":
    main()
