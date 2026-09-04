# baidu_mocap_qiyt_pilot — 三机位网球视频人体 mocap 可行性验证

目的：回答「这批新视频能不能产出可用的人体 mocap」。
**结论：能。** 三路 450/450 帧全检出，关键点可用率 100%。

---

## 1. 文件来源（谁写的）

| 文件 | 作者 | 说明 |
|---|---|---|
| `../baidu_mocap_20260831/README.md` | **学长 `wuzhx12022`** | 技术方案 / 需求文档（Sep 1 16:57）。不是代码。 |
| `../baidu_mocap_20260831/config.json` | **学长 `wuzhx12022`** | 原始配置（Sep 1 17:27）。 |
| `config.json` | **我（改自学长版）** | 只改了 3 处，见下。 |
| `prepare_baidu.py` | 我 | 抽帧 + 缩放 + 首帧预览 + manifest。 |
| `make_contact_sheet.py` | 我 | 多时刻联系表，用于挑选有效时间窗。 |
| `check_sync.py` | 我 | **整帧差分法，结果不可靠，仅留档**，勿用于结论。 |
| `eval_gvhmr_pilot.py` | 我 | GVHMR 质量评估 + 人体速度曲线同步性分析。 |

学长目录 `baidu_mocap_20260831/` 已加入 `.gitignore` —— 非本人作品且含组内私有路径，未获授权前不发布到公开仓库。

### 我相对学长 `config.json` 改了什么

只改了 3 个字段，其余（含 `intrinsics_candidate`）原样保留：

| 字段 | 学长原值 | 我的值 | 原因 |
|---|---|---|---|
| `source_root` | `/inspurfs/group/mayuexin/renym/...` | `~/EasyMocap/BaiduNetdiskDownload` | 前者我账号 `Permission denied` |
| `output_root` | `baidu_mocap_20260831/work` | `baidu_mocap_qiyt_pilot/work` | 学长目录只读，不覆盖他的文件 |
| `source_fps` | （无） | `60` | 原配置漏了源帧率，抽帧需要 |

---

## 2. 数据来源

```
~/EasyMocap/BaiduNetdiskDownload/2025-07-08 上体采集数据 3相机 4K 运动员/热身/tennis_camera_{1,2,3}.mp4
```

即学长 README 默认指定的 clip。三路 ffprobe 实测一致：
**h264 / 3840×2160 / 60 fps / 23040 帧 / 384 秒**，各约 1.9 GB。

同目录下另有「多球」（各 4.3 GB）与「比赛」（各 6.3 GB）两组，本轮未测。

---

## 3. 处理流程

| 步骤 | 参数 | 产物 |
|---|---|---|
| 抽帧 | 隔帧取 60→30 fps（不插帧），缩放 1920×1080（不裁切），JPEG q95，起点 0，取 900 帧 | `work/images/<cam>/000000.jpg ~ 000899.jpg`（源 0–30 s） |
| 选窗 | 联系表显示 0–15 s 运动员在场边 → 取 15–30 s | 帧 450–899 |
| 编码 | 450 帧 | `work/videos_15_30s/tennis_camera_{1,2,3}.mp4` |
| GVHMR | job `993010_1~3`，CS286 分区 / 2080Ti，**不带 `--fx/--fy`**（demo.py 不接受该参数） | `~/GVHMR/outputs/baidu_pilot_15_30s_tennis_camera_{1,2,3}/` |
| 评估 | `eval_gvhmr_pilot.py` | `work/pilot_eval.json` |

原视频全程未修改。

复现命令：

```bash
cd ~/EasyMocap/baidu_mocap_qiyt_pilot
~/miniconda3/envs/cs224n/bin/python prepare_baidu.py --config config.json
sbatch ~/GVHMR/gvhmr_baidu_pilot.sbatch
~/miniconda3/envs/gvhmr/bin/python eval_gvhmr_pilot.py --out work/pilot_eval.json
```

---

## 4. 结果（15–30 s，450 帧）

| 指标 | camera_1 | camera_2 | camera_3 |
|---|---|---|---|
| 检出率 | 100% | 100% | 100% |
| ViTPose 置信度（均值） | 0.871 | 0.860 | 0.835 |
| 关键点可用率 | 100% | 100% | 100% |
| 人体框高 px（1080p） | 174.8 | 161.3 | 233.8 |
| 抖动加速度 p95 (m/s²) | 8.24 | 12.02 | 9.01 |

- 无漏检；最远的 camera_1 人体仍有 175 px 高，远景未构成障碍。
- 抖动 p95 8–12 m/s² 属单目 GVHMR 固有抖动 —— 这是后续做多视角三角化的动机。
- GVHMR 输出的 `preprocess/vitpose.pt` 已含 (450, 17, 3) 的 2D 关键点，**三角化无需另跑 ViTPose**。

---

## 5. ⚠️ 两个未解决项

### 5.1 内参分歧

| 来源 | fx | 水平 FOV |
|---|---|---|
| GVHMR 自估 | 2202.9 | ≈ 47° |
| 学长 FAST-Calib 候选 | 1510.8 | ≈ 65° |

差很远，直接影响 metric 尺度（上表「距相机距离」是按 fx=2202.9 算的，**在内参定案前不成立**）与三角化精度。
待确认：FAST-Calib 那组值是在 4K 还是 1080p 下标定。

验证思路：网球场是天然标定板 —— 双打场地 10.97 × 23.77 m、网高中间 0.914 m、发球线距网 6.4 m 均为标准值；标 2D 线 + 标准 3D 坐标 → `solvePnP` 求外参，点够多还可反解内参。

### 5.2 同步性尚未定论

`pilot_eval.json` 中 `sync.*.reliable = false`：相关性峰值突出度仅 0.001–0.009，top5 flatness < 0.007。
只能说**未见系统性偏移（±2 帧内）**，**不能宣称已确认同步**。15 秒样本 + 远景小目标证据不足，定论需更长片段或闪光灯 / 音频对齐。

注：早期用整帧差分做同步检测的方法已废弃 —— 最佳偏移总落在搜索边界（设 ±30 报 +30，设 ±120 报 −60），属边界伪影，是误报。

---

## 6. 下一步

1. 球场线标定 → 外参（+ 反解内参）
2. 三视角三角化 2D 关键点 → 3D 骨架
3. SMPL 拟合（复用 `scripts/triang_robust.py` 的身份锁定思路）
4. 与单目 GVHMR 结果对比抖动改善幅度
