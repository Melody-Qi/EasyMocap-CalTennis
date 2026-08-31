# CalTennis 多机位双打三维人体重建（基于 EasyMocap）

本仓库 = **上游 [zju3dv/EasyMocap](https://github.com/zju3dv/EasyMocap) 代码** + **CalTennis 实验的自定义脚本**。
根目录 `Readme.md` 是上游原始说明；本文件是本项目自己的说明。另一份 `CALTENNIS_PIPELINE.md` 记录逐阶段命令入口。

目标：把 4 路同步相机视频 → 稳定的双人 3D 骨架 → SMPL 网格（伪 GT）。

## 核心难点与解法

**难点**：多人三角化时，EasyMocap 默认的 `mvmp` 逐帧贪心配对，在球员移动或某视角漏检时会产生**身份翻转**（骨架在两人之间跳来跳去）。

**解法**（`scripts/triang_robust.py`，两招从结构上消除 ID 翻转）：

1. **身份由相机对锁定**：HIGH 侧球员（场地一端，x≈24）只由 `W_01 + E_03` 重建并写 id0；LOW 侧球员（x≈0）只由 `W_02 + E_04` 重建并写 id1 —— 结构上不可能跨人。
2. **following-anchor（跟踪锚点）**：每帧把上一帧的 3D 质心投影回各相机，挑最近的检测，锚点跟着人走，避免漏检时被远处错误检测拉偏。

实测（834 个双人帧）：**身份翻转次数 = 0**，两人 3D 质心恒距 14.5–17.9 m，帧间平均位移 0.03 m。

## 脚本索引

| 文件 | 作用 |
|---|---|
| `scripts/triang_robust.py` | **核心**：4 相机 2D 关键点 → 稳定 ID 的 3D 骨架（固定相机对 + following-anchor） |
| `scripts/smooth_k3d.py` | 将单帧离群点（>0.7 m 突跳）用相邻帧线性插值，输出平滑 `keypoints3d` |
| `scripts/vis_repro_clean.py` | 干净可视化：每个相机**只画它真实拍到的那名球员** + P0/P1 标签，消除"两人都投到每视角"的视觉混淆 |
| `scripts/vis_repro_4cam.py` | 旧可视化（把两人都投影到每个相机，已弃用，仅留档） |
| `scripts/detect_caltennis_4cam*.py` | 4 路 2D 关键点检测（含 ByteTrack 版本） |
| `scripts/diagnose_multiview_sync_arm.py` | 极线几何搜索残余帧偏移 + 统计快速挥拍时肩/肘/腕关键点质量 |
| `scripts/compare_smpl_temporal_lag.py` | 比较不同 SMPL 时序平滑权重下的拟合误差与抖动 |
| `scripts/stitch_track3d_fragments.py` | 依据球场位置拼接长期身份，可选补短缺口 |
| `scripts/easymocap_*.sbatch` | SLURM 作业脚本：串联检测 / 三角化 / SMPL 拟合 / 可视化 / ffmpeg |
| `scripts/smpl_female_render.yml` | SMPL 拟合配置（female 模型） |

## 典型流程

1. 检测：4 路视频提取 2D 关键点（YOLOv8 + ByteTrack，ViTPose-H 输出 BODY25）。
2. 三角化：`python scripts/triang_robust.py` → `keypoints3d`（960 帧）。
3. 平滑：`python scripts/smooth_k3d.py <keypoints3d 目录>`（就地平滑）。
4. SMPL 拟合：通过 `easymocap_*.sbatch` 提交集群，得到 `smpl_all`。
5. 可视化：`scripts/vis_repro_clean.py` 生成逐相机干净视频。

## 不进入 Git 的内容

见 `.gitignore`，要点：

- `data/`：原始视频、抽帧、标定副本、2D/3D 关键点、SMPL 参数与渲染结果（约 36 GB）
- `BaiduNetdiskDownload/`：从百度网盘下载的 3 批采集原始视频（约 42 GB）
- `models/`、`checkpoints/`、`*.pt`、`*.pth`、`*.pkl`：模型权重（受许可证约束或体积过大）
- `*.mp4` 等视频、`*.pcap` 等点云原始数据
- `*.out`、`*.log`、`slurm-*.out`：运行日志
- `outputs/`、`results/`：可再生成的实验产物

复现实验需自行准备有权使用的 SMPL 模型与检测器权重，并修改 sbatch 中的 `ROOT`、`GVHMR` 与 Python 环境路径。

## 已知限制 / 下一步

- LOW 侧相机约 13% 帧漏检（遮挡），目前靠时序插值；可上全局 K=2 聚类给持久 ID + Kalman 提升鲁棒性。
- CalTennis 官方数据集无 3D 标签，本项目产出的是**可用的伪 GT 生成流程**，评测只能靠跨视角一致性。
- 下一步方向：球与人体的 LiDAR + camera 融合（LiDAR 主动测距拿 metric 3D，绕开多视角 2D 关联难题）。`BaiduNetdiskDownload/2025-10-22 网球激光雷达采集 + 分析` 是该方向的素材，但**其中只有 RSView 上位机录屏，没有原始 `.pcap` 点云**，需向采集者索取。
