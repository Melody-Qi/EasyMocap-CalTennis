# CalTennis EasyMocap 双视角人体重建 Pipeline

基于 EasyMocap 的多相机（4 路）网球双打三维人体重建工具集。目标是把 4 路同步视频 → 稳定的双人 3D 骨架 → SMPL 网格。

## 背景与核心难点
- 输入：4 路同步相机（W_01 / W_02 / E_03 / E_04），已用球场标定（`intri.yml` / `extri.yml`）。
- 难点：多人三角化时，EasyMocap 默认的多人管线（`mvmp`）逐帧贪心配对，球员移动或某视角漏检时会出现**身份翻转**（骨架在两人之间跳来跳去）。

## 已解决的方案（关键）
`triang_robust.py` 用两招从结构上消除 ID 翻转：
1. **身份由相机对锁定**：HIGH 侧球员（场地一端，x≈24）只由 `W_01 + E_03` 重建并写 id0；LOW 侧球员（x≈0）只由 `W_02 + E_04` 重建并写 id1 —— 结构上不可能跨人。
2. **following-anchor（跟踪锚点）**：每一帧把上一帧的 3D 质心投影回各相机，挑最近的检测，锚点跟着人走，避免漏检时被远处的错误检测拉偏。

实测（834 个双人帧）：**身份翻转次数 = 0**，两人 3D 质心恒距 14.5–17.9 m，帧间平均位移 0.03 m。

## 文件索引
| 文件 | 作用 |
|---|---|
| `triang_robust.py` | **核心**：4 相机 2D 关键点 → 稳定 ID 的 3D 骨架（固定相机对 + following-anchor） |
| `smooth_k3d.py` | 将单帧离群点（>0.7 m 突跳）用相邻帧线性插值，输出平滑的 `keypoints3d` |
| `vis_repro_clean.py` | 渲染干净可视化：每个相机**只画它真实拍到的那名球员** + P0/P1 标签，消除"两人都投到每视角"的视觉混淆 |
| `vis_repro_4cam.py` | 旧可视化（把两人都投影到每个相机，已弃用，仅留档） |
| `detect_caltennis_4cam*.py` | 4 路 2D 关键点检测（含 ByteTrack 版本） |
| `easymocap_*.sbatch` | SLURM 作业脚本：串联三角化 / SMPL 拟合 / 可视化 / ffmpeg |
| `stitch_track3d_fragments.py`、`render_smpl_fourcam_robust.py` 等 | 后续时序拼接与 SMPL 渲染工具 |
| `smpl_female_render.yml` | SMPL 拟合配置（female 模型） |

> 其余 `dataset/`、`preprocess/`、`postprocess/`、`publish/` 为通用 EasyMocap 样例/工具脚本，作为参考保留。

## 典型流程
1. 检测：从 4 路视频提取 2D 关键点（VitPose / 检测器输出 `annots`）。
2. 三角化：`python triang_robust.py` → `output/mvmp_all/keypoints3d`（960 帧）。
3. 平滑：`python smooth_k3d.py output/mvmp_all/keypoints3d`（就地平滑）。
4. SMPL 拟合：通过 `easymocap_*.sbatch` 提交集群，得到 `smpl_all`。
5. 可视化：`vis_repro_clean.py` 生成逐相机干净视频。

## 数据位置
- 相机标定、视频、GT/伪 GT 产物（keypoints3d / smpl_all / 视频）均位于集群 `/public/home/CS286/qiyt2023-CS286/EasyMocap/data/caltennis_0224_4cam/`，**不纳入本仓库**（见 `.gitignore`）。

## 已知限制 / 下一步
- LOW 侧相机约 13% 帧漏检（遮挡），目前靠 temporal 插值；可上全局 K=2 聚类给持久 ID + Kalman 进一步提升鲁棒性。
- 与 CalTennis 官方数据集对齐：CalTennis 是无标签基准，仅用跨视角一致性做评测，本项目提供的是**可用的伪 GT 生成流程**。
