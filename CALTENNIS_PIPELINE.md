# CalTennis 多机位 pseudo-GT 实验代码说明

本仓库保留 EasyMocap 上游代码，以及 CalTennis 实验所需的少量自定义脚本。原始视频、图像帧、人体模型、检测权重、运行日志和输出结果不进入 Git。

## 当前主流程

1. `scripts/prepare_caltennis_easymocap.py`：抽帧并准备 EasyMocap 数据目录。
2. `scripts/detect_caltennis_4cam_bytetrack.py`：YOLOv8 + ByteTrack 检测跟踪，ViTPose-H 输出 BODY25 关键点。
3. `apps/demo/mvmp.py`：跨视角关联和多视角三角化。
4. `apps/demo/auto_track.py --track3d`：短时 3D 轨迹关联。
5. `scripts/stitch_track3d_fragments.py`：依据球场位置拼接长期身份并可选补短缺口。
6. `apps/demo/smpl_from_keypoints.py`：由 3D 关键点拟合 SMPL。

`scripts/easymocap_corrected_w2_bytetrack_full.sbatch` 是完整流程入口。各阶段路径和参数都在该脚本顶部集中定义。

## 诊断与对照

- `scripts/diagnose_multiview_sync_arm.py`：用标定相机的极线几何搜索残余帧偏移，并统计快速挥拍时肩、肘、腕 2D 关键点质量。
- `scripts/compare_smpl_temporal_lag.py`：比较不同 SMPL 时序平滑权重下的拟合误差、快速手臂误差和抖动。
- `scripts/easymocap_smpl_temporal_sweep.sbatch`：运行 `(5,1,1)`、`(0.5,0.1,0.2)`、`(0,0,1)`、`(0,0,0)` 四组平滑设置。
- `scripts/easymocap_corrected_w2_nointerp_comparison.sbatch`：比较开启和关闭 3D 缺失帧插值。

## 不上传 GitHub 的内容

- `data/`：原视频、抽帧、标定副本、2D/3D 关键点、SMPL 参数和渲染结果。
- `models/`、`checkpoints/`、`*.pt`、`*.pth`：受许可证约束或体积较大的模型文件。
- `*.out`、`*.log`：Slurm 和运行日志。
- `outputs/`、`results/`：可再生成的实验产物。

复现实验时，需要自行准备有权使用的 SMPL 模型和检测器权重，并修改 sbatch 中的 `ROOT`、`GVHMR` 与 Python 环境路径。
