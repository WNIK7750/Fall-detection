# Elderly Fall Detection Roadshow Demo

多视角 3D 老年人跌倒检测路演 demo。当前版本使用数据集视频模拟三路居家摄像头，在本地完成 YOLO 姿态估计、3D 重建、3D 时序分类，并在疑似事件时调用本地 Qwen3-VL 复核，最后通过 WebSocket 推送到社区工作人员监控端。

## 当前状态

- 三路 MCFD 视频输入模拟摄像头：`cam1/cam4/cam7`
- YOLOv8n-pose 批量姿态检测
- 多视角三角测量生成 3D 骨架
- 10 维 3D 特征 + BiLSTM/BiGRU/Attention 分类器
- 三级策略：`normal` / `suspected` / `fall`
- `suspected` 事件自动导出最近 90 帧原始 clip，并抽帧调用 Qwen3-VL
- 本地/LAN WebSocket dashboard 展示 `suspected` 与 `fall` 两列事件
- 无摄像头 playmode 可稳定演示完整主循环

## 隐私机制

本项目的目标场景是居家老人和社区工作人员对接。摄像头画面应尽可能在老人家中本地处理，不持续上传完整视频流。

当前 demo 采用同样的产品原则：

- 客户端只在 `suspected` 或 `fall` 时发送事件。
- `suspected` 第一时间只发送不含视频的告警和 3D classifier probability。
- 只有需要复核时才从内存 rolling buffer 导出最近 90 帧短片段；rolling buffer 默认保留 600 帧，约 20 秒。
- LLM 复核完成后，用同一个 `event_id` 更新事件状态。
- 数据集视频、LLM GGUF 权重、生成 clips 不进入 Git。
- 当前 `features/3d/` 体积很小，保留在 Git 中用于复现阈值扫描和训练结果；演示运行不依赖它。

## Roadshow Demo

### 一键 PowerShell 演示

Windows 下推荐直接运行：

```powershell
.\demo.ps1
```

脚本会启动 `monitor_server.py`、打开 dashboard，并运行 `continuous_event_client.py` 的真实数据集摄像头循环：

- 三路 MCFD 视频按 `data_tuple3.csv` 对齐后作为虚拟摄像头输入。
- YOLO/3D 重建/classifier 持续运行。
- 每换一个 scenario 都会重置 tracker、平滑器、floor estimator 和 3D 状态。
- Qwen3-VL 在客户端启动时加载一次，疑似事件发生时直接调用，不重复加载模型。
- 一个 scenario 播完后自动随机切到下一个 scenario。

如果现场机器没有 Qwen3-VL GGUF，先用无 LLM 模式验证摄像头循环和 dashboard：

```powershell
.\demo.ps1 -NoLLM
```

### 1. 启动监控端

```bash
python monitor_server.py --host 127.0.0.1 --port 8765
```

浏览器打开：

```text
http://127.0.0.1:8765/
```

Dashboard 通过 WebSocket 自动刷新；`Manual Sync` 只是断线或调试时的手动同步。

### 2. 无摄像头演示主循环

这是旧的脚本化演示方式，不需要真实摄像头，也不跑完整 YOLO/3D 链路。仅用于离线 smoke test，不作为正式路演主流程：

```bash
python playmode_event_loop.py --source-video clips/s01_1050_90.mp4 --server-url ws://127.0.0.1:8765/ws/events
```

预期流程：

1. 左列出现 `suspected`，不带视频。
2. 客户端导出 90 帧原始 clip，按 30 FPS 播放，对应约 3 秒。
3. Qwen3-VL 复核后，同一事件更新为 `fall` 并移动到右列。
4. 右列显示 clip 预览和 Qwen3-VL 结论。
5. 工作人员点击 `Mark handled`。

快速 smoke test（不加载 LLM、不发 WebSocket）：

```bash
python playmode_event_loop.py --no-send --no-llm --source-video clips/s01_1050_90.mp4 --trigger-frame 100 --max-frames 120
```

### 3. MCFD 三路视频连续客户端

使用当前数据集视频模拟三路摄像头，跑真实 YOLO/3D/classifier 主循环：

```bash
python continuous_event_client.py --roadshow-loop --start-near-fall --server-url ws://127.0.0.1:8765/ws/events
```

快速 smoke test：

```bash
python continuous_event_client.py --scenario 1 --max-frames 120 --force-trigger-frame 100 --force-p-fall 0.35 --no-llm --no-send --stop-after-event
```

Dashboard event policy:

- `fall`: immediately exports a clip and shows it in the Fall column.
- `suspected`: first sends an alert without video, then exports a clip for local Qwen3-VL review.
- LLM says fall: same `event_id` is updated to `fall`; the old suspected card disappears and the Fall card shows video.
- LLM says not fall: same event stays in the Suspected column as manual review. Staff can click `Request video`; after 30 seconds the video preview is released for human review.
- Dashboard playback does not stream one frame per request. Released clips are loaded through one `/clip-frames` request into the browser, then played locally with a play/pause button and progress bar.

### 4. LLM clip 复核单测

```bash
python scripts/qa/review_llm_clip.py --video clips/s01_1050_90.mp4
python scripts/qa/qa_llm_clip_batch.py --videos clips/s01_1050_90.mp4 clips/s01_0000_90_normal.mp4
```

## Model And Thresholds

Current 3D checkpoint:

```text
checkpoints/archive/enhanced_3d_20260712_185406/best_model.pth
```

Validation notes:

- Test split: 147 windows, fall ratio `0.3673`
- AUC-ROC: `0.9747`
- At threshold `0.40`: fall recall `1.000`, fall precision `0.818`, CM `[[81, 12], [0, 54]]`
- At threshold `0.30`: fall recall `1.000`, fall precision `0.783`, CM `[[78, 15], [0, 54]]`

Runtime policy:

- `p_fall < 0.40`: normal, no event
- `p_fall >= 0.40`: suspected, send immediate alert, export 90-frame clip, sample every 10 frames for Qwen3-VL
- LLM says fall: update the same `event_id` to `fall` and release the clip
- Events are emitted only after the rolling buffer has at least 90 frames, so released fall alerts have a complete 3-second evidence clip.

## Setup

Python 3.10+ recommended. The current dev machine uses Python 3.13 with CUDA PyTorch.

```bash
pip install -r requirements.txt
```

Important local artifacts not stored in Git:

- `dataset/mcfd/` - MCFD videos, from Kaggle: <https://www.kaggle.com/datasets/soumicksarker/multiple-cameras-fall-dataset>
- `models/llm/*.gguf` - Qwen3-VL model and mmproj files, from ModelScope: <https://www.modelscope.cn/models/Qwen/Qwen3-VL-4B-Instruct-GGUF/files>
- `clips/` - generated event clips and QA clips

Expected local layout:

```text
dataset/mcfd/dataset/chute01/cam1.avi
dataset/mcfd/dataset/chute01/cam4.avi
dataset/mcfd/dataset/chute01/cam7.avi
models/llm/Qwen3VL-4B-Instruct-Q4_K_M.gguf
models/llm/mmproj-Qwen3VL-4B-Instruct-Q8_0.gguf
```

`features/3d/` contains generated 3D feature arrays and scaler files for training and threshold sweep:

- It is not needed for the normal roadshow path (`demo.ps1`, `playmode_event_loop.py`, `continuous_event_client.py`).
- It is needed for `scripts/training/train_3d.py` and `scripts/training/sweep_threshold.py`.
- It does not introduce runtime randomness into the demo. The roadshow uses the archived checkpoint; the scaler is embedded in that checkpoint. Regenerating features can change future training results, so the current small 3D feature set is kept in Git for reproducibility.

Small model artifacts stored in Git:

- `yolov8n-pose.pt`
- `checkpoints/archive/enhanced_3d_20260712_185406/best_model.pth`
- `checkpoints/archive/enhanced_3d_20260712_185406/history.json`
- `features/3d/{train,val,test,scaler}/`

## Useful Commands

Extract 3D dataset features:

```bash
python scripts/training/extract_dataset_3d.py --device cuda --yolo-batch 32 --time-batch 64 --min-quality 0.15 --max-normal 300
```

Train 3D classifier:

```bash
python scripts/training/train_3d.py --seed 42 --no-smote
```

Sweep thresholds:

```bash
python scripts/training/sweep_threshold.py --split test --device cuda
```

OpenCV multiview visual demo:

```bash
python scripts/demo/demo_multiview.py --scenario 1
```

## Key Files

| Path | Purpose |
| --- | --- |
| `continuous_event_client.py` | Dataset-backed continuous event client |
| `playmode_event_loop.py` | No-camera roadshow playmode |
| `monitor_server.py` | FastAPI HTTP/WebSocket monitor server |
| `web/monitor_dashboard.html` | Staff dashboard |
| `pipeline/clip_buffer.py` | Privacy-preserving rolling clip buffer |
| `pipeline/event_schema.py` | Event payload builder |
| `pipeline/event_sender.py` | WebSocket sender |
| `pipeline/pose_inference.py` | YOLOv8 pose detection and tracking |
| `pipeline/triangulation.py` | Multi-view 3D reconstruction |
| `pipeline/features_3d.py` | 3D feature engineering |
| `pipeline/classifier.py` | 3D fall classifier wrapper |
| `pipeline/llm_reviewer.py` | Qwen3-VL clip/image review |
| `scripts/training/` | Feature extraction, training, threshold sweep |
| `scripts/qa/` | LLM and pose QA helpers |
| `scripts/demo/` | Visual/manual demo helpers |
| `scripts/tools/` | Dataset clip export and utility scripts |

## Demo Boundaries

- Current demo uses MCFD videos to simulate three cameras; real camera adapters are future work.
- MCFD camera calibration and 3D reconstruction are demo-suitable approximations, not production calibration.
- The monitor is local/LAN only: no auth, TLS, persistent DB, or audit log yet.
- Metrics must always cite dataset/split/test condition. Do not claim production accuracy such as "99.9%".

## License

Academic/coursework roadshow demo. Code and documentation are for learning and demonstration.
