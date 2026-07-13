# Elderly Fall Detection Roadshow Demo

多视角 3D 老年人跌倒检测路演 demo。当前版本使用数据集视频模拟三路居家摄像头，在本地完成 YOLO 姿态估计、3D 重建、3D 时序分类，并在疑似事件时调用本地 Qwen3-VL 复核，最后通过 WebSocket 推送到社区工作人员监控端。

## 当前状态

- 三路 MCFD 视频输入模拟摄像头：`cam1/cam4/cam7`
- YOLOv8n-pose 批量姿态检测
- 多视角三角测量生成 3D 骨架
- 10 维 3D 特征 + BiLSTM/BiGRU/Attention 分类器
- 三级策略：`normal` / `suspected` / `fall`
- `suspected` 事件自动导出最近 90 帧 clip 并调用 Qwen3-VL
- 本地/LAN WebSocket dashboard 展示 `suspected` 与 `fall` 两列事件
- 无摄像头 playmode 可稳定演示完整主循环

## 隐私机制

本项目的目标场景是居家老人和社区工作人员对接。摄像头画面应尽可能在老人家中本地处理，不持续上传完整视频流。

当前 demo 采用同样的产品原则：

- 客户端只在 `suspected` 或 `fall` 时发送事件。
- `suspected` 第一时间只发送不含视频的告警和 3D classifier probability。
- 只有需要复核时才从内存 rolling buffer 导出最近 90 帧短片段。
- LLM 复核完成后，用同一个 `event_id` 更新事件状态。
- 数据集视频、LLM GGUF 权重、生成 clips、生成 features 不进入 Git。

## Roadshow Demo

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

这是最稳的路演方式，不需要真实摄像头，也不需要现场跑完整 YOLO/3D 链路：

```bash
python playmode_event_loop.py --source-video clips/s01_1050_90.mp4 --server-url ws://127.0.0.1:8765/ws/events
```

预期流程：

1. 左列出现 `suspected`，不带视频。
2. 客户端导出 90 帧 clip。
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
python continuous_event_client.py --scenario 1 --max-frames 120 --force-trigger-frame 100 --force-p-fall 0.35 --server-url ws://127.0.0.1:8765/ws/events --stop-after-event
```

快速 smoke test：

```bash
python continuous_event_client.py --scenario 1 --max-frames 120 --force-trigger-frame 100 --force-p-fall 0.35 --no-llm --no-send --stop-after-event
```

### 4. LLM clip 复核单测

```bash
python review_llm_clip.py --video clips/s01_1050_90.mp4
python qa_llm_clip_batch.py --videos clips/s01_1050_90.mp4 clips/s01_0000_90_normal.mp4
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

- `p_fall < 0.30`: normal, no event
- `0.30 <= p_fall < 0.40`: suspected, send immediate alert, export 90-frame clip, run Qwen3-VL
- `p_fall >= 0.40`: direct fall alert, skip LLM

## Setup

Python 3.10+ recommended. The current dev machine uses Python 3.13 with CUDA PyTorch.

```bash
pip install -r requirements.txt
```

Important local artifacts not stored in Git:

- `dataset/mcfd/` - MCFD videos
- `models/llm/*.gguf` - Qwen3-VL model and mmproj files
- `features/3d/` - generated 3D features
- `clips/` - generated event clips and QA clips

Small model artifacts stored in Git:

- `yolov8n-pose.pt`
- `checkpoints/archive/enhanced_3d_20260712_185406/best_model.pth`
- `checkpoints/archive/enhanced_3d_20260712_185406/history.json`

## Useful Commands

Extract 3D dataset features:

```bash
python extract_dataset_3d.py --device cuda --yolo-batch 32 --time-batch 64 --min-quality 0.15 --max-normal 300
```

Train 3D classifier:

```bash
python train_3d.py --seed 42 --no-smote
```

Sweep thresholds:

```bash
python sweep_threshold.py --split test --device cuda
```

OpenCV multiview visual demo:

```bash
python demo_multiview.py --scenario 1
```

Older 2D baseline evaluation:

```bash
python ensemble_inference.py --eval
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

## Demo Boundaries

- Current demo uses MCFD videos to simulate three cameras; real camera adapters are future work.
- MCFD camera calibration and 3D reconstruction are demo-suitable approximations, not production calibration.
- The monitor is local/LAN only: no auth, TLS, persistent DB, or audit log yet.
- Metrics must always cite dataset/split/test condition. Do not claim production accuracy such as "99.9%".

## License

Academic/coursework roadshow demo. Code and documentation are for learning and demonstration.
