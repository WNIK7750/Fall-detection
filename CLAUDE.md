# CLAUDE.md — Fall Detection Project

Project-level instruction file for AI agents working on this repository. Read this first.

## Project

**老年人跌倒检测系统 — Roadshow Demo**

Goal: demonstrate a privacy-preserving elderly fall detection workflow using multi-view 3D pose and local Qwen3-VL review.

Current architecture:

```text
3-view video input / playmode
  -> YOLOv8n-pose batch inference
  -> multi-view triangulation
  -> 3D skeleton and 10-D features
  -> BiLSTM/BiGRU/Attention fall classifier
  -> normal / suspected / fall
  -> suspected event clip review with local Qwen3-VL
  -> WebSocket community dashboard
```

## Current Demo Status

All P0-P3 roadshow tasks are implemented.

- P0 continuous client: `continuous_event_client.py`
- No-camera playmode: `playmode_event_loop.py`
- Monitor server: `monitor_server.py`
- Dashboard: `web/monitor_dashboard.html`
- LLM clip review: `scripts/qa/review_llm_clip.py`, `scripts/qa/qa_llm_clip_batch.py`
- Runtime LLM review uses one background queue/worker with one loaded Qwen3-VL reviewer; the detector loop submits suspected clips and continues.
- YOLO pose uses native batch inference across the three synchronized camera frames. Do not split this into three same-GPU processes unless a measured multi-GPU/IO bottleneck justifies it.
- Current 3D checkpoint: `checkpoints/archive/enhanced_3d_20260712_185406/best_model.pth`
- Default review threshold: `0.40`
- Next classifier/data task: migrate training and realtime classifier windows from the current 20-step/subsampled setup to explicit 40 consecutive 3D feature frames.

Current best 3D classifier notes:

- Test split: 147 windows, fall ratio `0.3673`
- AUC-ROC: `0.9747`
- At threshold `0.40`: fall recall `1.000`, fall precision `0.818`, CM `[[81, 12], [0, 54]]`
- At threshold `0.30`: fall recall `1.000`, fall precision `0.783`, CM `[[78, 15], [0, 54]]`

Important: these metrics belong to the archived 20-step checkpoint. After the planned 40-frame feature extraction and retraining, update this section with the new checkpoint path and metrics.

## Planned 40-Frame Classifier Window

The desired next implementation is not a full architecture rewrite. Keep the existing pipeline shape:

```text
per-frame YOLO -> per-frame 3D skeleton -> per-frame 10-D feature
  -> rolling feature buffer -> latest 40 feature frames -> BiLSTM/BiGRU/Attention
```

Target data/inference semantics:

- One training sample is exactly 40 consecutive 3D feature frames at real 30 FPS.
- The model input becomes `(40, 10)` per sample after training code transposes from saved `(10, 40)` feature arrays.
- Fall sample: a 40-frame cam1-reference window must contain at least 25 consecutive fall-labeled frames.
- Normal sample: a 40-frame cam1-reference window must contain 0 fall-labeled frames and should keep a safety gap from fall intervals.
- Sampling may be random only after legal fall/normal candidate windows have been constructed.
- Use `data_tuple3.csv` plus the existing `pipeline.sync.compute_offsets()` / `MCFDReader` alignment behavior. Choose windows on the cam1 reference timeline; `MCFDReader` reads cam4/cam7 with the correct offsets.
- Old checkpoint and old `features/3d/*.npy` are not semantically valid for the 40-frame policy. Regenerate features, retrain, sweep thresholds, then update README/CLAUDE metrics.

## Privacy/Product Mechanism

This project targets elderly-home cameras connected to community staff. Treat privacy as a product requirement, not just a demo limitation.

- Process camera streams locally where possible.
- Do not continuously upload or persist full video streams.
- Keep a rolling in-memory buffer on the client.
- Send an immediate `suspected` event without video when `p_fall >= 0.40`; all such events go through Qwen3-VL review.
- Treat MCFD AVI files as 30 FPS real time even though the container reports 120 FPS.
- On classifier trigger, keep the trigger event pending until 151 post-trigger frames have been collected. Evidence clips contain 90 pre/trigger frames + 151 post frames = 241 frames, exactly 8.0 seconds at 30 FPS.
- Qwen3-VL reviews three images: one 3x3 contact sheet per camera view. Each sheet samples 9 frames from the 241-frame evidence window using stride 29.
- Do not run LLM review or release video until the full 241-frame evidence window is available.
- Use the same `event_id` for initial suspected alert and final LLM update.
- The current roadshow policy does not use direct classifier-only fall alerts; LLM decides whether a suspected event becomes `fall`.
- Dashboard playback fetches released clips via one `/clip-frames` request and plays the cached frames locally. Do not reintroduce per-frame HTTP polling for playback.

## Roadshow Commands

One-command Windows roadshow:

```powershell
.\demo.ps1
.\demo.ps1 -NoLLM
```

`demo.ps1` is the formal roadshow entry point. It starts the monitor and the
real `continuous_event_client.py` loop over synchronized MCFD cameras. Do not
use scripted playmode or forced probabilities for the main roadshow path.

Start monitor:

```bash
python monitor_server.py --host 127.0.0.1 --port 8765
```

Open:

```text
http://127.0.0.1:8765/
```

No-camera playmode, smoke-test only:

```bash
python playmode_event_loop.py --source-video clips/s01_1050_90.mp4 --server-url ws://127.0.0.1:8765/ws/events
```

Dataset-backed continuous client:

```bash
python continuous_event_client.py --roadshow-loop --start-near-fall --server-url ws://127.0.0.1:8765/ws/events
```

Fast smoke test without LLM or WebSocket:

```bash
python continuous_event_client.py --scenario 1 --max-frames 120 --force-trigger-frame 100 --force-p-fall 0.35 --no-llm --no-send --stop-after-event
```

LLM clip QA:

```bash
python scripts/qa/review_llm_clip.py --video clips/s01_1050_90.mp4
python scripts/qa/qa_llm_clip_batch.py --videos clips/s01_1050_90.mp4 clips/s01_0000_90_normal.mp4
```

3D extraction / training / threshold sweep:

```bash
python scripts/training/extract_dataset_3d.py --device cuda --yolo-batch 32 --time-batch 64 --min-quality 0.15 --max-normal 300
python scripts/training/train_3d.py --seed 42 --no-smote
python scripts/training/sweep_threshold.py --split test --device cuda
```

## Important Files

| Path | Purpose |
| --- | --- |
| `README.md` | User-facing runbook and demo explanation |
| `continuous_event_client.py` | MCFD-backed continuous detector/client |
| `playmode_event_loop.py` | No-camera demo loop |
| `monitor_server.py` | Local FastAPI/WebSocket monitor server |
| `web/monitor_dashboard.html` | Two-column staff dashboard |
| `pipeline/clip_buffer.py` | In-memory rolling clip buffer |
| `pipeline/event_schema.py` | Event JSON schema helper |
| `pipeline/event_sender.py` | WebSocket event sender |
| `pipeline/video_reader.py` | MCFD synchronized video reader |
| `pipeline/pose_inference.py` | YOLOv8 pose detection and simple tracking |
| `pipeline/calibration.py` | MCFD/Auvinet calibration approximation |
| `pipeline/triangulation.py` | 3D skeleton triangulation and smoothing |
| `pipeline/features_3d.py` | 10-D 3D feature extraction |
| `pipeline/classifier.py` | 3D classifier wrapper and heuristic fallback |
| `pipeline/llm_reviewer.py` | Qwen3-VL review logic |
| `scripts/training/` | Feature extraction, training, threshold sweep |
| `scripts/qa/` | LLM and pose QA helpers |
| `scripts/demo/` | Visual/manual demo helpers |
| `scripts/tools/` | Dataset clip export and utility scripts |
| `checkpoints/archive/enhanced_3d_20260712_185406/` | Current shipped 3D checkpoint |

## Local Artifacts And Git Hygiene

Do not commit:

- `dataset/mcfd/`
- `models/llm/*.gguf`
- `clips/`
- `results/llm_qa/`
- `checkpoints/enhanced_3d_*`

Commit and keep:

- `yolov8n-pose.pt`
- `checkpoints/archive/enhanced_3d_20260712_185406/best_model.pth`
- `checkpoints/archive/enhanced_3d_20260712_185406/history.json`
- `features/3d/{train,val,test,scaler}/`

## Demo Boundaries

- Current demo uses MCFD dataset videos to simulate three cameras.
- Real webcam/RTSP adapters are future work.
- MCFD calibration and 3D reconstruction are demo-suitable approximations.
- The dashboard is local/LAN only: no auth, TLS, database, retention policy, or audit trail yet.
- Metrics must name dataset/split/test condition. Never claim production-grade or unsupported accuracy.

## Working Rules For Agents

- Keep code changes scoped to the requested task.
- Prefer existing pipeline modules over new abstractions unless the new boundary is clearly useful.
- Preserve privacy semantics: no continuous video upload/storage.
- Keep thresholds configurable.
- After meaningful project state changes, update this `CLAUDE.md` or `README.md`; there is no longer a separate `Docs/Chat.md` memory file.
- Before committing, check `git status --short` and avoid adding ignored/generated artifacts.
- Use Chinese commit messages in the existing style, e.g. `feat: ...`, `fix: ...`, `docs: ...`.
