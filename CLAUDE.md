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
- LLM clip review: `review_llm_clip.py`, `qa_llm_clip_batch.py`
- Current 3D checkpoint: `checkpoints/archive/enhanced_3d_20260712_185406/best_model.pth`
- Default thresholds: low `0.30`, high `0.40`

Current best 3D classifier notes:

- Test split: 147 windows, fall ratio `0.3673`
- AUC-ROC: `0.9747`
- At threshold `0.40`: fall recall `1.000`, fall precision `0.818`, CM `[[81, 12], [0, 54]]`
- At threshold `0.30`: fall recall `1.000`, fall precision `0.783`, CM `[[78, 15], [0, 54]]`

## Privacy/Product Mechanism

This project targets elderly-home cameras connected to community staff. Treat privacy as a product requirement, not just a demo limitation.

- Process camera streams locally where possible.
- Do not continuously upload or persist full video streams.
- Keep a rolling in-memory buffer on the client.
- Send an immediate `suspected` event without video when `0.30 <= p_fall < 0.40`.
- Export/share only the latest 90-frame clip when a suspected event needs review.
- Use the same `event_id` for initial suspected alert and final LLM update.
- Direct `fall` events (`p_fall >= 0.40`) skip LLM by default.

## Roadshow Commands

Start monitor:

```bash
python monitor_server.py --host 127.0.0.1 --port 8765
```

Open:

```text
http://127.0.0.1:8765/
```

No-camera playmode, recommended for live roadshow:

```bash
python playmode_event_loop.py --source-video clips/s01_1050_90.mp4 --server-url ws://127.0.0.1:8765/ws/events
```

Dataset-backed continuous client:

```bash
python continuous_event_client.py --scenario 1 --max-frames 120 --force-trigger-frame 100 --force-p-fall 0.35 --server-url ws://127.0.0.1:8765/ws/events --stop-after-event
```

Fast smoke test without LLM or WebSocket:

```bash
python continuous_event_client.py --scenario 1 --max-frames 120 --force-trigger-frame 100 --force-p-fall 0.35 --no-llm --no-send --stop-after-event
```

LLM clip QA:

```bash
python review_llm_clip.py --video clips/s01_1050_90.mp4
python qa_llm_clip_batch.py --videos clips/s01_1050_90.mp4 clips/s01_0000_90_normal.mp4
```

3D extraction / training / threshold sweep:

```bash
python extract_dataset_3d.py --device cuda --yolo-batch 32 --time-batch 64 --min-quality 0.15 --max-normal 300
python train_3d.py --seed 42 --no-smote
python sweep_threshold.py --split test --device cuda
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
| `checkpoints/archive/enhanced_3d_20260712_185406/` | Current shipped 3D checkpoint |

## Local Artifacts And Git Hygiene

Do not commit:

- `dataset/mcfd/`
- `models/llm/*.gguf`
- `features/3d/`
- `clips/`
- `results/llm_qa/`
- `checkpoints/enhanced_3d_*`

Commit and keep:

- `yolov8n-pose.pt`
- `checkpoints/archive/enhanced_3d_20260712_185406/best_model.pth`
- `checkpoints/archive/enhanced_3d_20260712_185406/history.json`

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
