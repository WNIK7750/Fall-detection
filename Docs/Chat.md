# Fall Detection Project — Agent External Memory

> Last updated: 2026-07-13 | Agent: Codex GPT-5

---

## Project Identity

- **Project**: 老年人跌倒检测系统 — 20-Day Roadshow Demo
- **Goal**: Upgrade existing 2D single-view fall detection → Multi-view 3D system with LLM review
- **Timeline**: 20 days from 2026-07-10 (deadline: 2026-07-30)
- **Engine**: Python 3.x, PyTorch, OpenCV, ultralytics, llama-cpp-python
- **Hardware**: RTX 3060 Ti 8GB VRAM
- **Git**: https://github.com/WNIK7750/Fall-detection.git (main branch)
- **Primary Doc**: `Docs/prompt.md` (187 lines, full spec)

---

## Current State (Pre-Upgrade)

### Already Built (2D System)
- **Pose Extraction**: YOLOv8n-pose + MediaPipe Pose, 10-dim features per 20-frame window
- **Model**: BiLSTM + BiGRU + Bidirectional Multi-Head Self-Attention (~4.3M params full, ~130K compact)
- **Training**: SMOTE oversampling + Focal Loss (α=3.0, γ=2.0) + data augmentation
- **Inference**: Dual-model soft-voting ensemble (Model A high precision + Model B high recall)
- **Metrics**: Precision 75%, Recall 83.33%, F1 78.95%, AUC-ROC >0.94 (UR Fall Dataset)
- **Demo**: `demo_video.py` — real-time pose visualization + fall probability overlay
- **Datasets Used**: UR Fall Detection Dataset (URFD) — 70 videos, single camera (cam0 only)
- **Features Extracted**: `features/train|val|test/` — .npy files, 10-dim × 20-frame windows

### Key Files
| File | Role |
|------|------|
| `ensemble_inference.py` | Main inference entry (dual-model soft voting) |
| `demo_video.py` | Real-time demo with visualization |
| `train_enhanced.py` | Training script (BiLSTM+BiGRU+Attention) |
| `extract_features_v3.py` | Feature extraction (MediaPipe version) |
| `extract_gmdcsa24.py` | Feature extraction (YOLOv8 batch version) |
| `models/enhanced_lstm.py` | Model architecture definitions |
| `yolov8n-pose.pt` | YOLOv8 pose model weights (6.52 MB) |
| `checkpoints/` | 8 enhanced runs + archived experiments |
| `features/` | UR Fall extracted features (360 train, 92 val, 108 test) |

---

## Development Progress (2026-07-12)

### Environment ✅ Complete
| Component | Status | Detail |
|-----------|--------|--------|
| Python 3.13.10 + pip 26.1.2 | ✅ | All packages installed |
| PyTorch 2.6.0+cu124 | ✅ | CUDA, RTX 3060 Ti 8GB |
| ultralytics 8.4.92 | ✅ | YOLOv8n-pose |
| llama-cpp-python 0.3.33 | ✅ | LLM inference |
| Gradio 6.20.0 | ✅ | Demo UI |

### Data ✅ Downloaded
| Item | Location | Detail |
|------|----------|--------|
| MCFD | `dataset/mcfd/` | 24 chutes × 8 cams = 192 .avi |
| Qwen3-VL-4B | `models/llm/` | Q4_K_M + mmproj Q8_0 (~3.6GB) |
| Qwen3-VL-8B | `models/llm/` | Q4_K_M only, backup |

### LLM: 4B-First Strategy
- Use 4B to build and test full pipeline (fits 8GB VRAM comfortably)
- After pipeline works end-to-end, evaluate if 8B upgrade is worthwhile
- Current llama-cpp path uses `Qwen25VLChatHandler` plus the mmproj file for image input.
- Suspected events now use dual-pass VL review:
  1. raw visual pass on unmodified 3-camera collage
  2. assisted pass on pose-overlaid collage plus YOLO/3D/classifier evidence
  Final status is `fall` if either pass says fall above the confidence threshold.

### Pipeline Progress
| Module | Status | Files |
|--------|--------|-------|
| M2: YOLOv8 Batch | ✅ | `pipeline/video_reader.py`, `pose_inference.py`; raw-frame YOLO, keypoints undistorted before triangulation |
| M3: Triangulation | ✅ | `pipeline/calibration.py` (B-MCFD/Auvinet calibration), `triangulation.py`; reprojection refinement off by default |
| M4: 3D Features | ✅ | `pipeline/features_3d.py` (10-D, floor-adaptive) |
| M5: 3-Class Classifier | ✅ | `pipeline/classifier.py`, `extract_dataset_3d.py`, `train_3d.py`; best checkpoint `enhanced_3d_20260712_185406` |
| M6: LLM Review | ⚠️ Smoke passed | `pipeline/llm_reviewer.py`, `qa_llm_review.py`; needs broader scenario validation and end-to-end demo test |
| M7: Demo | ✅ | `demo_multiview.py` (OpenCV viz, quality gating, default checkpoint set) |

### Remaining
- Implement the P0 continuous detection client from `Docs/task-queue.md`:
  privacy-preserving 300-frame in-memory rolling buffer, export only 90-frame
  suspected-event clips, continuous YOLO/3D/classifier loop, per-person temporal
  tracking state, suspected/fall trigger policy, and WebSocket event sending.
- Build the P1 local/LAN community monitoring server and dashboard after P0.
- Broader VL validation on more exported 90-frame clips.
- Prepare current-dataset demo commands/scenarios; no separate task-queue item
  for demo materials because the roadshow will use the current dataset directly.
- Update README with current 3D pipeline and commands

### Current Task Queue Decision (2026-07-13)
- `Docs/task-queue.md` was rewritten into P0-P3 checklist form.
- P0 is the continuous detection client, not the dashboard.
- Privacy boundary: keep about 300 synchronized frames in memory, write no
  continuous video, and export only the latest 90 frames when suspected review
  is needed.
- Trigger policy:
  - `0.30 <= p_fall < 0.40` -> suspected; export 90-frame clip and call LLM.
  - If suspected but 90 frames are unavailable, escalate directly to `fall`.
  - `p_fall >= 0.40` -> direct fall alert without LLM.
  - `p_fall < 0.30` -> normal.
- Multi-person tracking should use temporal 3D state to reject unreasonable
  identity jumps and obvious bad detections before classification/events.
- P1 is the open local/LAN WebSocket monitoring dashboard. No auth, TLS, WAN,
  or database for the roadshow.

### P1/P2 Implementation Status (2026-07-13)
- P1 complete for roadshow validation:
  - `monitor_server.py` serves the local monitor at `http://127.0.0.1:8765/`.
  - Detector/demo events are received at `ws://127.0.0.1:8765/ws/events`.
  - Dashboard clients subscribe at `ws://127.0.0.1:8765/ws/dashboard`.
  - Generated clips under `clips/` are served at `/clips/...`.
  - `demo_event_sender.py` sends a sample WebSocket event without running the
    detector.
  - Staff can mark events handled from the dashboard or
    `POST /api/events/{event_id}/handled`.
- P1 smoke tests passed:
  - WebSocket event sender received `ack`.
  - `/api/events` returned the event with `clip_url`.
  - Dashboard HTML returned HTTP 200.
  - Handled endpoint set `handled=true`.
- P2 complete for current roadshow validation:
  - `review_llm_clip.py` now supports optional `--output`.
  - `qa_llm_clip_batch.py` batch-tests clips and reuses
    `LLMFallReviewer.review_video_file()` in real review mode.
  - Dry-run confirmed `clips/s01_1050_90.mp4` decodes, samples 10 frames with
    stride 9, and builds a contact sheet.
  - Real Qwen3-VL-4B review passed on:
    - `clips/s01_1050_90.mp4`: fall, confidence 0.95.
    - `clips/s01_0000_90_normal.mp4`: normal, confidence 0.98.
  - Batch result saved to `results/llm_qa/clip_batch_fall_normal.json`.
- New dependency note:
  - `websockets` is required so uvicorn can serve WebSocket routes; it was
    installed locally and added to `requirements.txt` with FastAPI/uvicorn.

### Safe from Git (large files)
- `features/3d/` — generated .npy for training (.gitignore)
- `checkpoints/enhanced_3d_*/` — new training checkpoints (.gitignore)
- `models/llm/*.gguf` — LLM weights (.gitignore)
- `dataset/mcfd/` — video files (.gitignore)
- `results/debug_pose.png` — debug output (.gitignore)

### Current Best 3D Model
- Checkpoint: `checkpoints/enhanced_3d_20260712_185406/best_model.pth`
- Archive copy: `checkpoints/archive/enhanced_3d_20260712_185406/`
- Test split: 147 windows, fall ratio 0.3673
- AUC-ROC: 0.9747
- Recommended thresholds:
  - `threshold_low = 0.30` → suspected / LLM review lower bound
  - `threshold_high = 0.40` → direct fall threshold
- At threshold 0.40: fall recall 1.000, fall precision 0.818, CM `[[81, 12], [0, 54]]`
- At threshold 0.30: fall recall 1.000, fall precision 0.783, CM `[[78, 15], [0, 54]]`

### Current Extraction/Training Notes
- Default split preset in `extract_dataset_3d.py` is `balanced`:
  - train `[1,2,3,5,6,7,8,9,10,11,12,13,14,15,16,24]`
  - val `[4,17,19,22]`
  - test `[18,20,21,23]`
- Training extraction should NOT use `--pose-fallback`; fallback is for demo/QA only.
- Current recommended extraction command:
  `python extract_dataset_3d.py --device cuda --yolo-batch 32 --time-batch 64 --min-quality 0.15 --max-normal 300`
- Current recommended training command:
  `python train_3d.py --seed 42 --no-smote`
- Current threshold sweep command:
  `python sweep_threshold.py --split test --device cuda`
- Demo command:
  `python demo_multiview.py --scenario 1`
- LLM smoke test command:
  `python qa_llm_review.py --scenario 1 --kind both --frames 1 --view-mode collage`

### Current LLM Smoke Result
- Model: Qwen3-VL-4B GGUF + mmproj through `Qwen25VLChatHandler`
- Input format: one recent three-camera collage frame; demo additionally runs a second assisted pass with pose overlays and structured evidence
- Scenario 1 normal sample: `is_fall=false`, reason says walking normally
- Scenario 1 fall sample: `is_fall=true`, confidence 0.95, reason says lying horizontally on white crash mat
- Demo integration uses `--llm-frames 1` and `--llm-min-confidence 0.60`; low-confidence false results keep status as `suspected`

### LLM Clip Review Boundary
- Production camera/buffer layer should export a short event clip, e.g. 90 frames at 30fps.
- LLM layer receives that clip file and does not own rolling-buffer storage.
- Default review policy: sample every 9th frame, cap at 10 frames, pack into one contact-sheet timeline image.
- Verified smoke result on `clips/s01_1050_90.mp4`: `is_fall=true`, confidence 0.95, ~12s vision processing on current machine.
- Review command for an exported clip:
  `python review_llm_clip.py --video clips/event_001.mp4`
- QA-only MCFD clip export command:
  `python export_mcfd_clip.py --scenario 1 --start-frame 1050 --frames 90 --output clips/s01_1050_90.mp4`

---

## Target State (After 20-Day Upgrade)

### 7 Modules to Build/Upgrade

1. **Module 1 — Dataset**: Two independent validation lines
   - Line A (3D accuracy): Multi-view dataset → triangulation → MPJPE validation (Skip NTU, use alternative)
   - Line B (fall detection): Reuse existing UR Fall Dataset pipeline

2. **Module 2 — YOLOv8n-pose Batch Inference**: 3-view batch input, 640×640 resize

3. **Module 3 — Triangulation**: Camera calibration + multi-view triangulation + occlusion handling + Kalman smoothing

4. **Module 4 — 3D Feature Engineering**: Convert existing 10 features from 2D→3D

5. **Module 5 — 3-Class Classifier**: Upgrade binary → 3-class (normal/suspected/fall), configurable thresholds

6. **Module 6 — LLM Review**: Qwen3-VL-8B Q4_K_M via llama-cpp-python, trigger on "suspected"

7. **Module 7 — Demo Visualization**: Web UI or OpenCV window showing full pipeline

### Key Architecture Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Multi-view dataset | **Multiple Cameras Fall Dataset (MCFD)** | 8 cameras, 24 scenarios (22 fall+confound, 2 confound only), freely downloadable via Kaggle/Baidu |
| 3D ground truth | **None — manual visual validation** | Lucas will manually inspect reconstructed 3D skeletons for correctness; no automated MPJPE needed |
| Camera calibration | **Synthetic/manual parameters** | MCFD has no calibration files; define 3 virtual cameras in code (e.g. -45°/0°/+45°, same height) |
| LLM model | Qwen3-VL-8B Q4_K_M primary; Qwen3-VL-4B fallback | 8B on 8GB tight (~7.8GB), requires layer offloading; 4B safe at ~3.6GB |
| LLM framework | llama-cpp-python | Supports Qwen3-VL GGUF + mmproj, CUDA acceleration |
| Visualization | Gradio web app (preferred) or OpenCV | Quick to build; shows all pipeline stages |

---

## Hardware Constraints & Mitigations

### RTX 3060 Ti 8GB VRAM Budget
| Component | Size | Notes |
|-----------|------|-------|
| YOLOv8n-pose | ~30 MB | Trivial, runs on GPU no problem |
| BiLSTM+BiGRU | ~17 MB | CPU inference is fine, GPU even better |
| Qwen3-VL-8B Q4_K_M GGUF | ~5.4 GB | Language model weights |
| Qwen3-VL mmproj (vision encoder) | ~1.7 GB | Loaded during image encoding |
| KV Cache (short context) | ~0.2 GB | 1-3 frames, ~500 tokens |
| CUDA overhead | ~0.5 GB | Driver + framework |
| **Total** | **~7.8 GB** | ⚠️ Very tight — requires layer offloading |

**Mitigation**: Set `n_gpu_layers` to offload ~5-10 layers to CPU, or use Qwen3-VL-4B (~3.6GB total, fits easily).

### Real-time 3-Camera Feasibility
- The viable design is event-driven: YOLO + 3D classifier run continuously; VL review runs only on suspected windows.
- For a 16GB demo target, keep the VL model at 4B, avoid concurrent LLM calls, and keep YOLO at `yolov8n-pose`.
- This should fit a 5070 Ti-class machine for roadshow use if suspected triggers are rare; it is not a production high-throughput design.

---

## Dataset Strategy

### Multi-View Input — MCFD (唯一多视角数据集)
- **Multiple Cameras Fall Dataset (MCFD)** — 24 scenarios, 8 IP cameras
  - Kaggle: https://www.kaggle.com/datasets/soumicksarker/multiple-cameras-fall-dataset
  - 百度AI Studio: https://aistudio.baidu.com/aistudio/datasetdetail/63166
  - 官网: https://www.iro.umontreal.ca/~labimage/Dataset/ (有反爬保护)
  - 22 scenarios contain fall + confounding events; 2 scenarios confounding only
  - **无相机标定参数** → 使用代码中手动定义3个虚拟相机的投影矩阵
  - **无3D骨骼真值** → 人工目视验证重建效果

### Camera Calibration — 合成参数
- MCFD 无标定文件，在代码中定义3台虚拟相机：
  - 同一高度，水平角度 -45°/0°/+45°
  - 假设分辨率为原始视频分辨率
  - 焦距和主点使用合理假设值（如 f=1000, cx=w/2, cy=h/2）
- 标记为 demo 简化处理，非真实标定

### 3D Validation — 人工目视
- Lucas 手动观察重建出的3D骨架是否合理（关节不飞散、运动轨迹连续）
- 不做自动化 MPJPE 计算
- 评估报告中诚实标注"人工目视验证"

### Validation Line B — Fall Classification
- Reuse existing UR Fall Dataset pipeline (already working)
- Optional: Extract GMDCSA24 features if time permits

---

## Risk Register

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| MCFD lacks camera calibration | **Confirmed** | Low | Use synthetic camera params; document as demo simplification |
| MCFD lacks 3D ground truth | **Confirmed** | Low | Manual visual validation by Lucas; document honestly |
| Qwen3-VL-8B OOM on 8GB | Medium | Medium | Use Qwen3-VL-4B; document 8B as "production upgrade" |
| llama-cpp-python vision API changes | Low | Medium | Pin version; test with simple image first |
| Triangulation visually unconvincing | Low | Medium | Tune synthetic camera params; acceptable for demo |
| 20-day timeline too tight | Low | High | Priority ordering ensures core pipeline works even if polish is minimal |

---

## Collaboration Protocol

- **Agent role**: Code architecture, implementation, documentation, research
- **Lucas role**: Editor wiring, testing, dataset download/management, demo rehearsal
- **Status model**: documented → ported → wired → tested → parity (from Wanderer Tales project)
- **Communication**: Structured numbered items (A.1, B.1), brief verification checklist per session
- **This file**: `Docs/Chat.md` is the agent's external memory; read at session start

---

## Quick Reference: Key URLs

- Prompt spec: `Docs/prompt.md`
- MCFD Kaggle: https://www.kaggle.com/datasets/soumicksarker/multiple-cameras-fall-dataset
- MCFD 百度AI Studio: https://aistudio.baidu.com/aistudio/datasetdetail/63166
- MCFD 官网: https://www.iro.umontreal.ca/~labimage/Dataset/
- Qwen3-VL-8B GGUF: https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct-GGUF
- Qwen3-VL-4B GGUF: https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct-GGUF (fallback)
- UR Fall Dataset: https://fenix.ur.edu.pl/~mkepski/ds/uf.html (already have cam0)
- llama-cpp-python: https://github.com/abetlen/llama-cpp-python
