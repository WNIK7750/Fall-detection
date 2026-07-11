# CLAUDE.md — Fall Detection Project

> Project-level AI instruction file. Read this first when working on this codebase.

---

## ⚠️ CRITICAL: External Memory

**This project's agent memory is stored in `Docs/Chat.md`.**  
Always read `Docs/Chat.md` at the start of each session. It contains:
- Current project state and progress
- Architecture decisions and their rationale
- Hardware constraints (RTX 3060 Ti 8GB)
- Dataset strategy and known issues
- Risk register and mitigations
- Collaboration protocol with Lucas

**After any meaningful progress, update `Docs/Chat.md`** to reflect new state. Replace outdated information; do not append journal entries.

---

## Project Overview

**老年人跌倒检测系统** — Multi-view 3D fall detection system for elderly care roadshow demo.

- **Timeline**: 20 days (2026-07-10 → 2026-07-30)
- **Hardware target**: RTX 3060 Ti 8GB VRAM
- **Primary spec**: `Docs/prompt.md` (all requirements, architecture, acceptance criteria)
- **Current baseline**: Working 2D single-view system (UR Fall Dataset, BiLSTM+BiGRU, Precision 75%/Recall 83%) → Upgrading to 3-view 3D + LLM review

---

## Architecture (Target)

```
3-view Video Input → YOLOv8n-pose Batch → 2D Keypoints
    → Multi-view Triangulation → 3D Skeleton
    → 3D Feature Engineering (10-dim) → BiLSTM+BiGRU Classifier
    → 3-Class Output (Normal/Suspected/Fall)
        ├─ Normal → No action
        ├─ Fall (high confidence) → Alert
        └─ Suspected → Qwen3-VL-8B Review → Final decision
```

7 modules in `Docs/prompt.md` §二: Dataset, YOLOv8, Triangulation, Features, Classifier, LLM, Visualization.

---

## Key Files

| File/Dir | Purpose |
|-----------|---------|
| `Docs/prompt.md` | Complete requirements spec (187 lines) |
| `Docs/Chat.md` | **Agent external memory — read first** |
| `ensemble_inference.py` | Current 2D inference entry point |
| `demo_video.py` | Current real-time demo |
| `train_enhanced.py` | Training script (BiLSTM+BiGRU) |
| `models/enhanced_lstm.py` | Model architecture definitions |
| `extract_features_v3.py` | Feature extraction (MediaPipe) |
| `extract_gmdcsa24.py` | Feature extraction (YOLOv8 batch) |
| `yolov8n-pose.pt` | YOLOv8 pose weights |
| `checkpoints/` | Trained model weights (8 enhanced runs) |
| `features/` | Extracted .npy features (UR Fall) |
| `dataset/` | UR Fall video split lists |
| `requirements.txt` | Python dependencies |

---

## Commands

```bash
# Evaluate current 2D ensemble model
python ensemble_inference.py --eval

# Real-time demo on a video
python demo_video.py

# Train enhanced model
python train_enhanced.py

# Extract features from UR Fall videos (YOLOv8)
python extract_gmdcsa24.py
```

---

## Constraints & Rules

1. **Honesty requirement** (from prompt.md §四): Never claim "99.9% accuracy". All metrics must state test conditions and dataset. Clearly separate "architecture design" from "current demo implementation".

2. **Demo scope** (from prompt.md §背景): Don't over-engineer. No real cameras, no production security, no physical calibration rig. Use dataset video files as input.

3. **3-class output**: Normal / Suspected / Fall (not binary). Thresholds must be configurable, not hardcoded.

4. **Two independent validation lines**: Line A (3D reconstruction MPJPE) and Line B (fall classification metrics) are validated separately. Acknowledge this limitation honestly.

5. **VRAM budget**: 8GB total. Qwen3-VL-8B Q4_K_M (~5.4GB GGUF + ~1.7GB mmproj) is tight. Have Qwen3-VL-4B fallback ready.

6. **Dual-model ensemble** (current system): Model A (high precision) + Model B (high recall), soft voting 0.5+0.5, threshold 0.70.

---

## External Memory Protocol

1. **Session start**: Read `Docs/Chat.md`
2. **Session end**: Update `Docs/Chat.md` with progress, decisions, blockers
3. **Key decisions**: Record rationale in `Docs/Chat.md`
4. **New files created**: Note in `Docs/Chat.md` under relevant module section
5. **Status changes**: Update module status in `Docs/Chat.md`
