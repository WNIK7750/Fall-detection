# Task Queue

This file records the next work after context compaction. Do not push or commit unless the user asks.

## P0 - 40-Frame Classifier Data Policy

- [x] Update `scripts/training/extract_dataset_3d.py` to generate explicit 40-frame samples.
- [x] Use cam1 as the reference timeline and rely on `MCFDReader`/CSV-derived offsets for cam4/cam7 synchronization.
- [x] Build legal fall candidates first, then sample from them:
  - window length: 40 consecutive real-time frames
  - fall label: at least 25 consecutive frames overlap a fall interval
  - random selection is allowed only among legal candidates
- [x] Build legal normal candidates first, then sample from them:
  - window length: 40 consecutive real-time frames
  - normal label: 0 frames overlap any fall interval
  - keep a safety gap from fall intervals where possible
- [x] Save features as `(N, 10, 40)` and labels as binary `normal/fall`.
- [x] Compile-check extraction code.

Validation:

```powershell
python -m compileall scripts\training\extract_dataset_3d.py pipeline\features_3d.py
python scripts\training\extract_dataset_3d.py --device cuda --yolo-batch 32 --time-batch 64 --min-quality 0.15 --max-normal 300
```

Done when:

- New `features/3d/{train,val,test}/X.npy` arrays have `shape[1:] == (10, 40)`.
- Logs show fall/normal window counts per split.
- No fall-labeled sample is created from a window with less than 25 consecutive fall frames.
- No normal-labeled sample overlaps fall frames.

## P0 - 40-Frame Realtime Inference

- [x] Update `continuous_event_client.py` defaults:
  - `classifier_window_steps=40`
  - `classifier_window_subsample=1` or remove subsampling from the realtime classifier path
  - `PersonState.features` retains at least 40 frames
- [x] Feed the classifier the latest 40 consecutive feature frames.
- [x] Keep existing delayed evidence policy: 90 pre/trigger frames + 151 post frames = 241-frame/8-second clip.
- [x] Compile-check client code.

Validation:

```powershell
python -m compileall continuous_event_client.py
python continuous_event_client.py --scenario 3 --start-frame 303 --max-frames 520 --no-send --no-llm --stop-after-event
```

Done when:

- Client runs without shape errors using a 40-frame checkpoint.
- Triggered events still wait until post-context is available before final clip/LLM update.

## P0 - Retrain And Threshold Sweep

- [x] Confirm `models/enhanced_lstm.py` accepts sequence length 40 without shape assumptions.
- [x] Retrain after regenerating features.
- [x] Sweep thresholds on the new checkpoint.
- [x] Pick a roadshow threshold favoring recall while controlling obvious false positives.

Commands:

```powershell
python scripts\training\train_3d.py --seed 20260713 --no-smote
python scripts\training\sweep_threshold.py --checkpoint checkpoints\<new_run>\best_model.pth --split test --device cuda
```

Done when:

- New checkpoint path is recorded.
- Test/val metrics are copied into README and CLAUDE.
- `continuous_event_client.py --checkpoint checkpoints\<new_run>\best_model.pth ...` works.

Completed result:

- Checkpoint: `checkpoints/archive/enhanced_3d_20260713_142749/best_model.pth`
- Extraction/training seed: `20260713`, no SMOTE
- Test AUC: `0.8030`
- Threshold `0.075`: fall precision `0.392`, recall `1.000`
- Threshold `0.40`: fall precision `0.702`, recall `0.698`
- Chute03 realtime smoke triggered at frame `630` within labeled interval `603-633`

## P1 - Documentation And Runbook Refresh

- [ ] Update README current-state text after the new checkpoint exists.
- [ ] Update CLAUDE current checkpoint and metrics after retraining.
- [ ] Replace old smoke commands that assume 90-frame clips or 20-step classifier windows.
- [ ] Document that MCFD AVI metadata reports 120 FPS but the project treats the footage as real 30 FPS.
- [ ] Keep dataset/model download links and local-artifact guidance current.

Done when:

- A new reader can run extraction, training, threshold sweep, and roadshow from README alone.
- CLAUDE accurately reflects the active checkpoint and classifier window policy.

## P1 - Roadshow End-To-End QA

- [ ] Run `.\demo.ps1 -NoLLM` and verify dashboard behavior.
- [ ] Run `.\demo.ps1` with Qwen3-VL if GPU llama-cpp is healthy.
- [ ] Verify suspected initial alert appears without video.
- [ ] Verify final update includes 241-frame/8-second clip metadata.
- [ ] Verify released video opens in the modal player with play/pause and progress bar.
- [ ] Verify Fall-column and Suspected-column handle/request countdown behavior.

Done when:

- One clean roadshow pass works without forced probabilities.
- At least one event reaches final state with the same `event_id`.

## P2 - LLM Stability And GPU Path

- [ ] Re-test Qwen3-VL with the three-view contact-sheet path.
- [ ] If access violations return, reduce `--n-gpu-layers` and document the working value.
- [ ] Keep `llama-cpp-python` CUDA verification command in README.
- [ ] Add a small QA command for reviewing one generated 241-frame clip.

Done when:

- LLM review runs on GPU or the fallback/no-LLM roadshow path is clearly documented.
- Failures report as `suspected` instead of blocking the detector loop.

## P2 - Dataset QA Tools

- [ ] Add or extend a script to print candidate windows and fall-overlap counts.
- [ ] Add a quick inspection mode for one scenario/camera set.
- [ ] Optionally export a few sampled fall/normal clips for visual sanity checks.

Done when:

- It is easy to prove that fall samples contain >=25 consecutive fall frames.
- It is easy to prove normal samples contain 0 fall frames.

## P3 - Future Product Work

- [ ] Replace MCFD calibration approximations with real camera calibration for deployment.
- [ ] Add real camera/RTSP adapters.
- [ ] Add persistent event storage, auth, TLS, audit log, and retention/deletion policy.
- [ ] Consider a third `risk/pre_fall` training label only after reliable manual annotation rules exist.
- [ ] Consider smoother multi-person association across views if roadshow scenarios require it.
