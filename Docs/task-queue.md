# Task Queue

Last updated: 2026-07-12

## Roadshow Must-Have

### 1. Community Monitoring Demo
- Build a local WebSocket server for multiple client connections.
- Build a simple dashboard page for community staff.
- Show incoming fall events with:
  - client/device ID
  - address
  - timestamp
  - status: `suspected` / `fall`
  - 3D classifier probability
  - Qwen3-VL review result
  - short event video preview
- Add a demo client that sends a sample event over WebSocket.
- Keep this local/LAN only for now; no public WAN, auth, or TCP protocol work.

### 2. Client Event Loop Integration
- Define the client-side event payload schema.
- Wire the real detection loop to:
  - maintain or receive a short rolling event clip from the camera layer
  - run 3D classifier continuously
  - trigger on `0.30 <= p_fall < 0.40`
  - call LLM clip review
  - send event JSON to the monitoring server
- Keep camera rolling-buffer/export logic outside the LLM module.

### 3. LLM Clip Review Interface
- Current default is fixed:
  - upstream exports a 90-frame short clip
  - `review_llm_clip.py` samples every 9th frame
  - max 10 sampled frames
  - sampled frames are packed into one contact-sheet timeline image
- Smoke-tested command:
  `python review_llm_clip.py --video clips/s01_1050_90.mp4`
- Need test on several fall and normal clips before demo.

### 4. Demo Materials
- Prepare 2-3 fall clips and 1-2 normal clips.
- Prepare one scripted roadshow flow:
  - normal monitoring state
  - suspected trigger
  - Qwen3-VL review
  - community dashboard alert
  - staff marks event as handled
- Keep commands short and repeatable.

## Important Polish

### 5. Documentation Cleanup
- README is still mostly the old 2D project document.
- Add a concise “Roadshow Demo” section with:
  - 3D extraction/train/eval commands
  - current best checkpoint
  - threshold policy: low `0.30`, high `0.40`
  - LLM clip review command
  - community dashboard run command after it exists
- Keep honesty notes: MCFD calibration and dataset limitations.

### 6. Scenario Validation
- Validate LLM clip review on more than scenario 1.
- Validate classifier demo behavior on selected MCFD scenarios.
- Check for chair/background false positives after keeping YOLO strict in demo.
- Save a short list of recommended scenarios for live presentation.

### 7. Git/Artifact Hygiene
- Confirm only intended checkpoint archive files are tracked.
- Keep generated clips under `clips/` and out of Git.
- Keep LLM weights, MCFD videos, and generated features ignored.
- Commit current stable state before larger frontend/server work.

## Optional Enhancements

### 8. Better LLM Evidence
- Add optional structured evidence to clip review:
  - `p_fall`
  - skeleton quality
  - torso angle
  - COM height / velocity
- Keep visual review independent enough that LLM can catch YOLO misses.

### 9. Real Camera Adapter
- Add camera-source abstraction after dashboard works:
  - file input
  - webcam index
  - RTSP URL
- Keep this behind a narrow interface so demo dataset code remains usable.

### 10. Performance Tuning
- Measure YOLO + 3D classifier FPS on target GPU.
- Measure Qwen3-VL clip review latency with contact sheet defaults.
- If LLM latency is still high, test fewer sampled frames:
  - stride 12, max 8
  - stride 15, max 6

## Later Productization

### 11. Security and Deployment
- Authentication for clients and staff dashboard.
- TLS/WSS.
- Event persistence database.
- File upload endpoint instead of local shared paths.
- Retention policy for short clips.

### 12. Reliability
- Client reconnect/backoff.
- Server-side heartbeats.
- Offline queue for events.
- Duplicate event suppression.
- Event state machine and audit log.
