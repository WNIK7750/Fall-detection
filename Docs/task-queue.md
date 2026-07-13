# Task Queue

Last updated: 2026-07-13

Legend: `( )` pending, `(o)` done.

## P0 - Continuous Detection Client

(o) Define the local event payload schema for client-to-monitor WebSocket events.

(o) Add a privacy-preserving rolling video buffer in the camera/client layer:
- Keep about 300 recent synchronized frames in memory.
- Do not write video continuously to disk.
- Export only the latest 90 frames when a suspected event needs LLM review.
- If 90 frames are not available at trigger time, escalate directly to `fall`.

(o) Keep three camera streams running continuously through YOLOv8n-pose batch inference.

(o) Keep the 3D reconstruction and fall classifier running continuously over tracked people.

(o) Add per-person temporal state for multi-person tracking:
- Maintain recent 2D detections, 3D skeletons, feature windows, quality scores, and status.
- Use prior 3D position and timestamp to reject physically unreasonable identity jumps.
- Reject obvious unstable detections such as sudden person-to-object switches or skeletons with poor quality.

(o) Implement suspected/fall trigger policy:
- `0.30 <= p_fall < 0.40` -> suspected, request clip review.
- `p_fall >= 0.40` -> direct fall alert.
- `p_fall < 0.30` -> normal, no event.
- Add cooldown/deduplication so one fall does not spam repeated alerts.

(o) On suspected trigger, pass the trigger timestamp/frame index back to the main loop.

(o) Export the 90-frame event clip around the trigger from the rolling buffer.

(o) Call the LLM clip-review interface only for suspected events with an available clip.

(o) Send the final event JSON over WebSocket:
- direct classifier fall
- LLM-confirmed fall
- unresolved suspected event if the LLM cannot decide or errors

(o) Add a runnable dataset-backed client command using current MCFD videos.

(o) Smoke test the continuous client without LLM using a current scenario.

(o) Smoke test the continuous client with LLM on at least one suspected/fall clip.

## P1 - Community Monitoring Demo

(o) Build a local/LAN WebSocket monitoring server.

(o) Build a simple dashboard page for community staff.

(o) Show incoming events with:
- client/device ID
- address
- timestamp
- status: `suspected` / `fall`
- 3D classifier probability
- Qwen3-VL review result
- short event video preview

(o) Add a staff action to mark an event as handled.

(o) Add a demo sender that can send a sample event without running the detector.

(o) Serve generated event clips locally so the dashboard can preview them.

(o) Keep the server open/local only for roadshow use: no auth, TLS, WAN deployment, or database.

(o) Smoke test dashboard event receive, video preview, and handled state.

## P2 - LLM Clip Review Interface

(o) Keep the clip-review boundary fixed: upstream exports a short clip, LLM module only reviews that clip.

(o) Use the current default review policy:
- upstream exports 90 frames
- `review_llm_clip.py` samples every 9th frame
- max 10 sampled frames
- sampled frames are packed into one contact-sheet timeline image

(o) Preserve the smoke-tested command:
`python review_llm_clip.py --video clips/s01_1050_90.mp4`

(o) Reuse `LLMFallReviewer.review_video_file()` from the continuous event client.

(o) Test clip review on several current-dataset fall and normal clips.

(o) Record recommended scenarios and commands for the roadshow.

## P3 - Polish And Documentation

(o) Update `README.md` with a concise Roadshow Demo section:
- 3D extraction/train/eval commands
- current best checkpoint
- threshold policy: low `0.30`, high `0.40`
- LLM clip review command
- community dashboard run command
- continuous client run command

(o) Update product/demo boundary notes:
- MCFD calibration is demo/synthetic or dataset-derived approximation.
- Current demo uses local dataset videos, not real deployed cameras.
- No encryption/login/database in the roadshow monitoring interface.
- Event clips are exported only on trigger as the privacy-preserving product mechanism.

(o) Update `Docs/Chat.md` after implementation milestones.

(o) Confirm generated clips remain ignored by Git.

(o) Confirm LLM weights, MCFD videos, and generated features remain ignored by Git.

(o) Run a final end-to-end roadshow rehearsal:
- normal monitoring state
- suspected trigger
- 90-frame clip export
- Qwen3-VL review
- WebSocket dashboard alert
- staff marks event as handled
