"""
No-camera playmode for demonstrating the P0 event loop.

This does not require physical cameras, YOLO, 3D reconstruction, or classifier
weights. It replays an existing clip or generates synthetic frames, keeps the
same privacy-preserving rolling buffer, exports only on trigger, optionally
runs the LLM clip reviewer, and sends the same event payload to the monitor.

Run monitor first:
    python monitor_server.py --host 127.0.0.1 --port 8765

Predictable dashboard demo:
    python playmode_event_loop.py --source-video clips/s01_1050_90.mp4 --server-url ws://127.0.0.1:8765/ws/events

No server smoke:
    python playmode_event_loop.py --no-send --no-llm --max-frames 120
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

import cv2
import numpy as np

from pipeline.clip_buffer import RollingClipBuffer
from pipeline.event_schema import build_event_payload
from pipeline.event_sender import send_event
from pipeline.llm_reviewer import LLMFallReviewer


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = BASE_DIR / "models" / "llm" / "Qwen3VL-4B-Instruct-Q4_K_M.gguf"
DEFAULT_MMPROJ = BASE_DIR / "models" / "llm" / "mmproj-Qwen3VL-4B-Instruct-Q8_0.gguf"


def iter_video_or_synthetic(source_video: str | None, max_frames: int, size: tuple[int, int]):
    if source_video and Path(source_video).exists():
        cap = cv2.VideoCapture(source_video)
        frames: list[np.ndarray] = []
        try:
            while len(frames) < max_frames:
                ok, frame = cap.read()
                if not ok:
                    break
                frames.append(frame)
        finally:
            cap.release()
        if frames:
            i = 0
            while i < max_frames:
                yield frames[i % len(frames)].copy()
                i += 1
            return

    width, height = size
    for i in range(max_frames):
        frame = np.full((height, width, 3), (245, 247, 251), dtype=np.uint8)
        cv2.rectangle(frame, (40, 40), (width - 40, height - 40), (218, 225, 235), 2)
        x = 90 + min(i, max_frames - 1) * max(1, (width - 180) // max(max_frames, 1))
        if i < max_frames * 0.62:
            cv2.circle(frame, (x, 140), 24, (37, 99, 235), -1)
            cv2.line(frame, (x, 164), (x, 270), (37, 99, 235), 6)
            cv2.line(frame, (x, 205), (x - 45, 250), (37, 99, 235), 5)
            cv2.line(frame, (x, 205), (x + 45, 250), (37, 99, 235), 5)
            cv2.line(frame, (x, 270), (x - 38, 340), (37, 99, 235), 5)
            cv2.line(frame, (x, 270), (x + 38, 340), (37, 99, 235), 5)
            label = "normal walking"
        else:
            y = 320
            cv2.circle(frame, (x - 70, y), 24, (194, 65, 12), -1)
            cv2.line(frame, (x - 45, y), (x + 95, y + 8), (194, 65, 12), 8)
            cv2.line(frame, (x + 15, y + 4), (x + 85, y - 32), (194, 65, 12), 5)
            cv2.line(frame, (x + 15, y + 4), (x + 92, y + 42), (194, 65, 12), 5)
            label = "suspected fall posture"
        cv2.putText(frame, f"PLAYMODE frame {i:03d} - {label}", (28, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (31, 41, 55), 2, cv2.LINE_AA)
        yield frame


def make_event_clip_path(output_dir: str, status: str) -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return Path(output_dir) / f"playmode_{status}_{stamp}.mp4"


def load_reviewer(args: argparse.Namespace) -> LLMFallReviewer | None:
    if not args.llm:
        return None
    reviewer = LLMFallReviewer(
        model_path=args.model,
        mmproj_path=args.mmproj,
        n_ctx=args.n_ctx,
        n_gpu_layers=args.n_gpu_layers,
    )
    if not reviewer.ready:
        raise RuntimeError("LLM reviewer failed to load")
    return reviewer


def review_clip(reviewer: LLMFallReviewer | None, clip_path: str, args: argparse.Namespace) -> dict[str, Any]:
    if reviewer is None:
        return {
            "status": "suspected",
            "is_fall": None,
            "confidence": 0.0,
            "reason": "Playmode LLM disabled; showing unresolved suspected workflow.",
        }
    result = reviewer.review_video_file(
        video_path=clip_path,
        frame_stride=args.llm_frame_stride,
        max_frames=args.llm_max_frames,
        contact_sheet=True,
    )
    if result.get("is_fall") is True and float(result.get("confidence", 0.0)) >= args.llm_min_confidence:
        status = "fall"
    elif result.get("is_fall") is False and float(result.get("confidence", 0.0)) >= args.llm_min_confidence:
        status = "normal"
    else:
        status = "suspected"
    return {
        "status": status,
        "is_fall": result.get("is_fall"),
        "confidence": result.get("confidence"),
        "reason": result.get("reason", ""),
        "sampled_frames": result.get("sampled_frames"),
    }


def send_or_print(payload: dict[str, Any], args: argparse.Namespace) -> None:
    if args.no_send:
        print("[playmode:event]", json.dumps(payload, ensure_ascii=False))
        return
    ack = send_event(args.server_url, payload)
    print(f"[playmode:sent] ack={ack}")


def run_playmode(args: argparse.Namespace) -> None:
    reviewer = load_reviewer(args)
    buffer = RollingClipBuffer(max_frames=args.buffer_frames, fps=args.fps)
    trigger_sent = False
    p_fall = 0.35 if args.mode == "suspected" else 0.42
    initial_status = "suspected" if p_fall < args.threshold_high else "fall"

    Path(args.clip_dir).mkdir(parents=True, exist_ok=True)
    print(
        f"[playmode] mode={args.mode} source={args.source_video or 'synthetic'} "
        f"trigger_frame={args.trigger_frame} buffer={args.buffer_frames}"
    )

    for frame_index, frame in enumerate(iter_video_or_synthetic(args.source_video, args.max_frames, (1152, 384))):
        buffer.append(frame_index, frame)
        if args.realtime:
            time.sleep(max(0.0, 1.0 / args.fps))

        if trigger_sent or frame_index < args.trigger_frame:
            continue

        clip_path: str | None = None
        final_status = initial_status
        event_id = f"event_{time.strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:6]}"
        if initial_status == "suspected":
            initial_payload = build_event_payload(
                client_id=args.client_id,
                device_id=args.device_id,
                address=args.address,
                status="suspected",
                p_fall=p_fall,
                threshold_low=args.threshold_low,
                threshold_high=args.threshold_high,
                clip_path=None,
                scenario=None,
                frame_index=frame_index,
                track_id=0,
                quality=0.88,
                llm_review={
                    "status": "pending_review",
                    "is_fall": None,
                    "confidence": None,
                    "reason": "Suspected fall detected by scripted playmode; preparing 90-frame clip for LLM review.",
                },
                trigger_reason=f"playmode scripted {initial_status} at frame {frame_index}",
                event_id=event_id,
            )
            send_or_print(initial_payload, args)
            output = make_event_clip_path(args.clip_dir, "suspected")
            clip_path = buffer.export_latest(output, frames=args.event_clip_frames, fps=args.fps)
            if clip_path is None:
                final_status = "fall"
                llm_review = {
                    "status": "fall",
                    "is_fall": True,
                    "confidence": None,
                    "reason": "Less than 90 buffered frames available; escalated by policy.",
                }
            else:
                llm_review = review_clip(reviewer, clip_path, args)
                final_status = llm_review.get("status", "suspected")
        else:
            output = make_event_clip_path(args.clip_dir, "fall")
            clip_path = buffer.export_latest(output, frames=min(args.event_clip_frames, len(buffer)), fps=args.fps)
            llm_review = {
                "status": "skipped_direct_fall",
                "is_fall": True,
                "confidence": None,
                "reason": "Playmode high-threshold direct fall; LLM skipped.",
            }

        payload = build_event_payload(
            client_id=args.client_id,
            device_id=args.device_id,
            address=args.address,
            status=final_status,
            p_fall=p_fall,
            threshold_low=args.threshold_low,
            threshold_high=args.threshold_high,
            clip_path=clip_path,
            scenario=None,
            frame_index=frame_index,
            track_id=0,
            quality=0.88,
            llm_review=llm_review,
            trigger_reason=f"playmode scripted {initial_status} at frame {frame_index}",
            event_id=event_id,
        )
        send_or_print(payload, args)
        trigger_sent = True
        if args.stop_after_event:
            break

    if not trigger_sent:
        print("[playmode] completed without trigger; increase --max-frames or lower --trigger-frame")


def main() -> None:
    parser = argparse.ArgumentParser(description="No-camera event-loop playmode")
    parser.add_argument("--source-video", default="clips/s01_1050_90.mp4")
    parser.add_argument("--mode", choices=["suspected", "direct-fall"], default="suspected")
    parser.add_argument("--max-frames", type=int, default=150)
    parser.add_argument("--trigger-frame", type=int, default=100)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--buffer-frames", type=int, default=300)
    parser.add_argument("--event-clip-frames", type=int, default=90)
    parser.add_argument("--threshold-low", type=float, default=0.30)
    parser.add_argument("--threshold-high", type=float, default=0.40)
    parser.add_argument("--clip-dir", default="clips")
    parser.add_argument("--server-url", default="ws://127.0.0.1:8765/ws/events")
    parser.add_argument("--client-id", default="playmode-client-01")
    parser.add_argument("--device-id", default="no-camera-playmode")
    parser.add_argument("--address", default="Demo Community / playmode loop")
    parser.add_argument("--llm", dest="llm", action="store_true",
                        help="Enable LLM review for suspected events (default)")
    parser.add_argument("--no-llm", dest="llm", action="store_false",
                        help="Disable LLM review for fast smoke tests")
    parser.set_defaults(llm=True)
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--mmproj", default=str(DEFAULT_MMPROJ))
    parser.add_argument("--n-gpu-layers", type=int, default=-1)
    parser.add_argument("--n-ctx", type=int, default=4096)
    parser.add_argument("--llm-frame-stride", type=int, default=9)
    parser.add_argument("--llm-max-frames", type=int, default=10)
    parser.add_argument("--llm-min-confidence", type=float, default=0.60)
    parser.add_argument("--no-send", action="store_true")
    parser.add_argument("--realtime", action="store_true")
    parser.add_argument("--stop-after-event", action="store_true", default=True)
    args = parser.parse_args()
    run_playmode(args)


if __name__ == "__main__":
    main()
