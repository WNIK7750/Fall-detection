"""
Continuous dataset-backed fall detection event client.

This is the P0 client loop: it uses MCFD videos as three virtual cameras,
keeps YOLO/3D/classifier running continuously, exports a short event clip only
when needed, optionally runs Qwen3-VL clip review, and sends a JSON event to the
local monitor server.

Run monitor first:
    python monitor_server.py --host 127.0.0.1 --port 8765

Fast smoke without LLM / without sending:
    python continuous_event_client.py --scenario 1 --max-frames 180 --force-trigger-frame 120 --no-send --no-llm

Roadshow-style event to dashboard:
    python continuous_event_client.py --scenario 1 --server-url ws://127.0.0.1:8765/ws/events --force-trigger-frame 120
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

import cv2
import numpy as np

from pipeline.calibration import get_projections, undistort_keypoints
from pipeline.classifier import FallClassifier, HeuristicClassifier
from pipeline.clip_buffer import RollingClipBuffer
from pipeline.event_schema import build_event_payload
from pipeline.event_sender import send_event
from pipeline.features_3d import FloorEstimator, compute_features
from pipeline.llm_reviewer import LLMFallReviewer, make_multiview_collage
from pipeline.pose_inference import MultiPersonTracker, PoseDetector
from pipeline.triangulation import KalmanSmoother, Triangulator
from pipeline.video_reader import MCFDReader


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_3D_CHECKPOINT = BASE_DIR / "checkpoints" / "archive" / "enhanced_3d_20260712_185406" / "best_model.pth"
DEFAULT_MODEL = BASE_DIR / "models" / "llm" / "Qwen3VL-4B-Instruct-Q4_K_M.gguf"
DEFAULT_MMPROJ = BASE_DIR / "models" / "llm" / "mmproj-Qwen3VL-4B-Instruct-Q8_0.gguf"


@dataclass
class PersonState:
    features: deque[np.ndarray] = field(default_factory=lambda: deque(maxlen=60))
    qualities: deque[dict[str, Any]] = field(default_factory=lambda: deque(maxlen=60))
    skeletons: deque[np.ndarray] = field(default_factory=lambda: deque(maxlen=60))
    centroids: deque[np.ndarray] = field(default_factory=lambda: deque(maxlen=30))
    speeds: deque[float] = field(default_factory=lambda: deque(maxlen=30))
    floor_estimator: FloorEstimator = field(default_factory=FloorEstimator)
    smoother: KalmanSmoother = field(default_factory=KalmanSmoother)
    status: str = "normal"
    last_event_frame: int = -10_000
    rejected_frames: int = 0


def skeleton_centroid(skeleton: np.ndarray) -> np.ndarray | None:
    valid = skeleton[~np.isnan(skeleton).any(axis=1)]
    if len(valid) < 5:
        return None
    return valid.mean(axis=0)


def temporal_gate(
    state: PersonState,
    skeleton: np.ndarray,
    quality: dict[str, Any],
    fps: float,
    max_abs_speed: float,
    max_speed_multiplier: float,
    min_quality: float,
) -> tuple[bool, str]:
    """Reject unstable 3D detections before they enter feature/classifier state."""
    score = float(quality.get("score", 0.0))
    valid_ratio = float(quality.get("valid_joint_ratio", 0.0))
    if score < min_quality or valid_ratio < 0.35:
        return False, f"low skeleton quality score={score:.2f} valid={valid_ratio:.2f}"

    centroid = skeleton_centroid(skeleton)
    if centroid is None:
        return False, "not enough valid 3D joints"

    if state.centroids:
        prev = state.centroids[-1]
        speed = float(np.linalg.norm(centroid - prev) * fps)
        median_speed = float(np.median(state.speeds)) if state.speeds else 0.0
        adaptive_limit = max(max_abs_speed, median_speed * max_speed_multiplier)
        if speed > adaptive_limit:
            return False, f"unreasonable 3D centroid jump speed={speed:.2f} limit={adaptive_limit:.2f}"
    return True, "accepted"


def update_temporal_state(state: PersonState, skeleton: np.ndarray, fps: float) -> None:
    centroid = skeleton_centroid(skeleton)
    if centroid is None:
        return
    if state.centroids:
        state.speeds.append(float(np.linalg.norm(centroid - state.centroids[-1]) * fps))
    state.centroids.append(centroid)


def status_from_probability(p_fall: float, threshold_low: float, threshold_high: float) -> str:
    if p_fall >= threshold_high:
        return "fall"
    if p_fall >= threshold_low:
        return "suspected"
    return "normal"


def make_clip_path(scenario: int, frame_index: int, status: str, output_dir: Path) -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return output_dir / f"event_s{scenario:02d}_f{frame_index:06d}_{status}_{stamp}.mp4"


def load_classifier(args: argparse.Namespace):
    if args.checkpoint:
        try:
            classifier = FallClassifier(
                checkpoint_path=args.checkpoint,
                threshold_low=args.threshold_low,
                threshold_high=args.threshold_high,
            )
            print(f"[classifier] loaded ML checkpoint: {args.checkpoint}")
            return classifier
        except (FileNotFoundError, RuntimeError, TypeError, ValueError) as exc:
            print(f"[classifier] WARNING: {exc}")
    print("[classifier] using heuristic fallback")
    return HeuristicClassifier()


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
            "reason": "LLM review disabled; event remains suspected.",
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
        "video_path": result.get("video_path"),
        "sampled_frames": result.get("sampled_frames"),
    }


def send_or_print(payload: dict[str, Any], args: argparse.Namespace) -> None:
    text = json.dumps(payload, ensure_ascii=False)
    if args.no_send:
        print(f"[event:no-send] {text}")
        return
    ack = send_event(args.server_url, payload)
    print(f"[event:sent] ack={ack}")


def build_event_sequence_from_buffer(
    *,
    args: argparse.Namespace,
    clip_buffer: RollingClipBuffer,
    reviewer: LLMFallReviewer | None,
    scenario: int,
    frame_index: int,
    track_id: int | None,
    p_fall: float,
    status: str,
    quality: float | None,
    trigger_reason: str,
) -> list[dict[str, Any]]:
    """Create one or more payloads for a trigger using the shared P0 policy.

    Suspected events are intentionally split:
      1. immediate suspected alert without video, so staff sees it at once
      2. final update with clip/LLM result using the same event_id
    """
    clip_path: str | None = None
    llm_review: dict[str, Any] | None = None
    final_status = status
    output_dir = Path(args.clip_dir)
    event_id = f"event_{time.strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:6]}"
    payloads: list[dict[str, Any]] = []

    if status == "suspected":
        payloads.append(build_event_payload(
            client_id=args.client_id,
            device_id=args.device_id,
            address=args.address,
            status="suspected",
            p_fall=p_fall,
            threshold_low=args.threshold_low,
            threshold_high=args.threshold_high,
            clip_path=None,
            scenario=scenario,
            frame_index=frame_index,
            track_id=track_id,
            quality=quality,
            llm_review={
                "status": "pending_review",
                "is_fall": None,
                "confidence": None,
                "reason": "Suspected fall detected by 3D classifier; preparing 90-frame clip for LLM review.",
            },
            trigger_reason=trigger_reason,
            event_id=event_id,
        ))
        output_path = make_clip_path(scenario, frame_index, status, output_dir)
        clip_path = clip_buffer.export_latest(output_path, frames=args.event_clip_frames, fps=args.fps)
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
    elif status == "fall":
        output_path = make_clip_path(scenario, frame_index, status, output_dir)
        clip_path = clip_buffer.export_latest(
            output_path,
            frames=min(args.event_clip_frames, len(clip_buffer)),
            fps=args.fps,
        )
        llm_review = {
            "status": "skipped_direct_fall",
            "is_fall": True,
            "confidence": None,
            "reason": "3D classifier probability exceeded high threshold; LLM skipped.",
        }

    payloads.append(build_event_payload(
        client_id=args.client_id,
        device_id=args.device_id,
        address=args.address,
        status=final_status,
        p_fall=p_fall,
        threshold_low=args.threshold_low,
        threshold_high=args.threshold_high,
        clip_path=clip_path,
        scenario=scenario,
        frame_index=frame_index,
        track_id=track_id,
        quality=quality,
        llm_review=llm_review,
        trigger_reason=trigger_reason,
        event_id=event_id,
    ))
    return payloads


def run_client(args: argparse.Namespace) -> None:
    data_root = BASE_DIR / "dataset" / "mcfd"
    camera_ids = args.cameras or [1, 4, 7]
    output_dir = Path(args.clip_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    detector = PoseDetector(enable_fallback=args.pose_fallback)
    projections = get_projections(camera_ids)
    tri = Triangulator(projections)
    tracker = MultiPersonTracker(num_views=len(camera_ids))
    classifier = load_classifier(args)
    reviewer = load_reviewer(args)
    clip_buffer = RollingClipBuffer(max_frames=args.buffer_frames, fps=args.fps)
    states: dict[int, PersonState] = {}
    forced_event_sent = False

    frame_idx = 0
    start_time = time.time()
    print(
        f"[client] scenario={args.scenario} cameras={camera_ids} "
        f"thresholds={args.threshold_low:.2f}/{args.threshold_high:.2f} "
        f"buffer={args.buffer_frames} export={args.event_clip_frames}"
    )

    with MCFDReader(
        str(data_root),
        args.scenario,
        camera_ids,
        start_frame=args.start_frame,
        end_frame=None if args.max_frames is None else args.start_frame + args.max_frames,
        undistort=False,
    ) as reader:
        for frames in reader:
            frame_idx = args.start_frame + reader.current_frame - 1
            collage = make_multiview_collage(frames, labels=[f"Cam{cid}" for cid in camera_ids])
            clip_buffer.append(frame_idx, collage)

            raw_persons = detector.infer_batch_all(frames)
            tracked = tracker.update(raw_persons)
            view0_tracks = tracked[0] if tracked else []
            if args.max_people > 0:
                view0_tracks = view0_tracks[:args.max_people]

            for pidx, (track_id, kp_view0) in enumerate(view0_tracks):
                state = states.setdefault(track_id, PersonState())

                person_kps_raw = []
                for v in range(len(tracked)):
                    v_tracks = tracked[v]
                    if v == 0:
                        person_kps_raw.append(kp_view0)
                    elif pidx < len(v_tracks):
                        person_kps_raw.append(v_tracks[pidx][1])
                    else:
                        person_kps_raw.append(np.zeros((17, 3), dtype=np.float32))

                person_kps = [
                    undistort_keypoints(kp, camera_ids[v])
                    for v, kp in enumerate(person_kps_raw)
                ]

                raw_skel = tri.triangulate(person_kps)
                quality = tri.compute_quality(person_kps, raw_skel)
                skel = state.smoother.update(raw_skel)

                accepted, gate_reason = temporal_gate(
                    state,
                    skel,
                    quality,
                    fps=args.fps,
                    max_abs_speed=args.max_centroid_speed,
                    max_speed_multiplier=args.max_speed_multiplier,
                    min_quality=args.min_temporal_quality,
                )
                if not accepted:
                    state.rejected_frames += 1
                    if args.verbose:
                        print(f"[gate] frame={frame_idx} track={track_id} rejected: {gate_reason}")
                    continue

                update_temporal_state(state, skel, fps=args.fps)
                state.qualities.append(quality)

                prev_skel = state.skeletons[-1] if state.skeletons else None
                features = compute_features(skel, prev_skel, floor_estimator=state.floor_estimator)
                if features is not None:
                    if state.features:
                        features[3] = features[2] - state.features[-1][2]
                    else:
                        features[3] = 0.0
                    state.features.append(features)
                state.skeletons.append(skel.copy())

                if len(state.features) < 20 or frame_idx % args.classify_stride != 0:
                    continue

                window = np.array(list(state.features)[-20:])
                cls_result = classifier.predict(window)
                p_fall = float(cls_result["fall_prob"])
                status = status_from_probability(p_fall, args.threshold_low, args.threshold_high)
                mean_quality = float(np.mean([q["score"] for q in list(state.qualities)[-20:]]))
                if mean_quality < args.quality_threshold and status == "fall":
                    status = "suspected"
                    gate_reason = f"fall downgraded by recent quality={mean_quality:.2f}"

                forced = args.force_trigger_frame is not None and frame_idx >= args.force_trigger_frame
                if forced and state.last_event_frame < 0:
                    p_fall = args.force_p_fall
                    status = status_from_probability(p_fall, args.threshold_low, args.threshold_high)
                    gate_reason = "forced play trigger for smoke/demo"

                state.status = status
                if status == "normal":
                    continue
                if frame_idx - state.last_event_frame < args.event_cooldown_frames:
                    continue

                trigger_reason = (
                    f"classifier p_fall={p_fall:.3f}; status={status}; "
                    f"quality={mean_quality:.2f}; {gate_reason}"
                )
                payloads = build_event_sequence_from_buffer(
                    args=args,
                    clip_buffer=clip_buffer,
                    reviewer=reviewer,
                    scenario=args.scenario,
                    frame_index=frame_idx,
                    track_id=track_id,
                    p_fall=p_fall,
                    status=status,
                    quality=mean_quality,
                    trigger_reason=trigger_reason,
                )
                for payload in payloads:
                    send_or_print(payload, args)
                state.last_event_frame = frame_idx
                if forced:
                    forced_event_sent = True

                if args.stop_after_event:
                    elapsed = time.time() - start_time
                    print(f"[client] stopped after first event at frame={frame_idx}, elapsed={elapsed:.1f}s")
                    return

            if (
                args.force_trigger_frame is not None
                and not forced_event_sent
                and frame_idx >= args.force_trigger_frame
            ):
                p_fall = args.force_p_fall
                status = status_from_probability(p_fall, args.threshold_low, args.threshold_high)
                payloads = build_event_sequence_from_buffer(
                    args=args,
                    clip_buffer=clip_buffer,
                    reviewer=reviewer,
                    scenario=args.scenario,
                    frame_index=frame_idx,
                    track_id=None,
                    p_fall=p_fall,
                    status=status,
                    quality=None,
                    trigger_reason="forced trigger before classifier produced a window",
                )
                for payload in payloads:
                    send_or_print(payload, args)
                forced_event_sent = True
                if args.stop_after_event:
                    elapsed = time.time() - start_time
                    print(f"[client] stopped after forced event at frame={frame_idx}, elapsed={elapsed:.1f}s")
                    return

            if args.log_every and frame_idx % args.log_every == 0:
                print(f"[client] frame={frame_idx} buffer={len(clip_buffer)} tracks={len(states)}")

    print(f"[client] complete frames={frame_idx - args.start_frame + 1}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Continuous fall detection event client")
    parser.add_argument("--scenario", type=int, default=1)
    parser.add_argument("--cameras", type=int, nargs="+", default=[1, 4, 7])
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--checkpoint", default=str(DEFAULT_3D_CHECKPOINT))
    parser.add_argument("--threshold-low", type=float, default=0.30)
    parser.add_argument("--threshold-high", type=float, default=0.40)
    parser.add_argument("--quality-threshold", type=float, default=0.45)
    parser.add_argument("--min-temporal-quality", type=float, default=0.15)
    parser.add_argument("--max-centroid-speed", type=float, default=25.0)
    parser.add_argument("--max-speed-multiplier", type=float, default=8.0)
    parser.add_argument("--classify-stride", type=int, default=5)
    parser.add_argument("--max-people", type=int, default=1)
    parser.add_argument("--buffer-frames", type=int, default=300)
    parser.add_argument("--event-clip-frames", type=int, default=90)
    parser.add_argument("--event-cooldown-frames", type=int, default=300)
    parser.add_argument("--clip-dir", default="clips")
    parser.add_argument("--server-url", default="ws://127.0.0.1:8765/ws/events")
    parser.add_argument("--client-id", default="continuous-client-01")
    parser.add_argument("--device-id", default="mcfd-camset-1-4-7")
    parser.add_argument("--address", default="Demo Community / MCFD virtual cameras")
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
    parser.add_argument("--pose-fallback", action="store_true")
    parser.add_argument("--no-send", action="store_true")
    parser.add_argument("--stop-after-event", action="store_true")
    parser.add_argument("--force-trigger-frame", type=int, default=None)
    parser.add_argument("--force-p-fall", type=float, default=0.35)
    parser.add_argument("--log-every", type=int, default=60)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    run_client(args)


if __name__ == "__main__":
    main()
