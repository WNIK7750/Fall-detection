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

Roadshow live loop:
    python continuous_event_client.py --roadshow-loop --start-near-fall --server-url ws://127.0.0.1:8765/ws/events
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

import cv2
import numpy as np

from pipeline.calibration import get_projections, undistort_keypoints
from pipeline.classifier import FallClassifier, HeuristicClassifier
from pipeline.clip_buffer import BufferedFrame, RollingClipBuffer
from pipeline.event_schema import build_event_payload
from pipeline.event_sender import send_event
from pipeline.features_3d import FloorEstimator, compute_features
from pipeline.llm_reviewer import LLMFallReviewer, make_contact_sheet, make_multiview_collage
from pipeline.pose_inference import MultiPersonTracker, PoseDetector
from pipeline.triangulation import KalmanSmoother, Triangulator
from pipeline.video_reader import MCFDReader, find_fall_segments, load_annotations


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


@dataclass
class PendingEvent:
    event_id: str
    scenario: int
    trigger_frame: int
    release_frame: int
    track_id: int | None
    p_fall: float
    quality: float | None
    trigger_reason: str
    status: str = "suspected"


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
    max_speed_rejects: int,
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
            if state.rejected_frames >= max_speed_rejects:
                return True, (
                    f"accepted after speed-gate recovery speed={speed:.2f} "
                    f"limit={adaptive_limit:.2f}"
                )
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
    _ = threshold_high
    if p_fall >= threshold_low:
        return "suspected"
    return "normal"


def make_clip_path(scenario: int, frame_index: int, status: str, output_dir: Path) -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return output_dir / f"event_s{scenario:02d}_f{frame_index:06d}_{status}_{stamp}.mp4"


def scenario_start_frame(args: argparse.Namespace, data_root: Path, scenario: int, rng: random.Random) -> int:
    if not args.start_near_fall:
        return args.start_frame
    annotations = load_annotations(str(data_root))
    falls = find_fall_segments(annotations, scenario, cam_id=args.cameras[0])
    if not falls:
        return args.start_frame
    start, _ = rng.choice(falls)
    return max(0, start - args.pre_roll_frames)


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
        contact_sheet=False,
    )
    if result.get("is_fall") is True and float(result.get("confidence", 0.0)) >= args.llm_min_confidence:
        status = "fall"
    elif result.get("is_fall") is False and float(result.get("confidence", 0.0)) >= args.llm_min_confidence:
        status = "manual_review"
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


def review_event_evidence(
    reviewer: LLMFallReviewer | None,
    clip_path: str,
    contact_sheets: list[np.ndarray],
    args: argparse.Namespace,
) -> dict[str, Any]:
    if reviewer is None:
        return {
            "status": "suspected",
            "is_fall": None,
            "confidence": 0.0,
            "reason": "LLM review disabled; event remains suspected.",
        }
    if contact_sheets:
        result = reviewer.review_multiview_contact_sheets(contact_sheets, video_path=clip_path)
    else:
        result = reviewer.review_video_file(
            video_path=clip_path,
            frame_stride=args.llm_frame_stride,
            max_frames=args.llm_max_frames,
            contact_sheet=False,
        )
    if result.get("is_fall") is True and float(result.get("confidence", 0.0)) >= args.llm_min_confidence:
        status = "fall"
    elif result.get("is_fall") is False and float(result.get("confidence", 0.0)) >= args.llm_min_confidence:
        status = "manual_review"
    else:
        status = "suspected"
    return {
        "status": status,
        "is_fall": result.get("is_fall"),
        "confidence": result.get("confidence"),
        "reason": result.get("reason", ""),
        "video_path": result.get("video_path"),
        "sampled_frames": result.get("sampled_frames"),
        "views": result.get("views"),
        "contact_sheet": result.get("contact_sheet"),
    }


def send_or_print(payload: dict[str, Any], args: argparse.Namespace) -> None:
    text = json.dumps(payload, ensure_ascii=False)
    if args.no_send:
        print(f"[event:no-send] {text}")
        return
    ack = send_event(args.server_url, payload)
    print(f"[event:sent] ack={ack}")


class LLMReviewManager:
    """Single-worker LLM queue with one long-lived loaded reviewer.

    The main detector loop submits suspected clips and keeps running. The
    worker serializes Qwen3-VL requests so the model stays loaded once and
    GPU/CPU memory is not thrashed by concurrent LLM calls.
    """

    def __init__(self, enabled: bool, args: argparse.Namespace):
        self._executor = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="llm-review")
            if enabled
            else None
        )
        self._enabled = enabled
        self._args = args
        self._reviewer: LLMFallReviewer | None = None
        self._pending: list[Future] = []

    def submit(self, task: Callable[[], None]) -> Future | None:
        if self._executor is None:
            task()
            return None
        self._pending = [future for future in self._pending if not future.done()]
        future = self._executor.submit(task)
        future.add_done_callback(self._log_failure)
        self._pending.append(future)
        return future

    def wait_all(self) -> None:
        for future in list(self._pending):
            future.result()
        self._pending.clear()

    def review_clip(self, clip_path: str) -> dict[str, Any]:
        if not self._enabled:
            return review_clip(None, clip_path, self._args)
        if self._reviewer is None:
            self._reviewer = load_reviewer(self._args)
        return review_clip(self._reviewer, clip_path, self._args)

    def review_event_evidence(self, clip_path: str, contact_sheets: list[np.ndarray]) -> dict[str, Any]:
        if not self._enabled:
            return review_event_evidence(None, clip_path, contact_sheets, self._args)
        if self._reviewer is None:
            self._reviewer = load_reviewer(self._args)
        return review_event_evidence(self._reviewer, clip_path, contact_sheets, self._args)

    def shutdown(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=False)

    @staticmethod
    def _log_failure(future: Future) -> None:
        try:
            exc = future.exception()
        except Exception as err:
            print(f"[llm-manager] failed to inspect review task: {err}")
            return
        if exc is not None:
            print(f"[llm-manager] review task failed: {exc}")


def clip_meta_payload(clip_info: Any) -> dict[str, Any]:
    return {
        "frame_count": clip_info.frame_count,
        "fps": clip_info.fps,
        "start_frame": clip_info.start_frame,
        "end_frame": clip_info.end_frame,
        "start_timestamp_sec": round(clip_info.start_timestamp_sec, 3),
        "end_timestamp_sec": round(clip_info.end_timestamp_sec, 3),
        "duration_sec": round(
            (clip_info.frame_count - 1) / max(clip_info.fps, 1e-6),
            3,
        ),
    }


def make_initial_suspected_payload(
    *,
    args: argparse.Namespace,
    event: PendingEvent,
) -> dict[str, Any]:
    payload = build_event_payload(
        client_id=args.client_id,
        device_id=args.device_id,
        address=args.address,
        status="suspected",
        p_fall=event.p_fall,
        threshold_low=args.threshold_low,
        threshold_high=args.threshold_high,
        clip_path=None,
        scenario=event.scenario,
        frame_index=event.trigger_frame,
        track_id=event.track_id,
        quality=event.quality,
        llm_review={
            "status": "waiting_for_post_context",
            "is_fall": None,
            "confidence": None,
            "reason": (
                "Suspected fall detected by 3D classifier; collecting post-event "
                f"context until frame {event.release_frame} before LLM review."
            ),
        },
        trigger_reason=event.trigger_reason,
        event_id=event.event_id,
    )
    payload["video_released"] = False
    payload["event_timestamp_sec"] = round(event.trigger_frame / max(args.fps, 1e-6), 3)
    payload["evidence_release_frame"] = event.release_frame
    return payload


def make_view_contact_sheets(
    buffered_frames: list[BufferedFrame],
    camera_count: int,
    sample_stride: int,
    sample_count: int,
) -> list[np.ndarray]:
    if not buffered_frames or buffered_frames[0].views is None:
        return []
    indices = [i * sample_stride for i in range(sample_count)]
    indices = [idx for idx in indices if idx < len(buffered_frames)]
    if not indices:
        return []

    sheets: list[np.ndarray] = []
    for view_idx in range(camera_count):
        view_frames = []
        for idx in indices:
            views = buffered_frames[idx].views
            if views is None or view_idx >= len(views):
                return []
            view_frames.append(views[view_idx])
        sheets.append(make_contact_sheet(view_frames, columns=3, tile_width=320, label_prefix=f"c{view_idx + 1}_t"))
    return sheets


def finalize_pending_event(
    *,
    args: argparse.Namespace,
    clip_buffer: RollingClipBuffer,
    review_manager: LLMReviewManager,
    event: PendingEvent,
) -> Future | None:
    output_path = make_clip_path(event.scenario, event.trigger_frame, event.status, Path(args.clip_dir))
    start_frame = event.trigger_frame - args.event_pre_frames + 1
    end_frame = event.trigger_frame + args.event_post_frames
    clip_info = clip_buffer.export_range_info(
        output_path,
        start_frame=start_frame,
        end_frame=end_frame,
        fps=args.clip_fps,
    )
    if clip_info is None:
        if args.verbose:
            print(
                f"[event] frame={event.trigger_frame} could not export evidence "
                f"range={start_frame}-{end_frame}; buffer has {len(clip_buffer)} frames"
            )
        return None

    evidence_frames = clip_buffer.range_by_frame(start_frame, end_frame)
    contact_sheets = make_view_contact_sheets(
        evidence_frames,
        camera_count=len(args.cameras),
        sample_stride=args.llm_sheet_stride,
        sample_count=args.llm_sheet_frames,
    )
    clip_path = clip_info.path

    def review_and_send() -> None:
        llm_review = review_manager.review_event_evidence(clip_path, contact_sheets)
        reviewed_status = llm_review.get("status", "suspected")
        final_status = "fall" if reviewed_status == "fall" else "suspected"
        final_payload = build_event_payload(
            client_id=args.client_id,
            device_id=args.device_id,
            address=args.address,
            status=final_status,
            p_fall=event.p_fall,
            threshold_low=args.threshold_low,
            threshold_high=args.threshold_high,
            clip_path=clip_path,
            scenario=event.scenario,
            frame_index=event.trigger_frame,
            track_id=event.track_id,
            quality=event.quality,
            llm_review=llm_review,
            trigger_reason=event.trigger_reason,
            event_id=event.event_id,
        )
        final_payload["video_released"] = final_status == "fall" and clip_path is not None
        final_payload["event_timestamp_sec"] = round(event.trigger_frame / max(args.fps, 1e-6), 3)
        final_payload["clip_meta"] = clip_meta_payload(clip_info)
        final_payload["llm_sample"] = {
            "views": len(contact_sheets),
            "frames_per_view": args.llm_sheet_frames,
            "stride_frames": args.llm_sheet_stride,
        }
        send_or_print(final_payload, args)

    return review_manager.submit(review_and_send)


def emit_suspected_event(
    *,
    args: argparse.Namespace,
    clip_buffer: RollingClipBuffer,
    review_manager: LLMReviewManager,
    scenario: int,
    frame_index: int,
    track_id: int | None,
    p_fall: float,
    quality: float | None,
    trigger_reason: str,
) -> Future | None:
    """Send suspected immediately, then run LLM review without blocking detection."""
    output_dir = Path(args.clip_dir)
    event_id = f"event_{time.strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:6]}"
    initial_payload = build_event_payload(
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
    )
    initial_payload["video_released"] = False
    initial_payload["event_timestamp_sec"] = round(frame_index / max(args.fps, 1e-6), 3)
    send_or_print(initial_payload, args)

    output_path = make_clip_path(scenario, frame_index, "suspected", output_dir)
    clip_info = clip_buffer.export_latest_info(output_path, frames=args.event_clip_frames, fps=args.clip_fps)
    clip_path = None if clip_info is None else clip_info.path

    def review_and_send() -> None:
        if clip_info is None:
            final_status = "fall"
            llm_review = {
                "status": "fall",
                "is_fall": True,
                "confidence": None,
                "reason": "Less than 90 buffered frames available; escalated by policy.",
            }
        else:
            llm_review = review_manager.review_clip(clip_path)
            reviewed_status = llm_review.get("status", "suspected")
            final_status = "fall" if reviewed_status == "fall" else "suspected"

        final_payload = build_event_payload(
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
        )
        final_payload["video_released"] = final_status == "fall" and clip_path is not None
        final_payload["event_timestamp_sec"] = round(frame_index / max(args.fps, 1e-6), 3)
        if clip_info is not None:
            final_payload["clip_meta"] = clip_meta_payload(clip_info)
        send_or_print(final_payload, args)

    return review_manager.submit(review_and_send)


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
    pre_send: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    """Create one or more payloads for a trigger using the shared P0 policy.

    Suspected events are intentionally split:
      1. immediate suspected alert without video, so staff sees it at once
      2. final update with hidden/released clip path using the same event_id

    The detector loop owns clip creation and release policy. The LLM only
    reviews an already-exported clip and returns a judgment.
    """
    clip_path: str | None = None
    clip_info = None
    llm_review: dict[str, Any] | None = None
    final_status = status
    output_dir = Path(args.clip_dir)
    event_id = f"event_{time.strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:6]}"
    payloads: list[dict[str, Any]] = []

    if status == "suspected":
        initial_payload = build_event_payload(
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
        )
        initial_payload["video_released"] = False
        initial_payload["event_timestamp_sec"] = round(frame_index / max(args.fps, 1e-6), 3)
        if pre_send is None:
            payloads.append(initial_payload)
        else:
            pre_send(initial_payload)
        output_path = make_clip_path(scenario, frame_index, status, output_dir)
        clip_info = clip_buffer.export_latest_info(output_path, frames=args.event_clip_frames, fps=args.clip_fps)
        clip_path = None if clip_info is None else clip_info.path
        if clip_info is None:
            final_status = "fall"
            llm_review = {
                "status": "fall",
                "is_fall": True,
                "confidence": None,
                "reason": "Less than 90 buffered frames available; escalated by policy.",
            }
        else:
            llm_review = review_clip(reviewer, clip_path, args)
            reviewed_status = llm_review.get("status", "suspected")
            final_status = "fall" if reviewed_status == "fall" else "suspected"
    elif status == "fall":
        output_path = make_clip_path(scenario, frame_index, status, output_dir)
        clip_info = clip_buffer.export_latest_info(
            output_path,
            frames=min(args.event_clip_frames, len(clip_buffer)),
            fps=args.clip_fps,
        )
        clip_path = None if clip_info is None else clip_info.path
        llm_review = {
            "status": "skipped_direct_fall",
            "is_fall": True,
            "confidence": None,
            "reason": "3D classifier probability exceeded high threshold; LLM skipped.",
        }

    final_payload = build_event_payload(
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
    )
    final_payload["video_released"] = final_status == "fall" and clip_path is not None
    final_payload["event_timestamp_sec"] = round(frame_index / max(args.fps, 1e-6), 3)
    if clip_info is not None:
        final_payload["clip_meta"] = clip_meta_payload(clip_info)
    payloads.append(final_payload)
    return payloads


def run_client(args: argparse.Namespace) -> None:
    data_root = BASE_DIR / "dataset" / "mcfd"
    camera_ids = args.cameras or [1, 4, 7]
    output_dir = Path(args.clip_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    detector = PoseDetector(enable_fallback=args.pose_fallback)
    projections = get_projections(camera_ids)
    tri = Triangulator(projections)
    classifier = load_classifier(args)
    review_manager = LLMReviewManager(enabled=args.llm, args=args)
    pending_reviews: list[Future] = []
    rng = random.Random(args.seed)
    scenario_pool = args.scenarios or [args.scenario]
    scenario_index = 0
    start_time = time.time()
    print(
        f"[client] scenarios={scenario_pool} cameras={camera_ids} "
        f"review_threshold={args.threshold_low:.2f} "
        f"buffer={args.buffer_frames} export={args.event_clip_frames}@{args.clip_fps:g}fps "
        f"pre={args.event_pre_frames} post={args.event_post_frames}"
    )

    try:
        while True:
            scenario = rng.choice(scenario_pool) if args.random_scenarios else scenario_pool[scenario_index % len(scenario_pool)]
            scenario_index += 1
            scenario_start = scenario_start_frame(args, data_root, scenario, rng)
            scenario_end = None if args.max_frames is None else scenario_start + args.max_frames
            print(f"[client] starting scenario={scenario} start_frame={scenario_start}")
    
            tracker = MultiPersonTracker(num_views=len(camera_ids))
            clip_buffer = RollingClipBuffer(max_frames=args.buffer_frames, fps=args.fps)
            states: dict[int, PersonState] = {}
            pending_events: list[PendingEvent] = []
            forced_event_sent = False
            last_frame_idx = scenario_start
    
            with MCFDReader(
                str(data_root),
                scenario,
                camera_ids,
                start_frame=scenario_start,
                end_frame=scenario_end,
                undistort=False,
            ) as reader:
                for frames in reader:
                    frame_idx = reader.ref_start_frame + reader.current_frame - 1
                    last_frame_idx = frame_idx
                    if args.realtime:
                        time.sleep(max(0.0, 1.0 / args.fps))
    
                    collage = make_multiview_collage(frames, labels=[f"Cam{cid}" for cid in camera_ids])
                    clip_buffer.append(frame_idx, collage, views=frames)
    
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
                            max_speed_rejects=args.max_speed_rejects,
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
                            features[3] = features[2] - state.features[-1][2] if state.features else 0.0
                            state.features.append(features)
                        state.skeletons.append(skel.copy())
    
                        required_features = (
                            (args.classifier_window_steps - 1) * args.classifier_window_subsample + 1
                        )
                        if len(state.features) < required_features or frame_idx % args.classify_stride != 0:
                            continue
                        state.rejected_frames = 0

                        recent_features = list(state.features)[-required_features:]
                        window = np.array(
                            recent_features[::args.classifier_window_subsample],
                            dtype=np.float32,
                            copy=True,
                        )
                        for i in range(len(window)):
                            window[i, 3] = window[i, 2] - window[i - 1, 2] if i > 0 else 0.0
                        cls_result = classifier.predict(window)
                        p_fall = float(cls_result["fall_prob"])
                        status = status_from_probability(p_fall, args.threshold_low, args.threshold_high)
                        mean_quality = float(np.mean([q["score"] for q in list(state.qualities)[-20:]]))
                        if args.log_prob_every and frame_idx % args.log_prob_every == 0:
                            print(
                                f"[prob] frame={frame_idx} track={track_id} "
                                f"p_fall={p_fall:.3f} status={status} quality={mean_quality:.2f}"
                            )
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
                        if len(clip_buffer) < args.event_pre_frames:
                            if args.verbose:
                                print(
                                    f"[event] frame={frame_idx} status={status} waiting for "
                                    f"{args.event_pre_frames} pre-event frames ({len(clip_buffer)} ready)"
                                )
                            continue
                        if any(event.track_id == track_id for event in pending_events):
                            continue

                        trigger_reason = (
                            f"classifier p_fall={p_fall:.3f}; status={status}; "
                            f"quality={mean_quality:.2f}; {gate_reason}"
                        )
                        event = PendingEvent(
                            event_id=f"event_{time.strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:6]}",
                            scenario=scenario,
                            trigger_frame=frame_idx,
                            release_frame=frame_idx + args.event_post_frames,
                            track_id=track_id,
                            p_fall=p_fall,
                            quality=mean_quality,
                            trigger_reason=trigger_reason,
                            status=status,
                        )
                        pending_events.append(event)
                        send_or_print(make_initial_suspected_payload(args=args, event=event), args)
                        state.last_event_frame = frame_idx
                        if forced:
                            forced_event_sent = True

                    if (
                        args.force_trigger_frame is not None
                        and not forced_event_sent
                        and frame_idx >= args.force_trigger_frame
                    ):
                        p_fall = args.force_p_fall
                        status = status_from_probability(p_fall, args.threshold_low, args.threshold_high)
                        if len(clip_buffer) < args.event_pre_frames:
                            if args.verbose:
                                print(
                                    f"[event] frame={frame_idx} forced trigger waiting for "
                                    f"{args.event_pre_frames} pre-event frames ({len(clip_buffer)} ready)"
                                )
                            continue
                        event = PendingEvent(
                            event_id=f"event_{time.strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:6]}",
                            scenario=scenario,
                            trigger_frame=frame_idx,
                            release_frame=frame_idx + args.event_post_frames,
                            track_id=None,
                            p_fall=p_fall,
                            quality=None,
                            trigger_reason="forced trigger before classifier produced a window",
                            status=status,
                        )
                        pending_events.append(event)
                        send_or_print(make_initial_suspected_payload(args=args, event=event), args)
                        forced_event_sent = True

                    ready_events = [event for event in pending_events if frame_idx >= event.release_frame]
                    for event in ready_events:
                        future = finalize_pending_event(
                            args=args,
                            clip_buffer=clip_buffer,
                            review_manager=review_manager,
                            event=event,
                        )
                        if future is not None:
                            pending_reviews.append(future)
                        pending_events.remove(event)
                        if args.stop_after_event:
                            for pending_future in pending_reviews:
                                pending_future.result()
                            elapsed = time.time() - start_time
                            print(
                                f"[client] stopped after finalized event at "
                                f"frame={frame_idx}, elapsed={elapsed:.1f}s"
                            )
                            return
    
                    if args.log_every and frame_idx % args.log_every == 0:
                        print(f"[client] frame={frame_idx} buffer={len(clip_buffer)} tracks={len(states)}")
    
            print(f"[client] scenario={scenario} complete last_frame={last_frame_idx}; resetting 3D state")
            if not args.roadshow_loop:
                break

        for future in pending_reviews:
            future.result()
        review_manager.wait_all()
        print("[client] complete")
    finally:
        review_manager.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser(description="Continuous fall detection event client")
    parser.add_argument("--scenario", type=int, default=1)
    parser.add_argument("--scenarios", type=int, nargs="+", default=None)
    parser.add_argument("--roadshow-loop", action="store_true",
                        help="Continuously play dataset scenarios and reset 3D state between videos")
    parser.add_argument("--random-scenarios", action="store_true", default=True)
    parser.add_argument("--sequential-scenarios", action="store_false", dest="random_scenarios")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--cameras", type=int, nargs="+", default=[1, 4, 7])
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--start-near-fall", action="store_true",
                        help="Use data_tuple3.csv to start shortly before a labeled fall segment")
    parser.add_argument("--pre-roll-frames", type=int, default=300)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--clip-fps", type=float, default=30.0,
                        help="Playback FPS for exported evidence clips; detection FPS remains --fps")
    parser.add_argument("--realtime", action="store_true")
    parser.add_argument("--checkpoint", default=str(DEFAULT_3D_CHECKPOINT))
    parser.add_argument("--threshold-low", type=float, default=0.40)
    parser.add_argument("--threshold-high", type=float, default=1.0,
                        help="Metadata only in current demo policy; all p_fall >= threshold-low goes to LLM review")
    parser.add_argument("--quality-threshold", type=float, default=0.45)
    parser.add_argument("--min-temporal-quality", type=float, default=0.15)
    parser.add_argument("--max-centroid-speed", type=float, default=150.0)
    parser.add_argument("--max-speed-multiplier", type=float, default=8.0)
    parser.add_argument("--max-speed-rejects", type=int, default=10,
                        help="Accept and resync after this many consecutive speed-gate rejects")
    parser.add_argument("--classify-stride", type=int, default=5)
    parser.add_argument("--classifier-window-steps", type=int, default=20)
    parser.add_argument("--classifier-window-subsample", type=int, default=2,
                        help="Use every Nth feature in the realtime classifier window; training fall windows used 2")
    parser.add_argument("--max-people", type=int, default=1)
    parser.add_argument("--buffer-frames", type=int, default=600)
    parser.add_argument("--event-pre-frames", type=int, default=90,
                        help="Frames retained before and including the trigger frame")
    parser.add_argument("--event-post-frames", type=int, default=151,
                        help="Frames collected after the trigger before exporting evidence")
    parser.add_argument("--event-clip-frames", type=int, default=241,
                        help="Total evidence clip frames; defaults to 90 pre + 151 post")
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
    parser.add_argument("--llm-frame-stride", type=int, default=10)
    parser.add_argument("--llm-max-frames", type=int, default=3)
    parser.add_argument("--llm-sheet-frames", type=int, default=9)
    parser.add_argument("--llm-sheet-stride", type=int, default=29)
    parser.add_argument("--llm-min-confidence", type=float, default=0.60)
    parser.add_argument("--pose-fallback", action="store_true")
    parser.add_argument("--no-send", action="store_true")
    parser.add_argument("--stop-after-event", action="store_true")
    parser.add_argument("--force-trigger-frame", type=int, default=None)
    parser.add_argument("--force-p-fall", type=float, default=0.45)
    parser.add_argument("--log-every", type=int, default=60)
    parser.add_argument("--log-prob-every", type=int, default=0,
                        help="Print classifier probability on matching frame indices for debugging")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    expected_clip_frames = args.event_pre_frames + args.event_post_frames
    if args.event_clip_frames != expected_clip_frames:
        print(
            f"[client] overriding event_clip_frames={args.event_clip_frames} "
            f"with pre+post={expected_clip_frames}"
        )
        args.event_clip_frames = expected_clip_frames
    run_client(args)


if __name__ == "__main__":
    main()
