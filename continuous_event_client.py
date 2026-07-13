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
import threading
import time
from collections import deque
from contextlib import nullcontext
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
from pipeline.llm_reviewer import (
    CAMERA_CONTACT_SHEET_PROMPT,
    LLMFallReviewer,
    SYSTEM_PROMPT,
    make_contact_sheet,
    make_multiview_collage,
)
from pipeline.pose_inference import MultiPersonTracker, PoseDetector
from pipeline.triangulation import KalmanSmoother, Triangulator
from pipeline.video_reader import MCFDReader, find_fall_segments, load_annotations


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_3D_CHECKPOINT = BASE_DIR / "checkpoints" / "archive" / "enhanced_3d_20260713_142749" / "best_model.pth"
DEFAULT_MODEL = BASE_DIR / "models" / "llm" / "Qwen3-VL-2B-Instruct-Q4_K_M.gguf"
DEFAULT_MMPROJ = BASE_DIR / "models" / "llm" / "mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf"


@dataclass
class PersonState:
    features: deque[np.ndarray] = field(default_factory=lambda: deque(maxlen=80))
    qualities: deque[dict[str, Any]] = field(default_factory=lambda: deque(maxlen=80))
    skeletons: deque[np.ndarray] = field(default_factory=lambda: deque(maxlen=80))
    centroids: deque[np.ndarray] = field(default_factory=lambda: deque(maxlen=30))
    speeds: deque[float] = field(default_factory=lambda: deque(maxlen=30))
    floor_estimator: FloorEstimator = field(default_factory=FloorEstimator)
    smoother: KalmanSmoother = field(default_factory=KalmanSmoother)
    status: str = "normal"
    last_event_frame: int = -10_000
    event_lock_until_frame: int = -10_000
    rejected_frames: int = 0
    last_feature_frame: int | None = None


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
    last_update_frame: int = -10_000


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
    low_quality = score < min_quality or valid_ratio < 0.35
    if low_quality and not state.skeletons:
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
    if low_quality:
        return True, f"accepted via smoother score={score:.2f} valid={valid_ratio:.2f}"
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
                device=args.classifier_device,
            )
            print(f"[classifier] loaded ML checkpoint: {args.checkpoint} device={classifier.device}")
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
        n_batch=args.n_batch,
        n_ubatch=args.n_ubatch,
        max_image_side=args.llm_image_max_side,
    )
    if not reviewer.ready:
        raise RuntimeError("LLM reviewer failed to load")
    return reviewer


def is_cuda_device(device: str | None) -> bool:
    if device is None:
        return False
    normalized = str(device).strip().lower()
    return normalized == "cuda" or normalized.startswith("cuda:") or normalized.isdigit()


def maybe_gpu_lock(lock: threading.RLock | None, enabled: bool):
    return lock if lock is not None and enabled else nullcontext()


def llm_error_result(reason: str) -> dict[str, Any]:
    return {
        "status": "suspected",
        "is_fall": None,
        "confidence": 0.0,
        "reason": reason,
    }


def print_llm_token_budget(reviewer: LLMFallReviewer, view_labels: list[str]) -> None:
    _ = view_labels
    print("[llm-tokens] text-only tokenizer counts; runtime sends up to 8 contact-sheet images per camera review")
    prompt = CAMERA_CONTACT_SHEET_PROMPT.format(view_label="cam")
    direct_count = reviewer.count_text_tokens(f"{SYSTEM_PROMPT}\n{prompt}", add_bos=False)
    print(f"[llm-tokens] single_camera_text={direct_count}")
    print(f"[llm-tokens] est_total_with_8_contact_sheets={direct_count + 529 * 8 if direct_count is not None else None}")


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
    view_sequences: list[list[np.ndarray]],
    view_labels: list[str],
    p_fall: float,
    args: argparse.Namespace,
) -> dict[str, Any]:
    if reviewer is None:
        return {
            "status": "suspected",
            "is_fall": None,
            "confidence": 0.0,
            "reason": "LLM review disabled; event remains suspected.",
        }
    view_results: list[dict[str, Any]] = []
    for label, frames in zip(view_labels, view_sequences):
        if not frames:
            continue
        frames = frames[:32]
        sheets = []
        for group_idx in range(8):
            group = frames[group_idx * 4:(group_idx + 1) * 4]
            if not group:
                continue
            sheets.append(
                make_contact_sheet(
                    group,
                    columns=2,
                    tile_width=args.llm_sheet_tile_width,
                    label_prefix="t",
                    start_index=group_idx * 4,
                    center_crop_square=True,
                )
            )
        if sheets:
            view_results.append(
                reviewer.review_camera_contact_sheets(
                    sheets,
                    view_label=label,
                    video_path=clip_path,
                    sampled_frames=len(frames),
                )
            )

    if view_results:
        nonfall_actions = {"squat_bend", "sit_down", "lie_down", "stand_up", "unclear"}
        llm_fall_views = [
            result for result in view_results
            if result.get("is_fall") is True
            and str(result.get("action", "fall")).strip() not in nonfall_actions
            and float(result.get("confidence", 0.0) or 0.0) >= args.llm_min_confidence
        ]
        valid_nonfall_views = [
            result for result in view_results
            if result.get("is_fall") is False and result.get("reason")
        ]
        all_views_nonfall = len(valid_nonfall_views) == len(view_results) == len(view_labels)
        high_3d_fallback = p_fall >= args.llm_3d_fall_threshold

        if len(llm_fall_views) >= args.llm_required_fall_views:
            status = "fall"
            is_fall = True
            confidence = max(float(result.get("confidence", 0.0) or 0.0) for result in llm_fall_views)
            reason = f"至少{args.llm_required_fall_views}路摄像头判断为摔倒：" + "; ".join(
                f"{result.get('view')}: {result.get('reason', '')}" for result in llm_fall_views
            )
        elif all_views_nonfall and high_3d_fallback:
            status = "fall"
            is_fall = True
            confidence = max(float(result.get("confidence", 0.0) or 0.0) for result in view_results)
            reason = f"三路LLM均未确认摔倒，但3D分类器p_fall={p_fall:.3f}达到兜底阈值{args.llm_3d_fall_threshold:.2f}。"
        else:
            status = "suspected"
            is_fall = False if all_views_nonfall else None
            confidence = max(float(result.get("confidence", 0.0) or 0.0) for result in view_results)
            reason = (
                f"LLM支持摔倒的摄像头数量为{len(llm_fall_views)}/{len(view_results)}，"
                f"未达到{args.llm_required_fall_views}路要求，且3D分类器未达到高置信兜底；保留为疑似事件。"
            )

        return {
            "status": status,
            "is_fall": is_fall,
            "confidence": round(confidence, 3),
            "reason": reason[:300],
            "video_path": clip_path,
            "sampled_frames": max((result.get("sampled_frames", 0) or 0) for result in view_results),
            "views": len(view_results),
            "review_mode": "per_camera_81_frames_or_with_3d_fallback",
            "contact_sheet": True,
            "view_results": view_results,
            "p_fall": round(float(p_fall), 4),
            "llm_3d_fall_threshold": args.llm_3d_fall_threshold,
            "llm_required_fall_views": args.llm_required_fall_views,
        }

    result = reviewer.review_video_file(
        video_path=clip_path,
        frame_stride=args.llm_frame_stride,
        max_frames=args.llm_max_frames,
        contact_sheet=False,
    )
    if p_fall >= args.llm_3d_fall_threshold:
        result["is_fall"] = True
        result["confidence"] = max(float(result.get("confidence", 0.0) or 0.0), args.llm_3d_fall_threshold)
        result["reason"] = f"Fallback review unavailable; 3D classifier p_fall={p_fall:.3f} reached high-confidence threshold."
    else:
        result["reason"] = result.get("reason", "Fallback clip review used; event remains suspected.")
    if result.get("is_fall") is True and float(result.get("confidence", 0.0)) >= args.llm_min_confidence:
        status = "fall"
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
        "fusion_mode": result.get("fusion_mode"),
        "review_mode": result.get("review_mode", args.llm_review_mode),
        "contact_sheet": result.get("contact_sheet"),
        "view_descriptions": result.get("view_descriptions"),
        "combined_description": result.get("combined_description"),
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

    def __init__(self, enabled: bool, args: argparse.Namespace, gpu_lock: threading.RLock | None = None):
        self._executor = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="llm-review")
            if enabled
            else None
        )
        self._enabled = enabled
        self._args = args
        self._reviewer: LLMFallReviewer | None = None
        self._pending: list[Future] = []
        self._gpu_lock = gpu_lock
        self._printed_token_budget = False

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
        try:
            with maybe_gpu_lock(self._gpu_lock, self._args.llm_gpu_lock):
                if self._reviewer is None:
                    self._reviewer = load_reviewer(self._args)
                return review_clip(self._reviewer, clip_path, self._args)
        except Exception as exc:
            return llm_error_result(f"LLM review failed; event remains suspected: {exc}")

    def review_event_evidence(
        self,
        clip_path: str,
        view_sequences: list[list[np.ndarray]],
        view_labels: list[str],
        p_fall: float,
    ) -> dict[str, Any]:
        if not self._enabled:
            return review_event_evidence(None, clip_path, view_sequences, view_labels, p_fall, self._args)
        try:
            with maybe_gpu_lock(self._gpu_lock, self._args.llm_gpu_lock):
                if self._reviewer is None:
                    self._reviewer = load_reviewer(self._args)
                if self._args.llm_debug_tokens and not self._printed_token_budget:
                    print_llm_token_budget(self._reviewer, view_labels)
                    self._printed_token_budget = True
                return review_event_evidence(self._reviewer, clip_path, view_sequences, view_labels, p_fall, self._args)
        except Exception as exc:
            return llm_error_result(f"LLM review failed; event remains suspected: {exc}")

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
    if event.p_fall >= args.llm_review_threshold:
        llm_review = {
            "status": "waiting_for_post_context",
            "is_fall": None,
            "confidence": None,
            "reason": (
                "Suspected fall detected by 3D classifier; collecting post-event "
                f"context until frame {event.release_frame} before LLM review."
            ),
        }
    else:
        llm_review = {
            "status": "classifier_suspicious_below_llm_threshold",
            "is_fall": None,
            "confidence": 0.0,
            "reason": (
                f"3D classifier p_fall={event.p_fall:.3f} is suspicious but below "
                f"LLM review threshold {args.llm_review_threshold:.3f}; event remains suspected unless a stronger trigger merges in."
            ),
        }
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
        llm_review=llm_review,
        trigger_reason=event.trigger_reason,
        event_id=event.event_id,
    )
    payload["video_released"] = False
    payload["event_timestamp_sec"] = round(event.trigger_frame / max(args.fps, 1e-6), 3)
    payload["evidence_release_frame"] = event.release_frame
    return payload


def find_pending_event(
    pending_events: list[PendingEvent],
    track_id: int | None,
) -> PendingEvent | None:
    for event in pending_events:
        if event.track_id == track_id:
            return event
    return None


def should_merge_trigger(
    event: PendingEvent,
    *,
    frame_index: int,
    p_fall: float,
    args: argparse.Namespace,
) -> bool:
    if frame_index > event.trigger_frame + args.event_merge_frames:
        return False
    return p_fall > event.p_fall + args.event_merge_min_probability_delta


def merge_trigger_into_event(
    event: PendingEvent,
    *,
    frame_index: int,
    p_fall: float,
    quality: float | None,
    trigger_reason: str,
    args: argparse.Namespace,
) -> None:
    event.trigger_frame = frame_index
    event.release_frame = frame_index + args.event_post_frames
    event.p_fall = p_fall
    event.quality = quality
    event.trigger_reason = f"{trigger_reason}; merged into existing event episode"
    event.last_update_frame = frame_index


def make_view_frame_sequences(
    buffered_frames: list[BufferedFrame],
    camera_count: int,
    center_frame: int,
    before_count: int,
    after_count: int,
    stride: int,
) -> list[list[np.ndarray]]:
    if not buffered_frames:
        return []
    by_frame = {item.frame_index: item for item in buffered_frames}
    offsets = [i * stride for i in range(-before_count, after_count + 1)]
    target_frames = [center_frame + offset for offset in offsets]
    if not target_frames:
        return []

    sequences: list[list[np.ndarray]] = []
    for view_idx in range(camera_count):
        frames: list[np.ndarray] = []
        for frame_index in target_frames:
            item = by_frame.get(frame_index)
            if item is None:
                continue  # tolerate gaps from decoding errors
            views = item.views
            if views is None or view_idx >= len(views):
                continue
            frames.append(views[view_idx])
        if frames:
            sequences.append(frames)
    return sequences


def make_centered_view_sequences(
    buffered_frames: list[BufferedFrame],
    camera_count: int,
    center_frame: int,
    stride: int,
    sample_count: int = 32,
) -> list[list[np.ndarray]]:
    """Sample a centered frame sequence, shifting into the available evidence window if needed."""
    if not buffered_frames:
        return []
    stride = max(1, stride)
    sample_count = max(1, sample_count)
    by_frame = {item.frame_index: item for item in buffered_frames}
    available_start = buffered_frames[0].frame_index
    available_end = buffered_frames[-1].frame_index

    start_frame = center_frame - (sample_count // 2) * stride
    end_frame = start_frame + (sample_count - 1) * stride
    if start_frame < available_start:
        shift = available_start - start_frame
        start_frame += shift
        end_frame += shift
    if end_frame > available_end:
        shift = end_frame - available_end
        start_frame -= shift
        end_frame -= shift
    start_frame = max(start_frame, available_start)

    target_frames = [start_frame + i * stride for i in range(sample_count)]

    sequences: list[list[np.ndarray]] = []
    for view_idx in range(camera_count):
        frames: list[np.ndarray] = []
        for frame_index in target_frames:
            item = by_frame.get(frame_index)
            if item is None:
                continue
            views = item.views
            if views is None or view_idx >= len(views):
                continue
            frames.append(views[view_idx])
        if frames:
            sequences.append(frames)
    return sequences


def finalize_pending_event(
    *,
    args: argparse.Namespace,
    clip_buffer: RollingClipBuffer,
    review_manager: LLMReviewManager,
    event: PendingEvent,
) -> Future | None:
    if event.p_fall < args.llm_review_threshold:
        def send_suspicious_without_llm() -> None:
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
                    "status": "skipped_below_llm_threshold",
                    "is_fall": None,
                    "confidence": 0.0,
                    "reason": (
                        f"3D classifier p_fall={event.p_fall:.3f} is below "
                        f"LLM review threshold {args.llm_review_threshold:.3f}; kept as suspected."
                    ),
                },
                trigger_reason=event.trigger_reason,
                event_id=event.event_id,
            )
            payload["video_released"] = False
            payload["event_timestamp_sec"] = round(event.trigger_frame / max(args.fps, 1e-6), 3)
            send_or_print(payload, args)

        return review_manager.submit(send_suspicious_without_llm)

    output_path = make_clip_path(event.scenario, event.trigger_frame, event.status, Path(args.clip_dir))
    start_frame = event.trigger_frame - args.event_pre_frames + 1
    end_frame = event.trigger_frame + args.event_post_frames

    # Clamp evidence window to available buffer frames (handles scenario-edge triggers)
    buf_range = clip_buffer.frame_range()
    if buf_range is not None:
        buf_start, buf_end = buf_range
        start_frame = max(start_frame, buf_start)
        end_frame = min(end_frame, buf_end)

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
    llm_center_frame = event.trigger_frame - args.classifier_window_steps // 2
    view_sequences = make_centered_view_sequences(
        evidence_frames,
        camera_count=len(args.cameras),
        center_frame=llm_center_frame,
        stride=args.llm_evidence_sample_stride,
        sample_count=args.llm_sample_frames,
    )
    view_labels = [f"cam{camera_id}" for camera_id in args.cameras]
    clip_path = clip_info.path

    def review_and_send() -> None:
        llm_review = review_manager.review_event_evidence(clip_path, view_sequences, view_labels, event.p_fall)
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
            "views": len(view_sequences),
            "frames_per_view": len(view_sequences[0]) if view_sequences else 0,
            "source_window_frames": len(evidence_frames),
            "center_frame": llm_center_frame,
            "stride_frames": args.llm_evidence_sample_stride,
            "sheets_per_view": int(np.ceil((len(view_sequences[0]) if view_sequences else 0) / 4)),
            "mode": llm_review.get("review_mode", args.llm_review_mode),
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

    gpu_lock = threading.RLock()
    pose_uses_gpu = is_cuda_device(args.pose_device)
    classifier_uses_gpu = is_cuda_device(args.classifier_device)

    detector = PoseDetector(device=args.pose_device, enable_fallback=args.pose_fallback)
    projections = get_projections(camera_ids)
    tri = Triangulator(projections)
    classifier = load_classifier(args)
    review_manager = LLMReviewManager(enabled=args.llm, args=args, gpu_lock=gpu_lock)
    pending_reviews: list[Future] = []
    emitted_event_keys: set[tuple[int, int]] = set()
    rng = random.Random(args.seed)
    scenario_pool = args.scenarios or [args.scenario]
    scenario_index = 0
    start_time = time.time()
    print(
        f"[client] scenarios={scenario_pool} cameras={camera_ids} "
        f"review_threshold={args.threshold_low:.3f} "
        f"buffer={args.buffer_frames} export={args.event_clip_frames}@{args.clip_fps:g}fps "
        f"pre={args.event_pre_frames} post={args.event_post_frames} "
        f"pose_device={detector.device}"
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
                    original_views = getattr(reader, "last_original_frames", None)
                    llm_views = original_views if original_views and len(original_views) == len(frames) else frames
                    clip_buffer.append(frame_idx, collage, views=llm_views)
    
                    with maybe_gpu_lock(gpu_lock, args.llm_gpu_lock and pose_uses_gpu):
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
                            state.features.clear()
                            state.qualities.clear()
                            state.skeletons.clear()
                            state.last_feature_frame = None
                            if args.verbose:
                                print(f"[gate] frame={frame_idx} track={track_id} rejected: {gate_reason}")
                            continue
    
                        if state.last_feature_frame is not None and frame_idx != state.last_feature_frame + 1:
                            state.features.clear()
                            state.qualities.clear()
                            state.skeletons.clear()

                        update_temporal_state(state, skel, fps=args.fps)
                        state.qualities.append(quality)
                        prev_skel = state.skeletons[-1] if state.skeletons else None
                        features = compute_features(skel, prev_skel, floor_estimator=state.floor_estimator)
                        if features is None:
                            state.features.clear()
                            state.qualities.clear()
                            state.skeletons.clear()
                            state.last_feature_frame = None
                            continue
                        features[3] = features[2] - state.features[-1][2] if state.features else 0.0
                        state.features.append(features)
                        state.skeletons.append(skel.copy())
                        state.last_feature_frame = frame_idx
    
                        required_features = args.classifier_window_steps
                        if len(state.features) < required_features or frame_idx % args.classify_stride != 0:
                            continue
                        state.rejected_frames = 0

                        recent_features = list(state.features)[-required_features:]
                        window = np.array(
                            recent_features,
                            dtype=np.float32,
                            copy=True,
                        )
                        for i in range(len(window)):
                            window[i, 3] = window[i, 2] - window[i - 1, 2] if i > 0 else 0.0
                        mean_quality = float(
                            np.mean(
                                [
                                    q["score"]
                                    for q in list(state.qualities)[-args.classifier_window_steps:]
                                ]
                            )
                        )
                        if mean_quality < args.quality_threshold:
                            if args.verbose:
                                print(
                                    f"[classifier] frame={frame_idx} track={track_id} "
                                    f"skipped low window quality={mean_quality:.2f}"
                                )
                            continue
                        with maybe_gpu_lock(gpu_lock, args.llm_gpu_lock and classifier_uses_gpu):
                            cls_result = classifier.predict(window)
                        p_fall = float(cls_result["fall_prob"])
                        status = status_from_probability(p_fall, args.threshold_low, args.threshold_high)
                        if args.log_prob_every and frame_idx % args.log_prob_every == 0:
                            print(
                                f"[prob] frame={frame_idx} track={track_id} "
                                f"p_fall={p_fall:.3f} status={status} quality={mean_quality:.2f}"
                            )
                        forced = args.force_trigger_frame is not None and frame_idx >= args.force_trigger_frame
                        if forced and state.last_event_frame < 0:
                            p_fall = args.force_p_fall
                            status = status_from_probability(p_fall, args.threshold_low, args.threshold_high)
                            gate_reason = "forced play trigger for smoke/demo"

                        state.status = status
                        if status == "normal":
                            continue
                        pending_event = find_pending_event(pending_events, track_id)
                        if pending_event is None and frame_idx < state.event_lock_until_frame:
                            if args.verbose:
                                print(
                                    f"[event-lock] frame={frame_idx} track={track_id} "
                                    f"locked_until={state.event_lock_until_frame}"
                                )
                            continue
                        if len(clip_buffer) < args.event_pre_frames:
                            if args.verbose:
                                print(
                                    f"[event] frame={frame_idx} status={status} waiting for "
                                    f"{args.event_pre_frames} pre-event frames ({len(clip_buffer)} ready)"
                            )
                            continue
                        event_key = (scenario, frame_idx)
                        if args.dedupe_scenario_frame and event_key in emitted_event_keys:
                            if args.verbose:
                                print(f"[event-dedupe] scenario={scenario} frame={frame_idx} already emitted")
                            continue
                        trigger_reason = (
                            f"classifier p_fall={p_fall:.3f}; status={status}; "
                            f"quality={mean_quality:.2f}; {gate_reason}"
                        )
                        if pending_event is not None:
                            if should_merge_trigger(
                                pending_event,
                                frame_index=frame_idx,
                                p_fall=p_fall,
                                args=args,
                            ):
                                merge_trigger_into_event(
                                    pending_event,
                                    frame_index=frame_idx,
                                    p_fall=p_fall,
                                    quality=mean_quality,
                                    trigger_reason=trigger_reason,
                                    args=args,
                                )
                                emitted_event_keys.add((scenario, frame_idx))
                                state.last_event_frame = frame_idx
                                state.event_lock_until_frame = frame_idx + args.event_lock_frames
                                if args.verbose:
                                    print(
                                        f"[event-merge] event_id={pending_event.event_id} "
                                        f"track={track_id} trigger={frame_idx} "
                                        f"release={pending_event.release_frame} p_fall={p_fall:.3f}"
                                    )
                            continue

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
                            last_update_frame=frame_idx,
                        )
                        pending_events.append(event)
                        emitted_event_keys.add(event_key)
                        send_or_print(make_initial_suspected_payload(args=args, event=event), args)
                        state.last_event_frame = frame_idx
                        state.event_lock_until_frame = frame_idx + args.event_lock_frames
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
                        event_key = (scenario, frame_idx)
                        if args.dedupe_scenario_frame and event_key in emitted_event_keys:
                            if args.verbose:
                                print(f"[event-dedupe] scenario={scenario} frame={frame_idx} already emitted")
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
                        emitted_event_keys.add(event_key)
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

            # --- Process events stuck at scenario boundary (release_frame > last_frame) ---
            for event in list(pending_events):
                if event.release_frame > last_frame_idx:
                    if args.verbose:
                        print(
                            f"[event] scenario-end finalizing event_id={event.event_id} "
                            f"trigger={event.trigger_frame} release={event.release_frame} "
                            f"last_frame={last_frame_idx}"
                        )
                future = finalize_pending_event(
                    args=args,
                    clip_buffer=clip_buffer,
                    review_manager=review_manager,
                    event=event,
                )
                if future is not None:
                    pending_reviews.append(future)
                pending_events.remove(event)

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
    parser.add_argument("--pose-device", default="cpu",
                        help="YOLO pose device. Default cpu keeps the demo GPU reserved for Qwen3-VL.")
    parser.add_argument("--classifier-device", default="cpu",
                        help="3D classifier device. Default cpu avoids GPU contention with Qwen3-VL.")
    parser.add_argument("--threshold-low", type=float, default=0.075)
    parser.add_argument("--threshold-high", type=float, default=1.0,
                        help="Metadata only in current demo policy; all p_fall >= threshold-low goes to LLM review")
    parser.add_argument("--quality-threshold", type=float, default=0.45)
    parser.add_argument("--min-temporal-quality", type=float, default=0.15)
    parser.add_argument("--max-centroid-speed", type=float, default=150.0)
    parser.add_argument("--max-speed-multiplier", type=float, default=8.0)
    parser.add_argument("--max-speed-rejects", type=int, default=10,
                        help="Accept and resync after this many consecutive speed-gate rejects")
    parser.add_argument("--classify-stride", type=int, default=5)
    parser.add_argument("--classifier-window-steps", type=int, default=40,
                        help="Consecutive 3D feature frames passed to the classifier")
    parser.add_argument("--max-people", type=int, default=1)
    parser.add_argument("--buffer-frames", type=int, default=600)
    parser.add_argument("--event-pre-frames", type=int, default=90,
                        help="Frames retained before and including the trigger frame")
    parser.add_argument("--event-post-frames", type=int, default=151,
                        help="Frames collected after the trigger before exporting evidence")
    parser.add_argument("--event-clip-frames", type=int, default=241,
                        help="Total evidence clip frames; defaults to 90 pre + 151 post")
    parser.add_argument("--event-lock-frames", type=int, default=300,
                        help="After an event episode, keep YOLO/3D/features running but suppress new events")
    parser.add_argument("--event-merge-frames", type=int, default=180,
                        help="Allow stronger triggers within this many frames to update one pending event")
    parser.add_argument("--event-merge-min-probability-delta", type=float, default=0.0,
                        help="Minimum p_fall increase required to move a pending event trigger")
    parser.add_argument("--dedupe-scenario-frame", dest="dedupe_scenario_frame", action="store_true",
                        help="Suppress duplicate events with the same dataset scenario and trigger frame")
    parser.add_argument("--no-dedupe-scenario-frame", dest="dedupe_scenario_frame", action="store_false")
    parser.set_defaults(dedupe_scenario_frame=True)
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
    parser.add_argument("--n-gpu-layers", type=int, default=20)
    parser.add_argument("--n-ctx", type=int, default=10240)
    parser.add_argument("--n-batch", type=int, default=2048,
                        help="llama.cpp logical batch size; keep modest on 8GB GPUs")
    parser.add_argument("--n-ubatch", type=int, default=64,
                        help="llama.cpp physical micro-batch size; lower values reduce VRAM spikes")
    parser.add_argument("--llm-frame-stride", type=int, default=10)
    parser.add_argument("--llm-max-frames", type=int, default=3)
    parser.add_argument("--llm-sheet-frames", type=int, default=9)
    parser.add_argument("--llm-sheet-stride", type=int, default=29)
    parser.add_argument("--llm-sheet-tile-width", type=int, default=480,
                        help="Tile width for each frame inside a 2x2 LLM contact sheet; 480 creates 960x960 sheets")
    parser.add_argument("--llm-evidence-sample-stride", type=int, default=4,
                        help="Sample every N frames around the suspicious center for per-camera LLM review")
    parser.add_argument("--llm-sample-frames", type=int, default=32,
                        help="Number of frames sampled per camera for LLM review")
    parser.add_argument("--llm-3d-fall-threshold", type=float, default=0.80,
                        help="If all camera LLM reviews say non-fall but p_fall reaches this threshold, still classify as fall")
    parser.add_argument("--llm-review-threshold", type=float, default=0.10,
                        help="Only run LLM review for suspected events whose p_fall reaches this threshold")
    parser.add_argument("--llm-required-fall-views", type=int, default=2,
                        help="Number of camera LLM reviews that must support fall before upgrading to fall")
    parser.add_argument("--llm-pre-samples", type=int, default=13,
                        help="Deprecated; old centered LLM sampling before the classifier-window center")
    parser.add_argument("--llm-post-samples", type=int, default=13,
                        help="Deprecated; old centered LLM sampling after the classifier-window center")
    parser.add_argument("--llm-temporal-stride", type=int, default=5,
                        help="Deprecated; old centered LLM frame stride")
    parser.add_argument("--llm-image-max-side", type=int, default=0,
                        help="Downscale each LLM image so its longest side is at most this many pixels; 0 disables resizing")
    parser.add_argument("--llm-min-confidence", type=float, default=0.60)
    parser.add_argument("--llm-review-mode", choices=["direct"], default="direct",
                        help="Three camera contact sheets sent in one Qwen3-VL call for direct judgment")
    parser.add_argument("--llm-gpu-lock", dest="llm_gpu_lock", action="store_true",
                        help="Serialize LLM and CUDA YOLO/classifier calls on one GPU")
    parser.add_argument("--no-llm-gpu-lock", dest="llm_gpu_lock", action="store_false",
                        help="Allow LLM and CUDA detector/classifier calls to run concurrently")
    parser.set_defaults(llm_gpu_lock=True)
    parser.add_argument("--llm-debug-tokens", action="store_true",
                        help="Print tokenizer counts for the LLM prompts on first review")
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
