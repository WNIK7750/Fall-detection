"""
Complete multi-view fall detection demo pipeline.

Modules: YOLO → Triangulation → 3D Features → Classifier → LLM Review → Display

Usage:
    # Visual demo (OpenCV windows)
    python demo_multiview.py --scenario 1

    # With LLM review enabled
    python demo_multiview.py --scenario 4 --llm

    # Save output video
    python demo_multiview.py --scenario 1 --output demo_output.avi

Controls (in OpenCV window):
    q / ESC : quit
    SPACE  : pause/resume
    n      : next scenario
"""

import os
import sys
import argparse
import time
from collections import deque

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from pipeline.video_reader import MCFDReader, load_annotations, find_fall_segments
from pipeline.pose_inference import PoseDetector, MultiViewPoseVisualizer, MultiPersonTracker
from pipeline.triangulation import Triangulator, KalmanSmoother
from pipeline.features_3d import compute_features, extract_window, FloorEstimator
from pipeline.classifier import FallClassifier, HeuristicClassifier
from pipeline.llm_reviewer import LLMFallReviewer, make_multiview_collage
from pipeline.calibration import get_projections, undistort_keypoints

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_3D_CHECKPOINT = os.path.join(
    BASE_DIR,
    "checkpoints",
    "enhanced_3d_20260712_185406",
    "best_model.pth",
)


# --- Color constants ---
GREEN  = (0, 255, 0)
YELLOW = (0, 255, 255)
RED    = (0, 0, 255)
WHITE  = (255, 255, 255)
BLUE   = (255, 0, 0)
ORANGE = (0, 165, 255)


def draw_status_bar(img: np.ndarray, text: str, color: tuple[int, int, int]) -> np.ndarray:
    """Draw a status bar at the top of the image."""
    h, w = img.shape[:2]
    bar_h = 40
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
    cv2.putText(img, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    return img


def draw_3d_skeleton(img: np.ndarray, skeleton_3d: np.ndarray, size: int = 200) -> np.ndarray:
    """Draw a miniature 3D skeleton (top-down view)."""
    from pipeline.pose_inference import SKELETON_EDGES

    canvas = np.zeros((size, size, 3), dtype=np.uint8)

    # Replace NaN nose with mid-shoulder for visualization
    skel = skeleton_3d.copy()
    if np.isnan(skel[0]).any() and not np.isnan(skel[5]).any() and not np.isnan(skel[6]).any():
        skel[0] = (skeleton_3d[5] + skeleton_3d[6]) / 2.0

    if np.isnan(skel).all():
        cv2.putText(canvas, "No 3D data", (20, size//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)
        return canvas

    valid = skel[~np.isnan(skel).any(axis=1)]
    if len(valid) < 2:
        return canvas

    xs, zs = valid[:, 0], valid[:, 2]
    x_range, z_range = xs.max() - xs.min() + 1e-6, zs.max() - zs.min() + 1e-6
    scale = min((size - 20) / max(x_range, z_range), 50)
    ox, oy = size // 2, size // 2

    for (i, j) in SKELETON_EDGES:
        if i < len(skel) and j < len(skel):
            if not np.isnan(skel[i]).any() and not np.isnan(skel[j]).any():
                pt1 = (int(ox + skel[i, 0] * scale), int(oy - skel[i, 2] * scale))
                pt2 = (int(ox + skel[j, 0] * scale), int(oy - skel[j, 2] * scale))
                cv2.line(canvas, pt1, pt2, GREEN, 1)

    for i in range(min(len(skel), 17)):
        if not np.isnan(skel[i]).any():
            pt = (int(ox + skel[i, 0] * scale), int(oy - skel[i, 2] * scale))
            cv2.circle(canvas, pt, 2, RED, -1)

    cv2.putText(canvas, "3D (top-down)", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, WHITE, 1)
    return canvas


def build_llm_context_collage(
    frames: list[np.ndarray],
    tracked: list[list[tuple[int, np.ndarray]]],
    camera_ids: list[int],
    current_status: str,
    current_quality: float,
    evidence_lines: list[str],
) -> np.ndarray:
    """Create a collage with pose overlays and compact text evidence."""
    vis_frames = []
    colors = [GREEN, (255, 255, 0), (0, 255, 255), (255, 0, 255)]
    for v, frame in enumerate(frames):
        vis = frame.copy()
        v_tracks = tracked[v] if v < len(tracked) else []
        for pidx, (track_id, kp) in enumerate(v_tracks):
            color = colors[pidx % len(colors)]
            vis = MultiViewPoseVisualizer.draw_keypoints(vis, kp, color=color)
            valid = kp[:, 2] > 0.3
            if valid.any():
                cx, cy = int(kp[valid, 0].mean()), int(kp[valid, 1].min()) - 10
                cv2.putText(vis, f"T{track_id}", (cx - 15, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        status_text = f"Cam{camera_ids[v]} | {current_status.upper()} | Q={current_quality:.2f}"
        vis = draw_status_bar(vis, status_text, {"normal": GREEN, "suspected": YELLOW, "fall": RED}.get(current_status, WHITE))
        vis_frames.append(vis)

    if evidence_lines:
        footer = np.zeros_like(vis_frames[0])
        y = 24
        for line in evidence_lines[:4]:
            cv2.putText(footer, line[:80], (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, WHITE, 1)
            y += 22
        vis_frames.append(footer)

    return np.hstack(vis_frames)


def run_demo(args):
    data_root = os.path.join(os.path.dirname(__file__), "dataset", "mcfd")
    camera_ids = args.cameras or [1, 4, 7]

    # Load annotations for ground-truth labels
    annotations = load_annotations(data_root)

    # --- Init modules ---
    print("Loading YOLOv8n-pose...")
    detector = PoseDetector(enable_fallback=False)

    print("Loading triangulation...")
    projections = get_projections(camera_ids)
    tri = Triangulator(projections)
    smoother = KalmanSmoother()

    # Multi-person tracker — maintains consistent IDs across frames
    tracker = MultiPersonTracker(num_views=len(camera_ids))

    # Classifier — ML model or heuristic fallback
    classifier = None
    if args.checkpoint:
        print(f"Loading classifier: {args.checkpoint}")
        try:
            classifier = FallClassifier(
                checkpoint_path=args.checkpoint,
                threshold_low=args.threshold_low,
                threshold_high=args.threshold_high,
            )
            print(f"  Params: {classifier.num_params:,}")
        except (FileNotFoundError, RuntimeError, TypeError, ValueError) as e:
            print(f"  WARNING: {e}")
    if classifier is None:
        print("Using heuristic classifier (rule-based fallback)")
        classifier = HeuristicClassifier()

    # LLM Reviewer (optional)
    reviewer = None
    if args.llm:
        base = os.path.dirname(__file__)
        model_path = os.path.join(base, "models", "llm", "Qwen3VL-4B-Instruct-Q4_K_M.gguf")
        mmproj_path = os.path.join(base, "models", "llm", "mmproj-Qwen3VL-4B-Instruct-Q8_0.gguf")
        print("Loading LLM reviewer...")
        reviewer = LLMFallReviewer(model_path, mmproj_path, n_gpu_layers=-1)

    # --- Video output ---
    video_writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        video_writer = cv2.VideoWriter(args.output, fourcc, 20, (1600, 600))

    # --- Main loop ---
    scenario = args.scenario
    print(f"\n=== Scenario {scenario} ===")
    fall_segments = find_fall_segments(annotations, scenario)
    print(f"Fall segments: {fall_segments}")

    tracker.reset()  # fresh track IDs for each scenario

    # Per-track person state: track_id -> {features:deque, skeletons:deque, smoother, status}
    # (track_id is per-view; we use view0's track IDs as the reference for display)
    person_states: dict[int, dict] = {}
    frame_buffer: list = []  # for LLM review context

    current_status = "normal"
    current_quality = 1.0
    llm_result = None
    main_cls_result = None
    main_feature_window = None
    paused = False
    frame_idx = 0

    with MCFDReader(data_root, scenario, camera_ids, undistort=False) as reader:
        cv2.namedWindow("Fall Detection Demo", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Fall Detection Demo", 1600, 600)

        for frames in reader:
            if paused:
                key = cv2.waitKey(30) & 0xFF
                if key == ord(" "): paused = False
                elif key == ord("q") or key == 27: break
                continue

            frame_idx += 1
            t_start = time.time()

            # 1. YOLO pose detection — ALL persons per view
            raw_persons = detector.infer_batch_all(frames)

            # 2. Multi-person tracking — stable IDs across frames
            tracked = tracker.update(raw_persons)
            # tracked[v] = [(track_id, keypoints), ...] sorted by track_id

            # Use view0's track IDs as the global person index for display
            view0_tracks = tracked[0] if len(tracked) > 0 else []
            if args.max_people > 0:
                view0_tracks = view0_tracks[:args.max_people]
            num_people = len(view0_tracks)

            if num_people == 0:
                # No one detected — show empty frame
                combined = np.hstack(frames)
                cv2.rectangle(combined, (0, combined.shape[0] - 30),
                              (combined.shape[1], combined.shape[0]), (10, 10, 10), -1)
                cv2.putText(combined, "FPS: -- | [q]uit [space]pause",
                            (10, combined.shape[0] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, WHITE, 1)
                cv2.imshow("Fall Detection Demo", combined)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27: break
                elif key == ord(" "): paused = True
                continue

            # 3. Process each tracked person independently
            all_statuses: list[str] = []
            main_skel = None

            for pidx, (track_id, kp_view0) in enumerate(view0_tracks):
                # Ensure per-track state exists
                if track_id not in person_states:
                    person_states[track_id] = {
                        'features': deque(maxlen=60),
                        'qualities': deque(maxlen=60),
                        'skeletons': deque(maxlen=60),
                        'floor_estimator': FloorEstimator(),
                        'smoother': KalmanSmoother(),
                        'status': 'normal',
                    }
                pstate = person_states[track_id]

                # Collect raw keypoints across all views for this person.
                # Keep raw coordinates for drawing; undistort keypoints only
                # before triangulation so YOLO sees the natural camera image.
                # Map track_id in each view to the corresponding detection
                person_kps_raw = []
                for v in range(len(tracked)):
                    # Find the detection with matching track_id, or use the same index
                    # if tracking is consistent across views (common for single person)
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

                # Triangulation
                raw_skel = tri.triangulate(person_kps)
                quality = tri.compute_quality(person_kps, raw_skel)
                pstate['qualities'].append(quality)
                skel = pstate['smoother'].update(raw_skel)

                if pidx == 0:
                    main_skel = skel.copy()

                # Features
                prev_skel = pstate['skeletons'][-1] if pstate['skeletons'] else None
                features = compute_features(skel, prev_skel, floor_estimator=pstate['floor_estimator'])
                if features is not None:
                    if pstate['features']:
                        features[3] = features[2] - pstate['features'][-1][2]
                    else:
                        features[3] = 0.0
                    pstate['features'].append(features)
                pstate['skeletons'].append(skel.copy())

                # Classification (every stride=5 frames)
                status = pstate.get('status', 'normal')
                if classifier and len(pstate['features']) >= 20 and frame_idx % 5 == 0:
                    window = np.array(list(pstate['features'])[-20:])
                    cls_result = classifier.predict(window)
                    status = cls_result["class_name"]
                    recent_quality = [q["score"] for q in list(pstate['qualities'])[-20:]]
                    mean_quality = float(np.mean(recent_quality)) if recent_quality else 0.0
                    if mean_quality < args.quality_threshold and status == "fall":
                        status = "suspected"
                    if pidx == 0:
                        main_cls_result = cls_result
                        main_feature_window = window
                pstate['status'] = status
                all_statuses.append(status)

                if pidx == 0:
                    current_status = status
                    current_quality = quality["score"]

            current_raw_collage = make_multiview_collage(frames, labels=[f"Cam{cid}" for cid in camera_ids])
            frame_buffer.append(current_raw_collage)
            if len(frame_buffer) > 10:
                frame_buffer.pop(0)

            # 4. LLM Review (on "suspected" for main person)
            if current_status == "suspected" and reviewer and reviewer.ready:
                if len(frame_buffer) >= args.llm_frames:
                    if main_cls_result:
                        cls_line = f"3D fall prob={main_cls_result['fall_prob']:.3f}"
                        evidence = (
                            f"3D classifier: {main_cls_result['class_name']} "
                            f"(p_fall={main_cls_result['fall_prob']:.3f}, "
                            f"thresholds={main_cls_result['threshold_low']:.2f}/{main_cls_result['threshold_high']:.2f}); "
                            f"recent skeleton quality={current_quality:.2f}; "
                            f"window_shape={None if main_feature_window is None else list(main_feature_window.shape)}"
                        )
                    else:
                        cls_line = "3D fall prob=n/a"
                        evidence = f"3D classifier result unavailable; recent skeleton quality={current_quality:.2f}"
                    review_frames_raw = frame_buffer[-args.llm_frames:]
                    review_frames_ctx = [
                        build_llm_context_collage(
                            frames,
                            tracked,
                            camera_ids,
                            current_status,
                            current_quality,
                            [
                                cls_line,
                                f"quality={current_quality:.2f}",
                                f"window={len(main_feature_window) if main_feature_window is not None else 0}x10",
                            ],
                        )
                    ]
                    llm_result = reviewer.review_dual(
                        raw_frames=review_frames_raw,
                        context_frames=review_frames_ctx,
                        evidence=evidence,
                        min_confidence=args.llm_min_confidence,
                    )
                    current_status = llm_result.get("final_status", current_status)
                frame_buffer = []

            # 5. Visualization — draw ALL persons per view
            vis_frames = []
            colors = [GREEN, (255, 255, 0), (0, 255, 255), (255, 0, 255)]
            for v, frame in enumerate(frames):
                vis = frame.copy()
                v_tracks = tracked[v] if v < len(tracked) else []

                for pidx, (track_id, kp) in enumerate(v_tracks):
                    color = colors[pidx % len(colors)]
                    vis = MultiViewPoseVisualizer.draw_keypoints(vis, kp, color=color)
                    # Draw track ID label near the person
                    valid = kp[:, 2] > 0.3
                    if valid.any():
                        cx, cy = int(kp[valid, 0].mean()), int(kp[valid, 1].min()) - 10
                        cv2.putText(vis, f"T{track_id}", (cx - 15, cy),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

                # Build multi-person status text
                status_parts = []
                for pidx, (track_id, _) in enumerate(view0_tracks):
                    s = all_statuses[pidx] if pidx < len(all_statuses) else "?"
                    status_parts.append(f"T{track_id}:{s}")
                status_text = (f"Cam{camera_ids[v]} | F{frame_idx} | "
                               f"{current_status.upper()} | Q={current_quality:.2f} | "
                               f"{' '.join(status_parts)}")
                vis = draw_status_bar(vis, status_text,
                    {"normal": GREEN, "suspected": YELLOW, "fall": RED}.get(current_status, WHITE))
                vis_frames.append(vis)

            # 3D mini-view using main person's skeleton (already computed, no flicker)
            if main_skel is None:
                main_skel = np.full((17, 3), np.nan, dtype=np.float64)
            skel_vis = draw_3d_skeleton(np.zeros((200, 200, 3), dtype=np.uint8), main_skel)
            skel_vis = cv2.resize(skel_vis, (vis_frames[0].shape[1], vis_frames[0].shape[0]))
            vis_frames.append(skel_vis)

            # Combine into one wide image
            combined = np.hstack(vis_frames)

            # LLM result overlay
            if llm_result:
                llm_text = f"LLM: {llm_result.get('final_status', llm_result.get('reason', ''))[:80]}"
                cv2.putText(combined, llm_text, (10, combined.shape[0] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, ORANGE, 1)

            # Footer bar
            h, w = combined.shape[:2]
            cv2.rectangle(combined, (0, h - 30), (w, h), (10, 10, 10), -1)
            fps = 1.0 / max(time.time() - t_start, 0.001)
            cv2.putText(combined, f"FPS: {fps:.1f} | [q]uit [space]pause",
                        (10, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, WHITE, 1)

            # Show
            cv2.imshow("Fall Detection Demo", combined)

            if video_writer:
                video_writer.write(combined)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
            elif key == ord(" "):
                paused = True

    cv2.destroyAllWindows()
    if video_writer:
        video_writer.release()
    print(f"\nDemo complete. Processed {frame_idx} frames.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-view Fall Detection Demo")
    parser.add_argument("--scenario", type=int, default=1, help="MCFD scenario (1-24)")
    parser.add_argument("--cameras", type=int, nargs="+", default=[1, 4, 7], help="Camera IDs")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_3D_CHECKPOINT,
                        help="Classifier checkpoint path")
    parser.add_argument("--threshold-low", type=float, default=0.30)
    parser.add_argument("--threshold-high", type=float, default=0.40)
    parser.add_argument("--max-people", type=int, default=1,
                        help="Maximum people to triangulate; 1 avoids unreliable cross-view ID matching")
    parser.add_argument("--quality-threshold", type=float, default=0.45,
                        help="Mean recent skeleton quality below this downgrades fall to suspected")
    parser.add_argument("--llm", action="store_true", help="Enable LLM review")
    parser.add_argument("--llm-frames", type=int, default=1,
                        help="Number of recent multiview collage frames to send to the LLM")
    parser.add_argument("--llm-min-confidence", type=float, default=0.60,
                        help="LLM result below this confidence keeps the status suspected")
    parser.add_argument("--output", type=str, default=None, help="Output video file")
    args = parser.parse_args()
    run_demo(args)
