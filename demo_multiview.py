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
from pipeline.pose_inference import PoseDetector, MultiViewPoseVisualizer
from pipeline.triangulation import Triangulator, KalmanSmoother
from pipeline.features_3d import compute_features, extract_window
from pipeline.classifier import FallClassifier
from pipeline.llm_reviewer import LLMFallReviewer
from pipeline.calibration import get_projections


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
    """Draw a miniature 3D skeleton on a 2D canvas (top-down + side view)."""
    from pipeline.pose_inference import SKELETON_EDGES

    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    if np.isnan(skeleton_3d).all():
        cv2.putText(canvas, "No 3D data", (20, size//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)
        return canvas

    # Project 3D to 2D for mini view
    valid = skeleton_3d[~np.isnan(skeleton_3d).any(axis=1)]
    if len(valid) < 2:
        return canvas

    # Top-down view (X-Z plane)
    xs = valid[:, 0]
    zs = valid[:, 2]
    x_range = xs.max() - xs.min() + 1e-6
    z_range = zs.max() - zs.min() + 1e-6
    scale = min((size - 20) / max(x_range, z_range), 50)
    ox, oy = size // 2, size // 2

    for (i, j) in SKELETON_EDGES:
        if i < len(skeleton_3d) and j < len(skeleton_3d):
            if not np.isnan(skeleton_3d[i]).any() and not np.isnan(skeleton_3d[j]).any():
                pt1 = (int(ox + skeleton_3d[i, 0] * scale), int(oy - skeleton_3d[i, 2] * scale))
                pt2 = (int(ox + skeleton_3d[j, 0] * scale), int(oy - skeleton_3d[j, 2] * scale))
                cv2.line(canvas, pt1, pt2, GREEN, 1)

    for i in range(min(len(skeleton_3d), 17)):
        if not np.isnan(skeleton_3d[i]).any():
            pt = (int(ox + skeleton_3d[i, 0] * scale), int(oy - skeleton_3d[i, 2] * scale))
            cv2.circle(canvas, pt, 2, RED, -1)

    cv2.putText(canvas, "3D (top-down)", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, WHITE, 1)
    return canvas


def run_demo(args):
    data_root = os.path.join(os.path.dirname(__file__), "dataset", "mcfd")
    camera_ids = args.cameras or [1, 4, 7]

    # Load annotations for ground-truth labels
    annotations = load_annotations(data_root)

    # --- Init modules ---
    print("Loading YOLOv8n-pose...")
    detector = PoseDetector()

    print("Loading triangulation...")
    projections = get_projections(camera_ids)
    tri = Triangulator(projections)
    smoother = KalmanSmoother()

    # Classifier (optional — requires trained checkpoint)
    classifier = None
    if args.checkpoint:
        print(f"Loading classifier: {args.checkpoint}")
        try:
            classifier = FallClassifier(
                checkpoint_path=args.checkpoint,
                threshold_low=args.threshold_low,
                threshold_high=args.threshold_high,
            )
            print(f"  Params: {classifier.num_params:,}, thresholds: {args.threshold_low}/{args.threshold_high}")
        except FileNotFoundError:
            print(f"  WARNING: Checkpoint not found, classifier disabled.")

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

    # Feature history (20-frame window)
    feature_history: deque = deque(maxlen=60)  # enough for 20 frames at subsample=3
    skeleton_history: deque = deque(maxlen=60)
    frame_buffer: list = []  # for LLM review context

    current_status = "normal"
    llm_result = None
    paused = False
    frame_idx = 0

    with MCFDReader(data_root, scenario, camera_ids) as reader:
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

            # 1. YOLO pose detection
            kps = detector.infer_batch(frames)

            # 2. Triangulation
            skel = tri.triangulate(kps)
            skel = smoother.update(skel)

            # 3. Features
            features = compute_features(skel, skeleton_history[-1] if skeleton_history else None)
            if features is not None:
                feature_history.append(features)
            skeleton_history.append(skel.copy())

            # 4. Classification (every stride=5 frames)
            cls_result = None
            if classifier and len(feature_history) >= 20 and frame_idx % 5 == 0:
                # Take last 20 features
                window = np.array(list(feature_history)[-20:])  # (20, 10)
                window = window.T[np.newaxis, ...]  # (1, 10, 20)
                cls_result = classifier.predict(window[0])  # predict takes (10, 20)
                current_status = cls_result["class_name"]

            # 5. LLM Review (on "suspected")
            if current_status == "suspected" and reviewer and reviewer.ready:
                # Capture keyframes from buffer
                if len(frame_buffer) >= 3:
                    llm_result = reviewer.review(frame_buffer[-3:])
                    if llm_result.get("is_fall"):
                        current_status = "fall"
                    elif llm_result.get("is_fall") is False:
                        current_status = "normal"
                frame_buffer = []  # reset buffer after review

            # Buffer frames for LLM (keep last few seconds)
            frame_buffer.append(frames[1].copy())  # center camera
            if len(frame_buffer) > 10:
                frame_buffer.pop(0)

            # 6. Visualization
            vis_frames = []
            for v, (frame, kp) in enumerate(zip(frames, kps)):
                vis = MultiViewPoseVisualizer.draw_keypoints(frame, kp)
                # Status bar
                status_color = {"normal": GREEN, "suspected": YELLOW, "fall": RED}.get(current_status, WHITE)
                status_text = f"Cam{camera_ids[v]} | Frame {frame_idx} | {current_status.upper()}"
                if cls_result:
                    status_text += f" (P_fall={cls_result['fall_prob']:.2f})"
                vis = draw_status_bar(vis, status_text, status_color)
                vis_frames.append(vis)

            # Add 3D skeleton mini-view
            skel_vis = draw_3d_skeleton(np.zeros((200, 200, 3), dtype=np.uint8), skel)
            vis_frames.append(skel_vis)

            # Combine into one wide image
            combined = np.hstack(vis_frames)

            # LLM result overlay
            if llm_result:
                llm_text = f"LLM: {llm_result.get('reason', '')[:80]}"
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
    parser.add_argument("--checkpoint", type=str, default=None, help="Classifier checkpoint path")
    parser.add_argument("--threshold-low", type=float, default=0.4)
    parser.add_argument("--threshold-high", type=float, default=0.85)
    parser.add_argument("--llm", action="store_true", help="Enable LLM review")
    parser.add_argument("--output", type=str, default=None, help="Output video file")
    args = parser.parse_args()
    run_demo(args)
