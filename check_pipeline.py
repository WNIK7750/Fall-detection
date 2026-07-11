"""
Quick smoke test: MCFD reader + YOLOv8n-pose batch inference.

Reads a few frames from MCFD scenario 1, runs YOLO batch inference on 3 views,
prints keypoint summary and saves a debug visualization image.

Usage:
    python check_pipeline.py
"""

import os
import sys
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from pipeline.video_reader import MCFDReader, load_annotations, find_fall_segments
from pipeline.pose_inference import PoseDetector, MultiViewPoseVisualizer

ROOT = os.path.join(os.path.dirname(__file__), "dataset", "mcfd")

def main():
    # 1. Check annotations
    anns = load_annotations(ROOT)
    print(f"Annotations: {len(anns)} rows")
    for s in range(1, 5):
        falls = find_fall_segments(anns, s)
        print(f"  Scenario {s}: {len(falls)} fall segments {falls}")

    # 2. Load YOLO
    detector = PoseDetector()
    print(f"\nYOLOv8n-pose device: {detector.device}")

    # 3. Read and infer first 30 frames of scenario 1 (3 cams)
    print("\n--- Testing MCFDReader + PoseDetector ---")
    with MCFDReader(ROOT, scenario=1, camera_ids=[1, 4, 7]) as reader:
        print(f"Total frames: {reader.total_frames}, {reader.num_views} views")

        total_keypoints = 0
        total_frames = 0

        for i, frames in enumerate(reader):
            if i >= 30:
                break
            kps = detector.infer_batch(frames)
            total_frames += 1

            for v, kp in enumerate(kps):
                valid = int((kp[:, 2] > 0.3).sum())
                total_keypoints += valid

            if i == 0 or i == 15 or i == 29:
                print(f"  Frame {i:3d}: valid kps = {[(kp[:,2]>0.3).sum() for kp in kps]}")

    avg_kp = total_keypoints / (total_frames * 3) if total_frames > 0 else 0
    print(f"\nAverage keypoints per view: {avg_kp:.1f}/17 ({total_frames} frames, 3 views)")

    # 4. Save a debug visualization
    print("\n--- Saving debug visualization ---")
    with MCFDReader(ROOT, scenario=1, camera_ids=[1, 4, 7], start_frame=50) as reader:
        for frames in reader:
            kps = detector.infer_batch(frames)
            vis_frames = []
            for v, (frame, kp) in enumerate(zip(frames, kps)):
                vis = MultiViewPoseVisualizer.draw_keypoints(frame, kp)
                cv2.putText(vis, f"View {v+1} (cam {(1,4,7)[v]})", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                vis_frames.append(vis)
            combined = np.hstack(vis_frames)
            out_path = os.path.join(os.path.dirname(__file__), "results", "debug_pose.png")
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            cv2.imwrite(out_path, combined)
            print(f"Saved: {out_path}")
            break

    print("\n=== ALL CHECKS PASSED ===")

if __name__ == "__main__":
    main()
