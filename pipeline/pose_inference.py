"""
YOLOv8n-pose batch inference for multi-view fall detection.

Runs YOLO pose estimation on 3 camera views simultaneously using native batch API.

Usage:
    detector = PoseDetector()
    keypoints = detector.infer_batch([frame1, frame2, frame3])
    # keypoints[i]: (17, 3) — (x, y, confidence) for 17 COCO keypoints
"""

from typing import Optional

import cv2
import numpy as np
from ultralytics import YOLO

# COCO 17 keypoint indices and names (for reference)
COCO_KEYPOINTS = [
    "nose",           # 0
    "left_eye",       # 1
    "right_eye",      # 2
    "left_ear",       # 3
    "right_ear",      # 4
    "left_shoulder",  # 5
    "right_shoulder", # 6
    "left_elbow",     # 7
    "right_elbow",    # 8
    "left_wrist",     # 9
    "right_wrist",    # 10
    "left_hip",       # 11
    "right_hip",      # 12
    "left_knee",      # 13
    "right_knee",     # 14
    "left_ankle",     # 15
    "right_ankle",    # 16
]

# Keypoint pairs for skeleton visualization
SKELETON_EDGES = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),   # arms
    (5, 11), (6, 12), (11, 12),                  # torso
    (11, 13), (13, 15), (12, 14), (14, 16),      # legs
    (0, 1), (0, 2), (1, 3), (2, 4),             # face
]


class PoseDetector:
    """YOLOv8n-pose detector with batch inference support."""

    def __init__(self, model_path: str = "yolov8n-pose.pt", device: Optional[str] = None):
        """
        Args:
            model_path: Path to YOLOv8n-pose weights.
            device: 'cuda', 'cpu', or None (auto-detect).
        """
        self.model = YOLO(model_path)
        self.device = device or ("cuda" if self._has_cuda() else "cpu")

    @staticmethod
    def _has_cuda() -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def infer_batch(self, frames: list[np.ndarray]) -> list[np.ndarray]:
        """Run pose detection on a batch of frames (one per camera view).

        Args:
            frames: List of BGR images (H×W×3), len=3 for 3-view setup.

        Returns:
            List of (N, 3) arrays, one per view. N is 17 for COCO keypoints.
            Each row is (x, y, confidence). Returns empty (0,3) array if no person detected.
        """
        results = self.model(frames, device=self.device, verbose=False)
        all_keypoints: list[np.ndarray] = []

        for r in results:
            if r.keypoints is None or r.keypoints.xy is None or len(r.keypoints.xy) == 0:
                all_keypoints.append(np.zeros((17, 3), dtype=np.float32))
                continue

            # Select the person closest to image center (handles multiple people)
            kps_all = r.keypoints.xy.cpu().numpy()  # (P, 17, 2)
            confs_all = r.keypoints.conf.cpu().numpy()  # (P, 17)
            img_w = r.orig_shape[1] if hasattr(r, 'orig_shape') else 640
            img_h = r.orig_shape[0] if hasattr(r, 'orig_shape') else 640

            if len(kps_all) == 1:
                best_idx = 0
            else:
                # Pick person closest to image center
                centers = []
                for p in range(len(kps_all)):
                    valid = confs_all[p] > 0.3
                    if valid.any():
                        cx = kps_all[p][valid, 0].mean()
                        cy = kps_all[p][valid, 1].mean()
                        dist = ((cx - img_w/2)**2 + (cy - img_h/2)**2)**0.5
                        centers.append(dist)
                    else:
                        centers.append(1e9)
                best_idx = int(np.argmin(centers))

            kp = kps_all[best_idx]
            conf = confs_all[best_idx]

            kp_with_conf = np.column_stack([kp, conf]).astype(np.float32)
            all_keypoints.append(kp_with_conf)

        return all_keypoints

    def infer_single(self, frame: np.ndarray) -> np.ndarray:
        """Run on single frame, returns (17, 3)."""
        return self.infer_batch([frame])[0]


class MultiViewPoseVisualizer:
    """Draw skeleton overlays on 3-view frames for debugging."""

    @staticmethod
    def draw_keypoints(
        frame: np.ndarray,
        keypoints: np.ndarray,
        conf_threshold: float = 0.3,
    ) -> np.ndarray:
        """Draw COCO 17 skeleton on frame.

        Args:
            frame: BGR image (H×W×3).
            keypoints: (17, 3) — (x, y, confidence).
            conf_threshold: Minimum confidence to draw a keypoint.

        Returns:
            Annotated frame (copy).
        """
        vis = frame.copy()

        for (i, j) in SKELETON_EDGES:
            if keypoints[i, 2] > conf_threshold and keypoints[j, 2] > conf_threshold:
                pt1 = (int(keypoints[i, 0]), int(keypoints[i, 1]))
                pt2 = (int(keypoints[j, 0]), int(keypoints[j, 1]))
                cv2.line(vis, pt1, pt2, (0, 255, 0), 2)

        for i in range(17):
            if keypoints[i, 2] > conf_threshold:
                pt = (int(keypoints[i, 0]), int(keypoints[i, 1]))
                cv2.circle(vis, pt, 3, (0, 0, 255), -1)

        return vis


# --- Quick test ---
if __name__ == "__main__":
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from pipeline.video_reader import MCFDReader

    root = os.path.join(os.path.dirname(__file__), "..", "dataset", "mcfd")
    root = os.path.abspath(root)

    detector = PoseDetector()
    print(f"Device: {detector.device}")

    with MCFDReader(root, scenario=1, camera_ids=[1, 4, 7]) as reader:
        print(f"Processing {reader.remaining} frames...")
        for i, frames in enumerate(reader):
            kps = detector.infer_batch(frames)
            for v, kp in enumerate(kps):
                valid = (kp[:, 2] > 0.3).sum()
                print(f"  Frame {i}, View {v}: {int(valid)}/17 keypoints visible")
            if i >= 9:  # Test first 10 frames
                break

    print("PoseDetector test OK")
