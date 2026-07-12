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

    def __init__(
        self,
        model_path: str = "yolov8n-pose.pt",
        device: Optional[str] = None,
        person_conf_threshold: float = 0.5,
        min_valid_keypoints: int = 5,
        kp_conf_threshold: float = 0.3,
        infer_batch_size: int = 32,
    ):
        """
        Args:
            model_path: Path to YOLOv8n-pose weights.
            device: 'cuda', 'cpu', or None (auto-detect).
            person_conf_threshold: Minimum YOLO person detection confidence (0-1).
                Detections below this are discarded (filters chairs, objects).
            min_valid_keypoints: Minimum number of keypoints with confidence
                > kp_conf_threshold to accept a person detection.
            kp_conf_threshold: Per-keypoint confidence threshold.
            infer_batch_size: YOLO internal batch size. MUST be > 1 for GPU utilization.
        """
        self.model = YOLO(model_path)
        self.device = device or ("cuda" if self._has_cuda() else "cpu")
        self.person_conf_threshold = person_conf_threshold
        self.min_valid_keypoints = min_valid_keypoints
        self.kp_conf_threshold = kp_conf_threshold
        self.infer_batch_size = infer_batch_size

    @staticmethod
    def _has_cuda() -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def _is_valid_person(self, r, p_idx: int, kps: np.ndarray, confs: np.ndarray) -> bool:
        """Check whether detection p_idx is a valid person (not a chair/object).

        Filters using:
          1. YOLO box detection confidence (must be ≥ person_conf_threshold)
          2. Minimum number of high-confidence keypoints (≥ min_valid_keypoints)

        Args:
            r: YOLO Results object for this view.
            p_idx: Person index within the view.
            kps: (17, 2) keypoint coordinates for this person.
            confs: (17,) keypoint confidences for this person.

        Returns:
            True if detection passes all filters.
        """
        # Filter 1: YOLO person detection confidence
        if hasattr(r, 'boxes') and r.boxes is not None:
            try:
                box_conf = float(r.boxes.conf[p_idx].cpu().numpy())
            except (IndexError, AttributeError):
                box_conf = 0.0
            if box_conf < self.person_conf_threshold:
                return False

        # Filter 2: minimum valid keypoints
        n_valid = int((confs > self.kp_conf_threshold).sum())
        if n_valid < self.min_valid_keypoints:
            return False

        return True

    def infer_batch(self, frames: list[np.ndarray]) -> list[np.ndarray]:
        """Run pose detection on a batch of frames (one per camera view).

        Args:
            frames: List of BGR images (H×W×3), len=3 for 3-view setup.

        Returns:
            List of (N, 3) arrays, one per view. N is 17 for COCO keypoints.
            Each row is (x, y, confidence). Returns zero-filled (17,3) array
            if no valid person detected.
        """
        results = self.model(frames, device=self.device, verbose=False,
                            batch=min(self.infer_batch_size, max(1, len(frames))))
        all_keypoints: list[np.ndarray] = []

        for r in results:
            if r.keypoints is None or r.keypoints.xy is None or len(r.keypoints.xy) == 0:
                all_keypoints.append(np.zeros((17, 3), dtype=np.float32))
                continue

            kps_all = r.keypoints.xy.cpu().numpy()  # (P, 17, 2)
            confs_all = r.keypoints.conf.cpu().numpy()  # (P, 17)
            img_w = r.orig_shape[1] if hasattr(r, 'orig_shape') else 640
            img_h = r.orig_shape[0] if hasattr(r, 'orig_shape') else 640

            # Filter valid persons first (exclude chairs/objects)
            valid_indices = []
            for p in range(len(kps_all)):
                if self._is_valid_person(r, p, kps_all[p], confs_all[p]):
                    valid_indices.append(p)

            if not valid_indices:
                all_keypoints.append(np.zeros((17, 3), dtype=np.float32))
                continue

            if len(valid_indices) == 1:
                best_idx = valid_indices[0]
            else:
                # Pick valid person closest to image center
                centers = []
                for p in valid_indices:
                    valid_kp = confs_all[p] > self.kp_conf_threshold
                    if valid_kp.any():
                        cx = kps_all[p][valid_kp, 0].mean()
                        cy = kps_all[p][valid_kp, 1].mean()
                        dist = ((cx - img_w/2)**2 + (cy - img_h/2)**2)**0.5
                        centers.append(dist)
                    else:
                        centers.append(1e9)
                best_idx = valid_indices[int(np.argmin(centers))]

            kp = kps_all[best_idx]
            conf = confs_all[best_idx]
            kp_with_conf = np.column_stack([kp, conf]).astype(np.float32)
            all_keypoints.append(kp_with_conf)

        return all_keypoints

    def infer_batch_all(self, frames: list[np.ndarray]) -> list[list[np.ndarray]]:
        """Run pose detection, return ALL valid persons per view.

        Returns:
            List of lists: [view0_persons, view1_persons, view2_persons]
            Each person is (17, 3) — (x, y, conf). Sorted by distance to center.
            Invalid detections (chairs/objects) are filtered out.
        """
        results = self.model(frames, device=self.device, verbose=False,
                            batch=min(self.infer_batch_size, max(1, len(frames))))
        all_persons: list[list[np.ndarray]] = []

        for r in results:
            if r.keypoints is None or r.keypoints.xy is None or len(r.keypoints.xy) == 0:
                all_persons.append([])
                continue

            kps_all = r.keypoints.xy.cpu().numpy()  # (P, 17, 2)
            confs_all = r.keypoints.conf.cpu().numpy()  # (P, 17)
            img_w = r.orig_shape[1] if hasattr(r, 'orig_shape') else 640
            img_h = r.orig_shape[0] if hasattr(r, 'orig_shape') else 640

            # Filter: only keep valid persons, compute center distance
            valid_persons = []
            for p in range(len(kps_all)):
                if not self._is_valid_person(r, p, kps_all[p], confs_all[p]):
                    continue
                kp = np.column_stack([kps_all[p], confs_all[p]]).astype(np.float32)
                valid_kp = confs_all[p] > self.kp_conf_threshold
                if valid_kp.any():
                    cx, cy = kps_all[p][valid_kp, 0].mean(), kps_all[p][valid_kp, 1].mean()
                    dist = ((cx - img_w/2)**2 + (cy - img_h/2)**2)**0.5
                else:
                    dist = 1e9
                valid_persons.append((kp, dist))

            # Sort closest first
            valid_persons.sort(key=lambda x: x[1])
            all_persons.append([vp[0] for vp in valid_persons])

        return all_persons

    def infer_single(self, frame: np.ndarray) -> np.ndarray:
        """Run on single frame, returns (17, 3)."""
        return self.infer_batch([frame])[0]

    def reset(self):
        """Reset detector state (call when switching scenarios/videos)."""
        pass


class MultiPersonTracker:
    """Simple multi-person tracker using spatial proximity matching.

    Maintains consistent person IDs across frames. Without this, person IDs
    are assigned per-frame (e.g., sorted by center distance), causing ID swaps
    whenever two people cross paths. This tracker matches current detections
    to previous tracks by minimizing Euclidean distance, keeping IDs stable.

    Usage:
        tracker = MultiPersonTracker(num_views=3)
        for frame in video:
            persons = detector.infer_batch_all(frames)  # raw per-frame order
            tracked = tracker.update(persons)            # stable track IDs
            # tracked[v] = [(track_id, keypoints), ...] sorted by track_id
    """

    def __init__(self, num_views: int = 3, max_displacement: float = 80.0,
                 max_lost: int = 15, min_confidence: float = 0.3):
        self.num_views = num_views
        self.max_displacement = max_displacement
        self.max_lost = max_lost
        self.min_confidence = min_confidence
        # Per-view: {track_id: {'center': (2,), 'age': int, 'lost': int}}
        self._tracks: list[dict[int, dict]] = [{} for _ in range(num_views)]
        self._next_ids: list[int] = [0] * num_views

    def update(
        self, persons_per_view: list[list[np.ndarray]]
    ) -> list[list[tuple[int, np.ndarray]]]:
        """Match detections to tracks, return (track_id, keypoints) per view.

        Only tracks with a current detection are returned. Track IDs are
        stable across frames for the same physical person.

        Args:
            persons_per_view: [view0_persons, ...] from infer_batch_all().
        Returns:
            Per-view lists of (track_id, keypoints) sorted by track_id.
        """
        result: list[list[tuple[int, np.ndarray]]] = []
        for v in range(self.num_views):
            persons = persons_per_view[v] if v < len(persons_per_view) else []
            result.append(self._update_view(v, persons))
        return result

    def _update_view(
        self, view_idx: int, persons: list[np.ndarray]
    ) -> list[tuple[int, np.ndarray]]:
        tracks = self._tracks[view_idx]

        if not persons:
            for tid in list(tracks):
                tracks[tid]['lost'] += 1
                if tracks[tid]['lost'] > self.max_lost:
                    del tracks[tid]
            return []

        # Centers of current detections
        curr_centers = np.array([self._center(kp) for kp in persons])

        # First frame: create tracks for all
        if not tracks:
            result = []
            for kp in persons:
                tid = self._next_ids[view_idx]
                self._next_ids[view_idx] += 1
                tracks[tid] = {'center': self._center(kp), 'age': 1, 'lost': 0}
                result.append((tid, kp))
            result.sort(key=lambda x: x[0])
            return result

        # Build cost matrix
        track_ids = sorted(tracks)
        track_centers = np.array([tracks[tid]['center'] for tid in track_ids])
        n_curr, n_track = len(curr_centers), len(track_centers)

        cost = np.full((n_curr, n_track), 1e9)
        for i in range(n_curr):
            for j in range(n_track):
                d = np.linalg.norm(curr_centers[i] - track_centers[j])
                if d < self.max_displacement:
                    cost[i, j] = d

        # Greedy matching (sufficient for 1–5 people)
        assignments: dict[int, int] = {}  # curr_idx -> track_id
        used_j: set[int] = set()
        for i in range(n_curr):
            j_best = int(np.argmin(cost[i]))
            if cost[i, j_best] < 1e8 and j_best not in used_j:
                assignments[i] = track_ids[j_best]
                used_j.add(j_best)

        # Age unmatched tracks
        matched_tids = set(assignments.values())
        for tid in track_ids:
            if tid not in matched_tids:
                tracks[tid]['lost'] += 1

        # Update matched / create new
        for i in range(n_curr):
            if i in assignments:
                tid = assignments[i]
                tracks[tid]['center'] = curr_centers[i].copy()
                tracks[tid]['lost'] = 0
                tracks[tid]['age'] += 1
            else:
                tid = self._next_ids[view_idx]
                self._next_ids[view_idx] += 1
                tracks[tid] = {'center': curr_centers[i].copy(), 'age': 1, 'lost': 0}
                assignments[i] = tid  # track newly created

        # Remove expired tracks
        for tid in list(tracks):
            if tracks[tid]['lost'] > self.max_lost:
                del tracks[tid]

        # Build result: matched pairs sorted by track_id
        result = [(assignments[i], persons[i]) for i in range(n_curr)]
        result.sort(key=lambda x: x[0])
        return result

    def reset(self):
        """Reset all trackers (call when switching scenarios)."""
        self._tracks = [{} for _ in range(self.num_views)]
        self._next_ids = [0] * self.num_views

    def _center(self, kp: np.ndarray) -> np.ndarray:
        valid = kp[:, 2] > self.min_confidence
        if valid.any():
            return np.array([kp[valid, 0].mean(), kp[valid, 1].mean()])
        return np.array([320.0, 240.0])


class MultiViewPoseVisualizer:
    """Draw skeleton overlays on 3-view frames for debugging."""

    @staticmethod
    def draw_keypoints(
        frame: np.ndarray,
        keypoints: np.ndarray,
        conf_threshold: float = 0.3,
        color: tuple[int, int, int] = (0, 255, 0),
    ) -> np.ndarray:
        """Draw COCO 17 skeleton on frame.

        Args:
            frame: BGR image (H×W×3).
            keypoints: (17, 3) — (x, y, confidence).
            conf_threshold: Minimum confidence to draw a keypoint.
            color: Line color for skeleton edges (BGR).

        Returns:
            Annotated frame (copy).
        """
        vis = frame.copy()

        for (i, j) in SKELETON_EDGES:
            if keypoints[i, 2] > conf_threshold and keypoints[j, 2] > conf_threshold:
                pt1 = (int(keypoints[i, 0]), int(keypoints[i, 1]))
                pt2 = (int(keypoints[j, 0]), int(keypoints[j, 1]))
                cv2.line(vis, pt1, pt2, color, 2)

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
