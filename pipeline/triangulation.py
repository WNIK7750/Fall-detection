"""
Multi-view triangulation: 2D keypoints from 3 cameras → 3D skeleton.

Uses DLT (Direct Linear Transform) via OpenCV's triangulatePoints.
Optional re-projection refinement is available for visualization, but is off
by default because re-projected points are not independent observations.

Usage:
    tri = Triangulator(projection_matrices=[P1, P4, P7])
    skeleton_3d = tri.triangulate(all_keypoints_2d)
    # skeleton_3d: (17, 3) — x, y, z in world coordinates
"""

import numpy as np
import cv2


class Triangulator:
    """3-view DLT triangulation."""

    def __init__(
        self,
        projection_matrices: list[np.ndarray],
        conf_threshold: float = 0.3,
        reproj_threshold: float = 0.3,
    ):
        """
        Args:
            projection_matrices: List of 3 (3x4) projection matrices [P1, P2, P3].
            conf_threshold: Min keypoint confidence to use a view for triangulation.
            reproj_threshold: Min confidence for a view to participate in re-projection.
        """
        assert len(projection_matrices) == 3, "Need exactly 3 projection matrices"
        self.P = projection_matrices
        self.conf_threshold = conf_threshold
        self.reproj_threshold = reproj_threshold

    def triangulate(
        self,
        keypoints_2d: list[np.ndarray],
        apply_reprojection: bool = False,
    ) -> np.ndarray:
        """Triangulate 17 keypoints from 3 views into 3D coordinates.

        Args:
            keypoints_2d: List of 3 arrays, each (17, 3) — (x, y, conf).
            apply_reprojection: If True, run a visualization-oriented
                re-projection loop. Keep this False for training/evaluation.

        Returns:
            (17, 3) array of 3D coordinates. Unreliable points are NaN.
        """
        skeleton_3d = np.full((17, 3), np.nan, dtype=np.float64)

        for j in range(17):
            skeleton_3d[j] = self._triangulate_one_point(keypoints_2d, j)

        if apply_reprojection:
            skeleton_3d = self._reprojection_refine(keypoints_2d, skeleton_3d)

        return skeleton_3d

    def compute_quality(
        self,
        keypoints_2d: list[np.ndarray],
        skeleton_3d: np.ndarray,
    ) -> dict:
        """Estimate measurement quality for a triangulated skeleton.

        This is a confidence heuristic for demo gating and visual QA. It is not
        a ground-truth probability.
        """
        from pipeline.calibration import project_3d_to_2d

        if len(keypoints_2d) != len(self.P):
            raise ValueError(f"Expected {len(self.P)} views, got {len(keypoints_2d)}")

        conf_stack = np.stack([kp[:, 2] for kp in keypoints_2d], axis=0)
        view_counts = (conf_stack > self.conf_threshold).sum(axis=0)
        valid_3d = ~np.isnan(skeleton_3d).any(axis=1)

        valid_joint_ratio = float(valid_3d.mean())
        multiview_joint_ratio = float((view_counts >= 2).mean())
        mean_views_per_joint = float(view_counts.mean())

        observed = conf_stack[conf_stack > self.conf_threshold]
        mean_conf = float(observed.mean()) if observed.size else 0.0

        core_indices = [0, 5, 6, 11, 12]
        core_valid_ratio = float(valid_3d[core_indices].mean())

        reproj_errors = []
        for j in range(17):
            if not valid_3d[j]:
                continue
            pt_3d = skeleton_3d[j]
            for v, kp in enumerate(keypoints_2d):
                if kp[j, 2] <= self.conf_threshold:
                    continue
                uv = project_3d_to_2d(pt_3d, self.P[v])
                err = float(np.linalg.norm(uv - kp[j, :2]))
                if np.isfinite(err):
                    reproj_errors.append(err)

        reproj_error_px = float(np.mean(reproj_errors)) if reproj_errors else float("nan")
        if np.isnan(reproj_error_px):
            reproj_score = 0.0
        else:
            # 20 px is good, 80+ px is poor for 640x640 YOLO-space frames.
            reproj_score = float(np.clip(1.0 - reproj_error_px / 80.0, 0.0, 1.0))

        score = (
            0.30 * valid_joint_ratio +
            0.25 * multiview_joint_ratio +
            0.20 * core_valid_ratio +
            0.15 * mean_conf +
            0.10 * reproj_score
        )

        return {
            "score": float(np.clip(score, 0.0, 1.0)),
            "valid_joint_ratio": valid_joint_ratio,
            "multiview_joint_ratio": multiview_joint_ratio,
            "mean_views_per_joint": mean_views_per_joint,
            "mean_conf": mean_conf,
            "core_valid_ratio": core_valid_ratio,
            "reproj_error_px": reproj_error_px,
        }

    def _triangulate_one_point(
        self,
        keypoints_2d: list[np.ndarray],
        joint_idx: int,
    ) -> np.ndarray:
        """Triangulate a single joint from 2-3 valid views.

        Returns (3,) or nan-filled array if triangulation fails.
        """
        # Collect valid views (confidence > threshold)
        valid_pts = []
        valid_Ps = []
        for v in range(3):
            kp = keypoints_2d[v]
            if kp[joint_idx, 2] > self.conf_threshold:
                valid_pts.append(kp[joint_idx, :2])
                valid_Ps.append(self.P[v])

        return self._triangulate_from_pairs(valid_pts, valid_Ps)

    def _reprojection_refine(
        self,
        keypoints_2d: list[np.ndarray],
        skeleton_3d: np.ndarray,
    ) -> np.ndarray:
        """Occlusion-aware re-projection refinement.

        For low-confidence views: project 3D point back to 2D,
        replace the low-quality observation, re-triangulate.
        """
        from pipeline.calibration import project_3d_to_2d

        refined = skeleton_3d.copy()

        for j in range(17):
            if np.any(np.isnan(refined[j])):
                continue

            pt_3d = refined[j]

            # Build fixed 2D observations for this joint
            fixed_pts = []
            fixed_Ps = []
            for v in range(3):
                if keypoints_2d[v][j, 2] >= self.reproj_threshold:
                    # Use original observation
                    fixed_pts.append(keypoints_2d[v][j, :2])
                    fixed_Ps.append(self.P[v])
                else:
                    # Use re-projected position
                    uv = project_3d_to_2d(pt_3d, self.P[v])
                    fixed_pts.append(np.array(uv))
                    fixed_Ps.append(self.P[v])

            # Re-triangulate using the helper
            refined[j] = self._triangulate_from_pairs(fixed_pts, fixed_Ps)

        return refined

    def _triangulate_from_pairs(
        self,
        pts_2d: list[np.ndarray],
        P_list: list[np.ndarray],
    ) -> np.ndarray:
        """Triangulate from 2-3 paired (2D point, projection matrix)."""
        if len(pts_2d) < 2:
            return np.full(3, np.nan)

        try:
            pts_3d_list = []
            pairs = [(0, 1)]
            if len(pts_2d) >= 3:
                pairs.extend([(0, 2), (1, 2)])

            for a, b in pairs:
                pt1 = pts_2d[a].reshape(2, 1).astype(np.float64)
                pt2 = pts_2d[b].reshape(2, 1).astype(np.float64)
                pts_4d = cv2.triangulatePoints(P_list[a], P_list[b], pt1, pt2)
                if abs(pts_4d[3, 0]) > 1e-10:
                    pts_3d_list.append(pts_4d[:3, 0] / pts_4d[3, 0])

            if not pts_3d_list:
                return np.full(3, np.nan)
            return np.mean(pts_3d_list, axis=0)
        except cv2.error:
            return np.full(3, np.nan)


class KalmanSmoother:
    """Exponential moving average smoother with NaN-aware state persistence."""

    def __init__(self, n_joints: int = 17, alpha: float = 0.3):
        """
        Args:
            n_joints: Number of skeleton joints (17 for COCO).
            alpha: Smoothing factor (0=keep previous, 1=use raw).
        """
        self.n_joints = n_joints
        self.alpha = alpha
        self._state = np.zeros((n_joints, 3), dtype=np.float64)
        self._initialized = [False] * n_joints

    def update(self, skeleton_3d: np.ndarray) -> np.ndarray:
        """Smooth skeleton, filling NaN joints with previous state.

        If a joint is NaN in the current frame, return the last known state.
        If a joint is valid, blend with exponential moving average.
        """
        smoothed = skeleton_3d.copy()

        for j in range(self.n_joints):
            if np.any(np.isnan(skeleton_3d[j])):
                # Joint missing — use last known state
                if self._initialized[j]:
                    smoothed[j] = self._state[j].copy()
            else:
                # Joint valid — blend with history
                if self._initialized[j]:
                    smoothed[j] = (self.alpha * skeleton_3d[j] +
                                   (1.0 - self.alpha) * self._state[j])
                self._state[j] = smoothed[j].copy()
                self._initialized[j] = True

        return smoothed


# --- Quick test ---
if __name__ == "__main__":
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

    from pipeline.calibration import DEFAULT_PROJECTIONS, DEFAULT_CAMERAS, undistort_keypoints
    from pipeline.video_reader import MCFDReader
    from pipeline.pose_inference import PoseDetector

    root = os.path.join(os.path.dirname(__file__), "..", "dataset", "mcfd")
    root = os.path.abspath(root)

    tri = Triangulator(DEFAULT_PROJECTIONS)
    smoother = KalmanSmoother()
    detector = PoseDetector()

    print("Testing triangulation on 10 frames...")
    with MCFDReader(root, scenario=1, camera_ids=DEFAULT_CAMERAS, undistort=False) as reader:
        for i, frames in enumerate(reader):
            if i >= 10:
                break
            kps_2d = detector.infer_batch(frames)
            kps_2d = [
                undistort_keypoints(kp, cam_id)
                for kp, cam_id in zip(kps_2d, DEFAULT_CAMERAS)
            ]
            skel_3d = tri.triangulate(kps_2d)
            skel_3d = smoother.update(skel_3d)

            valid_joints = np.sum(~np.isnan(skel_3d[:, 0]))
            if i == 0 or i == 5 or i == 9:
                print(f"  Frame {i}: {valid_joints}/17 joints reconstructed")
                if valid_joints > 0:
                    valid = skel_3d[~np.isnan(skel_3d[:, 0])]
                    print(f"    X: [{valid[:,0].min():.2f}, {valid[:,0].max():.2f}]")
                    print(f"    Y: [{valid[:,1].min():.2f}, {valid[:,1].max():.2f}]")
                    print(f"    Z: [{valid[:,2].min():.2f}, {valid[:,2].max():.2f}]")

    print("\nTriangulation test OK")
