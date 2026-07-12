"""
3D Feature Engineering — compute 10 biomechanical features from 3D skeleton.

Adapted from the existing 2D feature set (extract_features_v3.py) to 3D space.

IMPORTANT: MCFD calibration uses Z-up world coordinates:
  X = right, Y = forward/depth, Z = up (vertical).
  Gravity direction = -Z. Height is measured along Z.

Feature list (10-D, matching original 2D design):
  F1 : 宽高比 (Aspect Ratio)         — 3D bounding box width/height ratio
  F2 : 宽高比变化率 (ΔAR)            — frame-to-frame change of F1
  F3 : 躯干倾角 (Torso Angle)        — spine vector vs vertical (Z-axis)
  F4 : 躯干倾角变化率 (ΔTorso)       — raw value, delta in window extraction
  F5 : 质心高度 (COM Height)         — normalized Z of center-of-mass
  F6 : 质心垂直速度 (COM Velocity)   — Z-axis velocity of COM (per-step)
  F7 : 速度比 (Velocity Ratio)       — |Vz| / |Vxy| (vertical vs horizontal)
  F8 : 头脚距离 (Head-Foot Distance) — nose-to-ankle 3D euclidean distance
  F9 : 头部高度 (Head Height)        — normalized Z of nose
  F10: 脊柱方向 (Spine Direction)    — Z component of spine unit vector

Each feature is computed per-frame, then accumulated over a 20-frame
sliding window (stride 5) to feed the BiLSTM+BiGRU classifier.
"""

import numpy as np

# COCO keypoint indices
NOSE = 0
L_SHOULDER, R_SHOULDER = 5, 6
L_HIP, R_HIP = 11, 12
L_KNEE, R_KNEE = 13, 14
L_ANKLE, R_ANKLE = 15, 16

# Z-up world (MCFD calibration convention)
# Components: (X, Y, Z) where Z = height
AXIS_H = 2          # height axis index
STANDING_HEIGHT = 1.7  # meters, approximate standing person height


class FloorEstimator:
    """Estimate floor plane Z-level from observed ankle positions.

    Accumulates ankle Z values over time; floor = minimum Z (lowest point).
    This makes height features relative to the actual floor, not the
    arbitrary world origin from calibration.
    """

    def __init__(self, warmup_frames: int = 30):
        self._z_values: list[float] = []
        self._floor_z: float | None = None
        self._warmup = warmup_frames

    def update(self, skeleton_3d: np.ndarray):
        """Feed a new skeleton to refine floor estimate."""
        if self._floor_z is not None:
            return  # locked after warmup
        for j in (L_ANKLE, R_ANKLE):
            if not np.isnan(skeleton_3d[j]).any():
                self._z_values.append(float(skeleton_3d[j, AXIS_H]))
        if len(self._z_values) >= self._warmup:
            self._floor_z = float(np.percentile(self._z_values, 5))  # 5th percentile

    @property
    def floor_z(self) -> float:
        return self._floor_z if self._floor_z is not None else 0.0

    @property
    def ready(self) -> bool:
        return self._floor_z is not None




# ---------------------------------------------------------------------------
# Per-frame feature computation
# ---------------------------------------------------------------------------

def compute_features(
    skeleton_3d: np.ndarray,
    prev_skeleton: np.ndarray | None = None,
    floor_estimator: FloorEstimator | None = None,
) -> np.ndarray | None:
    """Compute 10-D feature vector from a single 3D skeleton.

    Args:
        skeleton_3d: (17, 3) world coordinates, may contain NaN for missing joints.
        prev_skeleton: (17, 3) previous frame for velocity/delta features.
        floor_estimator: FloorEstimator instance. If None, a temporary one is
            created (caller should pass a shared instance to avoid state pollution).

    Returns:
        (10,) feature vector, or None if too many joints missing.
    """
    # Use caller-provided estimator, or create a temporary one
    if floor_estimator is None:
        floor_estimator = FloorEstimator()
    # Check if we have minimum required joints
    required = [L_SHOULDER, R_SHOULDER, L_HIP, R_HIP, NOSE]
    if any(np.isnan(skeleton_3d[j]).any() for j in required):
        return None

    # Derived points
    mid_shoulder = (skeleton_3d[L_SHOULDER] + skeleton_3d[R_SHOULDER]) / 2.0
    mid_hip = (skeleton_3d[L_HIP] + skeleton_3d[R_HIP]) / 2.0
    mid_ankle = np.full(3, np.nan)
    if not np.isnan(skeleton_3d[L_ANKLE]).any() and not np.isnan(skeleton_3d[R_ANKLE]).any():
        mid_ankle = (skeleton_3d[L_ANKLE] + skeleton_3d[R_ANKLE]) / 2.0

    com = (mid_shoulder + mid_hip) / 2.0
    spine_vec = mid_shoulder - mid_hip
    spine_len = np.linalg.norm(spine_vec)

    head_pt = skeleton_3d[NOSE].copy()
    # Fallback: use mid-shoulder if nose is NaN (occluded/undetected)
    if np.isnan(head_pt).any():
        head_pt = mid_shoulder.copy()

    # --- F1: 3D 宽高比 (width/height in XZ plane) ---
    valid = skeleton_3d[~np.isnan(skeleton_3d).any(axis=1)]
    if len(valid) < 5:
        return None
    bbox_w = valid[:, 0].max() - valid[:, 0].min()   # X span
    bbox_h = valid[:, 2].max() - valid[:, 2].min()   # Z span (height)
    f1 = bbox_w / (bbox_h + 1e-8)
    f1 = np.clip(f1, 0.0, 5.0)

    # --- F2: 宽高比变化率 ---
    f2 = 0.0
    if prev_skeleton is not None:
        prev_valid = prev_skeleton[~np.isnan(prev_skeleton).any(axis=1)]
        if len(prev_valid) >= 5:
            prev_h = prev_valid[:, 2].max() - prev_valid[:, 2].min()
            prev_w = prev_valid[:, 0].max() - prev_valid[:, 0].min()
            f2 = f1 - np.clip(prev_w / (prev_h + 1e-8), 0, 5)
            f2 = np.clip(f2, -1.0, 1.0)

    # --- F3: 躯干倾角 (spine vs vertical) ---
    if spine_len > 1e-6:
        spine_unit = spine_vec / spine_len
        # cos between spine and +Z (upright direction)
        cos_theta = np.dot(spine_unit, np.array([0, 0, 1]))
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        f3 = np.degrees(np.arccos(cos_theta))
    else:
        f3 = 0.0

    # --- F4: 躯干倾角变化率 (placeholder — delta computed in extract_window) ---
    f4 = f3

    # --- F5: 质心高度 (relative to estimated floor, normalized) ---
    floor_estimator.update(skeleton_3d)
    floor_z = floor_estimator.floor_z
    f5 = (com[AXIS_H] - floor_z) / STANDING_HEIGHT
    f5 = np.clip(f5, -0.5, 3.0)

    # --- F6: 质心垂直速度 (Z-axis delta per step) ---
    f6 = 0.0
    if prev_skeleton is not None:
        prev_com = _safe_com(prev_skeleton)
        if prev_com is not None:
            f6 = (com[AXIS_H] - prev_com[AXIS_H])
            f6 = np.clip(f6, -2.0, 2.0)

    # --- F7: 速度比 |Vz| / |Vxy| ---
    f7 = 0.0
    if prev_skeleton is not None:
        prev_com = _safe_com(prev_skeleton)
        if prev_com is not None:
            v = com - prev_com
            vz = abs(v[AXIS_H])
            vxy = np.sqrt(v[0]**2 + v[1]**2)
            f7 = vz / (vxy + 1e-3)
            f7 = np.clip(f7, 0.0, 10.0)

    # --- F8: 头脚距离 (nose to ankle midpoint euclidean) ---
    if not np.any(np.isnan(mid_ankle)):
        f8 = np.linalg.norm(head_pt - mid_ankle) / STANDING_HEIGHT
        f8 = np.clip(f8, 0.0, 2.0)
    else:
        f8 = 1.0

    # --- F9: 头部高度 (relative to floor) ---
    f9 = (head_pt[AXIS_H] - floor_z) / STANDING_HEIGHT
    f9 = np.clip(f9, -0.5, 3.0)

    # --- F10: 脊柱方向Z分量 (1=upright, 0=horizontal, -1=inverted) ---
    if spine_len > 1e-6:
        f10 = spine_unit[AXIS_H]
    else:
        f10 = 1.0

    return np.array([f1, f2, f3, f4, f5, f6, f7, f8, f9, f10], dtype=np.float32)


def _safe_com(skel: np.ndarray) -> np.ndarray | None:
    """Safe center-of-mass from shoulder+hip midpoints."""
    req = [L_SHOULDER, R_SHOULDER, L_HIP, R_HIP]
    if any(np.isnan(skel[j]).any() for j in req):
        return None
    ms = (skel[L_SHOULDER] + skel[R_SHOULDER]) / 2.0
    mh = (skel[L_HIP] + skel[R_HIP]) / 2.0
    return (ms + mh) / 2.0


# ---------------------------------------------------------------------------
# Sliding window extraction
# ---------------------------------------------------------------------------

def extract_window(
    skeletons: list[np.ndarray],
    window_size: int = 20,
    stride: int = 5,
    subsample: int = 3,
) -> tuple[np.ndarray, list[int]]:
    """Extract sliding windows of features from a skeleton sequence.

    Args:
        skeletons: List of (17, 3) arrays, one per frame.
        window_size: Number of frames per window.
        stride: Step between window starts.
        subsample: Take every Nth frame to reduce redundancy at high FPS.

    Returns:
        X: (N, 10, window_size) — N windows, 10 features, window_size timesteps.
        frame_indices: Start frame index for each window.
    """
    # Create a fresh FloorEstimator per sequence (no global state)
    floor_estimator = FloorEstimator()

    # Subsample
    skeletons = skeletons[::subsample]

    # Compute per-frame features
    features: list[np.ndarray] = []
    for i, skel in enumerate(skeletons):
        prev = skeletons[i - 1] if i > 0 else None
        f = compute_features(skel, prev, floor_estimator=floor_estimator)
        if f is not None:
            features.append(f)

    # F4: compute delta of torso angle (F3) — frame-to-frame change rate
    # This was previously a dead placeholder (f4=f3); now contains real signal.
    for i in range(len(features)):
        if i > 0:
            features[i][3] = features[i][2] - features[i - 1][2]
        else:
            features[i][3] = 0.0

    if len(features) < window_size:
        return np.empty((0, 10, window_size), dtype=np.float32), []

    # Warn if floor never reached warmup (height features use default 0.0)
    if not floor_estimator.ready:
        print("WARNING: floor estimator never reached warmup "
              f"({len(features)} frames) — height features use default floor_z=0.0")

    # Sliding window
    windows = []
    indices = []
    for start in range(0, len(features) - window_size + 1, stride):
        window = np.array(features[start:start + window_size]).T  # (10, window_size)
        windows.append(window)
        indices.append(start * subsample)  # map back to original frame index

    return np.array(windows, dtype=np.float32), indices


def compute_delta_feature(features: list[np.ndarray], idx: int, window: int = 3) -> float:
    """Compute delta (rate of change) of feature at index over `window` frames."""
    if len(features) < window:
        return 0.0
    recent = features[-window:]
    vals = [f[idx] for f in recent if f is not None]
    if len(vals) < 2:
        return 0.0
    return float(vals[-1] - vals[0])


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

    from pipeline.calibration import DEFAULT_PROJECTIONS
    from pipeline.video_reader import MCFDReader
    from pipeline.pose_inference import PoseDetector
    from pipeline.triangulation import Triangulator, KalmanSmoother

    root = os.path.join(os.path.dirname(__file__), "..", "dataset", "mcfd")
    root = os.path.abspath(root)

    detector = PoseDetector()
    tri = Triangulator(DEFAULT_PROJECTIONS)
    smoother = KalmanSmoother()

    # Process 200 frames of scenario 1
    all_skeletons = []
    with MCFDReader(root, scenario=1, camera_ids=[1, 4, 7]) as reader:
        for i, frames in enumerate(reader):
            if i >= 200:
                break
            kps = detector.infer_batch(frames)
            skel = tri.triangulate(kps)
            skel = smoother.update(skel)
            all_skeletons.append(skel)

    # Extract windows
    X, indices = extract_window(all_skeletons)
    print(f"Extracted {len(X)} windows of shape {X.shape}")
    if len(X) > 0:
        print(f"Feature summary (first window):")
        print(f"  F1 (AR):      mean={X[0,0].mean():.3f}, std={X[0,0].std():.3f}")
        print(f"  F3 (Torso):   mean={X[0,2].mean():.1f}°, std={X[0,2].std():.1f}°")
        print(f"  F5 (COM H):   mean={X[0,4].mean():.3f}, std={X[0,4].std():.3f}")
        print(f"  F6 (COM Vy):  mean={X[0,5].mean():.3f}, std={X[0,5].std():.3f}")
        print(f"  F8 (H-F Dist): mean={X[0,7].mean():.3f}, std={X[0,7].std():.3f}")
        print(f"  F9 (Head H):  mean={X[0,8].mean():.3f}, std={X[0,8].std():.3f}")

    print("\nFeatures 3D test OK")
