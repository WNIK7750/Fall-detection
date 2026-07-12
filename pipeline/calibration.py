"""
Camera calibration for MCFD (Multiple Cameras Fall Dataset).

Parameters sourced from B-MCFD metadata release:
  https://github.com/xiaoran007/B-MCFD
  data/metadata/camera_calibration_report.json

Original technical report:
  Auvinet et al. "Multiple cameras fall data set", DIRO-UdeM TR 1350, 2010.

Calibration model: pinhole camera + radial/tangential distortion.
  Distortion: [k1, k2, p1, p2, k3]  (OpenCV convention)
  Extrinsics: X_cam = R * X_world + t

Image size: 720×480 (original) → 640×640 (YOLO input after resize+pad)
Intrinsics are provided for BOTH coordinate spaces.
"""

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Intrinsic parameters (720×480 original space)
# ---------------------------------------------------------------------------
# fmt: off
K_ORIG = {
    # Camera: fx, fy, u0, v0, alphac
    1: (471.3266, 426.6622, 346.6579, 218.3593, 0.0),
    2: (472.9590, 428.8467, 345.4698, 248.4972, 0.0),
    3: (467.1343, 422.9042, 332.8701, 273.1742, 0.0),
    4: (467.4011, 422.9211, 370.5110, 235.6763, 0.0),
    5: (468.5257, 424.2782, 356.8671, 260.5547, 0.0),
    6: (464.8039, 421.0881, 363.0517, 263.9025, 0.0),
    7: (471.6728, 426.9852, 373.0026, 216.5979, 0.0),
    8: (470.5613, 425.5946, 355.1725, 231.0570, 0.0),
}

DISTORTION = {
    1: (-0.39782, 0.13123, -0.00115, -0.00229, 0.0),
    2: (-0.38907, 0.12599,  0.00106, -0.00041, 0.0),
    3: (-0.41684, 0.15997,  0.00081, -0.00156, 0.0),
    4: (-0.40789, 0.15098, -0.00057,  0.00048, 0.0),
    5: (-0.41325, 0.15696, -0.00047,  0.00010, 0.0),
    6: (-0.40543, 0.14161, -0.00092, -0.00054, 0.0),
    7: (-0.39774, 0.13729, -0.00011,  0.00139, 0.0),
    8: (-0.41275, 0.14816, -0.00275, -0.00043, 0.0),
}

# Extrinsic parameters — X_cam = R @ X_world + t
ROTATION = {
    1: [[ 0.70023,  0.69394,  0.16771],
        [ 0.51742, -0.33143, -0.78894],
        [-0.49190,  0.63922, -0.59113]],
    2: [[ 0.98107,  0.18649,  0.05215],
        [ 0.15434, -0.59039, -0.79223],
        [-0.11695,  0.78528, -0.60799]],
    3: [[ 0.83628, -0.54265, -0.07855],
        [-0.42735, -0.55531, -0.71344],
        [ 0.34353,  0.63020, -0.69630]],
    4: [[-0.04619, -0.99738, -0.05567],
        [-0.55096,  0.07193, -0.83143],
        [ 0.83326, -0.00773, -0.55283]],
    5: [[-0.08292,  0.99656, -0.00153],
        [ 0.46107,  0.03701, -0.88659],
        [-0.88348, -0.07422, -0.46255]],
    6: [[-0.72478, -0.68283, -0.09183],
        [-0.42480,  0.54783, -0.78894],
        [ 0.54243, -0.48335, -0.68712]],
    7: [[-0.99768, -0.03604,  0.05768],
        [-0.06754,  0.62511, -0.77761],
        [-0.00803, -0.77971, -0.62609]],
    8: [[-0.76062,  0.64570,  0.06730],
        [ 0.39302,  0.54050, -0.74391],
        [-0.51672, -0.53938, -0.66489]],
}

TRANSLATION = {
    1: [-3.4765, -1.1561, 4.8455],
    2: [-2.8208,  0.9211, 2.7347],
    3: [-0.8689,  1.4263, 2.1678],
    4: [ 0.0141,  0.0340, 4.5147],
    5: [-0.8374, -0.2553, 5.8297],
    6: [ 2.9829,  0.1372, 3.1253],
    7: [ 2.8546,  0.0434, 4.3956],
    8: [ 1.7360, -2.1325, 6.3469],
}
# fmt: on

# ---------------------------------------------------------------------------
# Adjusted intrinsics for 640×640 YOLO input space
# ---------------------------------------------------------------------------
# video_reader.py: scale = 640/max(w,h), pad top/bottom to 640 square
# For 720×480: scale = 640/720 = 0.8889, pad_top = (640 - 480*scale)/2

ORIG_W, ORIG_H = 720, 480
YOLO_SIZE = 640
SCALE = YOLO_SIZE / max(ORIG_W, ORIG_H)  # 640/720 ≈ 0.8889
PAD_TOP = (YOLO_SIZE - ORIG_H * SCALE) / 2  # ≈ 106.7

def build_K_matrix_original(cam_id: int) -> np.ndarray:
    """K matrix in original 720×480 space."""
    fx, fy, u0, v0, _ = K_ORIG[cam_id]
    return np.array([
        [fx, 0,  u0],
        [0,  fy, v0],
        [0,  0,  1 ],
    ], dtype=np.float64)

def build_K_matrix_yolo(cam_id: int) -> np.ndarray:
    """K matrix adjusted for 640×640 YOLO input space (resize+pad)."""
    fx, fy, u0, v0, _ = K_ORIG[cam_id]
    return np.array([
        [fx * SCALE, 0,          u0 * SCALE],
        [0,          fy * SCALE, v0 * SCALE + PAD_TOP],
        [0,          0,          1],
    ], dtype=np.float64)

def build_R_matrix(cam_id: int) -> np.ndarray:
    return np.array(ROTATION[cam_id], dtype=np.float64)

def build_t_vector(cam_id: int) -> np.ndarray:
    return np.array(TRANSLATION[cam_id], dtype=np.float64)

def build_projection_matrix(cam_id: int, use_yolo_space: bool = True) -> np.ndarray:
    """Build P = K * [R | t] (3×4) projection matrix.

    Args:
        cam_id: Camera number 1-8.
        use_yolo_space: If True, use K adjusted for 640×640 YOLO input.
                        If False, use original 720×480 K.
    """
    K_mat = build_K_matrix_yolo(cam_id) if use_yolo_space else build_K_matrix_original(cam_id)
    R_mat = build_R_matrix(cam_id)
    t_vec = build_t_vector(cam_id)
    Rt = np.hstack([R_mat, t_vec.reshape(3, 1)])
    return K_mat @ Rt


# Default 3-camera setup (cam 1, 4, 7 — well-spaced around the room)
DEFAULT_CAMERAS = [1, 4, 7]
DEFAULT_PROJECTIONS = [build_projection_matrix(c) for c in DEFAULT_CAMERAS]

PROJECTION_MATRICES = {c: build_projection_matrix(c) for c in range(1, 9)}


# ---------------------------------------------------------------------------
# Undistortion — precompute remap maps for each camera
# ---------------------------------------------------------------------------
# MCFD cameras have significant radial distortion (k1 ≈ -0.4).
# Undistorting is essential for correct 3D triangulation because the
# pinhole K matrix assumes rectilinear projection.
#
# Maps are computed at original 720×480 resolution, then applied before
# resize+pad to YOLO 640×640 space.

_UNDISTORT_MAP1: dict[int, np.ndarray] = {}
_UNDISTORT_MAP2: dict[int, np.ndarray] = {}

def _ensure_undistort_maps():
    """Lazily compute undistort maps for all 8 cameras."""
    if _UNDISTORT_MAP1:
        return
    for cam_id in range(1, 9):
        K = build_K_matrix_original(cam_id)
        dist = np.array(DISTORTION[cam_id], dtype=np.float64)
        m1, m2 = cv2.initUndistortRectifyMap(
            K, dist, None, K, (ORIG_W, ORIG_H), cv2.CV_16SC2,
        )
        _UNDISTORT_MAP1[cam_id] = m1
        _UNDISTORT_MAP2[cam_id] = m2

def get_undistort_maps(camera_ids: list[int]) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Return (map1_list, map2_list) for the given cameras.

    Each map pair is suitable for cv2.remap(frame, m1, m2, cv2.INTER_LINEAR).
    """
    _ensure_undistort_maps()
    return (
        [_UNDISTORT_MAP1[cid] for cid in camera_ids],
        [_UNDISTORT_MAP2[cid] for cid in camera_ids],
    )

def undistort_keypoints(
    keypoints: np.ndarray,
    cam_id: int,
) -> np.ndarray:
    """Undistort 2D keypoint coordinates (alternative to full-frame undistortion).

    Use this when you want to keep frames distorted but correct keypoints
    for triangulation. Converts from original 720×480 distorted space to
    640×640 YOLO undistorted space.

    Args:
        keypoints: (17, 2) or (17, 3) — (x, y[, conf]) in YOLO 640×640 space.
        cam_id: Camera ID (1-8).

    Returns:
        Undistorted keypoints in YOLO 640×640 space, same shape as input.
    """
    _ensure_undistort_maps()
    K_orig = build_K_matrix_original(cam_id)
    K_yolo = build_K_matrix_yolo(cam_id)
    dist = np.array(DISTORTION[cam_id], dtype=np.float64)

    # Map from YOLO space → original 720×480 space
    inv_scale = 1.0 / SCALE
    pts_orig = keypoints[:, :2].copy().astype(np.float64)
    pts_orig[:, 0] = pts_orig[:, 0] * inv_scale
    pts_orig[:, 1] = (pts_orig[:, 1] - PAD_TOP) * inv_scale

    # Undistort at original resolution
    pts_undist = cv2.undistortPoints(
        pts_orig.reshape(-1, 1, 2),
        K_orig,
        dist,
        P=K_yolo,  # map directly to YOLO pixel coords
    ).reshape(-1, 2)

    # Reconstruct output array
    if keypoints.shape[1] == 3:
        result = np.column_stack([pts_undist, keypoints[:, 2]])
    else:
        result = pts_undist
    return result.astype(np.float32)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def get_projections(camera_ids: list[int]) -> list[np.ndarray]:
    return [PROJECTION_MATRICES[cid] for cid in camera_ids]


def project_3d_to_2d(points_3d: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Project 3D world points to 2D pixel coordinates (YOLO 640×640 space)."""
    pts = np.atleast_2d(points_3d)
    homogeneous = np.hstack([pts, np.ones((pts.shape[0], 1))])
    projected = homogeneous @ P.T
    projected = projected[:, :2] / projected[:, 2:3]
    if points_3d.ndim == 1:
        return projected[0]
    return projected


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("MCFD Calibration (from B-MCFD / Auvinet et al. 2010)")
    print(f"  Image: {ORIG_W}×{ORIG_H} → {YOLO_SIZE}×{YOLO_SIZE} (scale={SCALE:.4f}, pad_top={PAD_TOP:.0f}px)")

    for cid in DEFAULT_CAMERAS:
        K_orig = build_K_matrix_original(cid)
        K_yolo = build_K_matrix_yolo(cid)
        P = build_projection_matrix(cid)
        t = build_t_vector(cid)
        print(f"\n  Camera {cid}:")
        print(f"    K_orig: fx={K_orig[0,0]:.1f}, fy={K_orig[1,1]:.1f}, u0={K_orig[0,2]:.1f}, v0={K_orig[1,2]:.1f}")
        print(f"    K_yolo: fx={K_yolo[0,0]:.1f}, fy={K_yolo[1,1]:.1f}, u0={K_yolo[0,2]:.1f}, v0={K_yolo[1,2]:.1f}")
        print(f"    t: {t.round(2)}")

    # Project a test point
    test_pt = np.array([0.0, 1.0, 0.0])  # ~chest height at origin
    print(f"\n  Projection of {test_pt}:")
    for cid in DEFAULT_CAMERAS:
        uv = project_3d_to_2d(test_pt, PROJECTION_MATRICES[cid])
        print(f"    Cam{cid}: → ({uv[0]:.0f}, {uv[1]:.0f})")
    print("\nCalibration test OK")
