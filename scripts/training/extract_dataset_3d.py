"""
Extract 3D features from MCFD for fall classifier training.

Strategy:
  - Fall area: dense windowing around annotated fall segments (stride=2).
  - Normal area: sparse windowing from pre-fall frames (stride=5).
  - Uses undistorted frames for correct 3D triangulation.

Usage:
    python scripts/training/extract_dataset_3d.py              # full extraction
    python scripts/training/extract_dataset_3d.py --fast       # skip normal windows (quick)
"""

import os
import sys
import argparse
import time
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from pipeline.video_reader import MCFDReader, load_annotations, find_fall_segments
from pipeline.pose_inference import PoseDetector
from pipeline.triangulation import Triangulator, KalmanSmoother
from pipeline.features_3d import extract_window
from pipeline.calibration import undistort_keypoints

# Extraction parameters
FALL_SUBSAMPLE = 2     # denser for fall segments
FALL_STRIDE = 2        # more overlapping fall windows
NORM_SUBSAMPLE = 3     # sparser for normal segments
NORM_STRIDE = 5
WINDOW_SIZE = 20
FALL_CONTEXT = 200     # frames before/after fall segment to include
MAX_HARD_NORMAL_PER_SCENARIO = 80
TIME_BATCH_SIZE = 64   # timesteps per YOLO call before flattening views

SPLIT_PRESETS = {
    # More balanced by fall-event count than the original chronological split.
    # Scenario 24 has no fall and is kept in train to increase normal coverage.
    "balanced": {
        "train": [1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 24],
        "val": [4, 17, 19, 22],
        "test": [18, 20, 21, 23],
    },
    "chronological": {
        "train": list(range(1, 19)),
        "val": list(range(19, 21)),
        "test": list(range(21, 24)),
    },
}


def merge_frame_ranges(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge overlapping or touching [start, end) frame ranges."""
    if not ranges:
        return []
    ordered = sorted((max(0, int(s)), max(0, int(e))) for s, e in ranges if e > s)
    merged = [ordered[0]]
    for start, end in ordered[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def label_windows(
    X: np.ndarray,
    frame_starts: list[int],
    fall_segments: list[tuple[int, int]],
    subsample: int = FALL_SUBSAMPLE,
) -> np.ndarray:
    """Label windows by temporal center.

    MCFD fall annotations are short (~30 frames). The old any-overlap rule
    marked pre/post context as fall if a 40-frame window touched the segment by
    a single frame. For correctness, a positive window must be centered inside
    an annotated fall interval.
    """
    y = np.zeros(len(X), dtype=np.int64)
    win_span = WINDOW_SIZE * subsample  # original frames covered by one window
    for i, start in enumerate(frame_starts):
        center = start + win_span // 2
        for fs_start, fs_end in fall_segments:
            if fs_start <= center <= fs_end:
                y[i] = 1
                break
    return y


def extract_and_window(
    data_root, scenario, camera_ids, detector, tri,
    frame_ranges, fall_segs, subsample, stride,
    time_batch_size: int = TIME_BATCH_SIZE,
    min_quality: float = 0.35,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract skeletons and build windows for given frame ranges."""
    X_parts = []
    y_parts = []

    for start, end in merge_frame_ranges(frame_ranges):
        smoother = KalmanSmoother()
        range_skeletons = []
        with MCFDReader(data_root, scenario, camera_ids,
                        start_frame=start, end_frame=end,
                        undistort=False) as reader:
            # Time-batched YOLO inference: collect N timesteps, run one GPU call
            time_batch: list[list[np.ndarray]] = []  # [[cam1,cam2,cam3], ...]
            batch_size = time_batch_size
            absolute_start = reader.ref_start_frame

            for frames in reader:
                time_batch.append(frames)
                if len(time_batch) < batch_size:
                    continue

                # Flatten: [t0_c0, t0_c1, t0_c2, t1_c0, t1_c1, t1_c2, ...]
                flat_frames = [f for t in time_batch for f in t]
                flat_kps = detector.infer_batch(flat_frames)

                # De-interleave back into per-timestep (17,3) arrays
                num_views = len(camera_ids)
                for t in range(len(time_batch)):
                    kps = [
                        undistort_keypoints(flat_kps[t * num_views + v], camera_ids[v])
                        for v in range(num_views)
                    ]
                    raw_skel = tri.triangulate(kps)
                    quality = tri.compute_quality(kps, raw_skel)
                    if quality["score"] >= min_quality:
                        skel = smoother.update(raw_skel)
                    else:
                        skel = np.full((17, 3), np.nan, dtype=np.float64)
                    range_skeletons.append(skel)
                time_batch = []

            # Process remaining frames
            if time_batch:
                flat_frames = [f for t in time_batch for f in t]
                flat_kps = detector.infer_batch(flat_frames)
                num_views = len(camera_ids)
                for t in range(len(time_batch)):
                    kps = [
                        undistort_keypoints(flat_kps[t * num_views + v], camera_ids[v])
                        for v in range(num_views)
                    ]
                    raw_skel = tri.triangulate(kps)
                    quality = tri.compute_quality(kps, raw_skel)
                    if quality["score"] >= min_quality:
                        skel = smoother.update(raw_skel)
                    else:
                        skel = np.full((17, 3), np.nan, dtype=np.float64)
                    range_skeletons.append(skel)

        if not range_skeletons:
            continue

        X, indices = extract_window(range_skeletons, window_size=WINDOW_SIZE,
                                    stride=stride, subsample=subsample)
        if len(X) == 0:
            continue

        abs_indices = [absolute_start + idx for idx in indices]
        y = label_windows(X, abs_indices, fall_segs, subsample)
        X_parts.append(X)
        y_parts.append(y)

    if not X_parts:
        return np.empty((0, 10, WINDOW_SIZE)), np.empty((0,))

    return np.concatenate(X_parts), np.concatenate(y_parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true",
                        help="Fall segments only (skip normal windows)")
    parser.add_argument("--device", default=None,
                        help="YOLO device, e.g. cuda, cpu, 0. Default: auto")
    parser.add_argument("--yolo-batch", type=int, default=32,
                        help="Ultralytics internal image batch size")
    parser.add_argument("--time-batch", type=int, default=TIME_BATCH_SIZE,
                        help="Timesteps collected before each YOLO call")
    parser.add_argument("--max-normal", type=int, default=120,
                        help="Maximum normal windows kept per scenario")
    parser.add_argument("--min-quality", type=float, default=0.35,
                        help="Minimum skeleton quality for training extraction")
    parser.add_argument("--pose-fallback", action="store_true",
                        help="Enable relaxed YOLO fallback during extraction (not recommended)")
    parser.add_argument("--split-preset", choices=sorted(SPLIT_PRESETS), default="balanced",
                        help="Scenario split preset")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip a split if X.npy and y.npy already exist")
    args = parser.parse_args()

    data_root = str(REPO_ROOT / "dataset" / "mcfd")
    camera_ids = [1, 4, 7]

    splits = SPLIT_PRESETS[args.split_preset]
    print(f"Using split preset: {args.split_preset} → {splits}")
    annotations = load_annotations(data_root)

    print("Initializing YOLO + Triangulation...")
    detector = PoseDetector(
        device=args.device,
        infer_batch_size=args.yolo_batch,
        enable_fallback=args.pose_fallback,
    )
    from pipeline.calibration import DEFAULT_PROJECTIONS
    tri = Triangulator(DEFAULT_PROJECTIONS)

    for split_name, scenario_range in splits.items():
        out_dir = str(REPO_ROOT / "features" / "3d" / split_name)
        x_path = os.path.join(out_dir, "X.npy")
        y_path = os.path.join(out_dir, "y.npy")
        if args.skip_existing and os.path.exists(x_path) and os.path.exists(y_path):
            print(f"  [{split_name}] existing files found, skipping")
            continue

        split_t0 = time.perf_counter()
        X_all, y_all = [], []

        for s in scenario_range:
            fall_segs = find_fall_segments(annotations, s, cam_id=camera_ids[0])
            print(f"  [{split_name}] Scenario {s:2d}: {len(fall_segs)} fall segs "
                  f"@ {[f'{a}-{b}' for a,b in fall_segs[:3]]}...", end=" ", flush=True)

            X_scenario = []
            y_scenario = []

            if fall_segs:
                # --- Fall area: dense windowing around each fall segment ---
                fall_ranges = []
                earliest_fall = float('inf')
                for fs_start, fs_end in fall_segs:
                    r_start = max(0, fs_start - FALL_CONTEXT)
                    r_end = fs_end + FALL_CONTEXT
                    fall_ranges.append((r_start, r_end))
                    earliest_fall = min(earliest_fall, fs_start)

                Xf, yf = extract_and_window(
                    data_root, s, camera_ids, detector, tri,
                    fall_ranges, fall_segs,
                    subsample=FALL_SUBSAMPLE, stride=FALL_STRIDE,
                    time_batch_size=args.time_batch,
                    min_quality=args.min_quality,
                )
                # Keep centered fall windows plus a capped set of hard negatives
                # from near-fall context. This reduces pre/post-fall false positives.
                if len(Xf) > 0:
                    fall_mask = yf == 1
                    if fall_mask.sum() > 0:
                        X_scenario.append(Xf[fall_mask])
                        y_scenario.append(yf[fall_mask])
                    hard_norm_mask = ~fall_mask
                    hard_norm_count = int(hard_norm_mask.sum())
                    if hard_norm_count > 0:
                        hard_indices = np.flatnonzero(hard_norm_mask)
                        max_hard = min(MAX_HARD_NORMAL_PER_SCENARIO, len(hard_indices))
                        if len(hard_indices) > max_hard:
                            hard_indices = np.random.RandomState(4200 + s).choice(
                                hard_indices,
                                max_hard,
                                replace=False,
                            )
                        X_scenario.append(Xf[hard_indices])
                        y_scenario.append(yf[hard_indices])

                # --- Normal area: sparse windowing, avoid fall region ---
                if not args.fast:
                    NORM_WINDOW = 2000  # frames to read for normal area
                    if earliest_fall > FALL_CONTEXT + NORM_WINDOW:
                        # Plenty of frames before fall
                        norm_ranges = [(int(earliest_fall) - FALL_CONTEXT - NORM_WINDOW,
                                        int(earliest_fall) - FALL_CONTEXT // 2)]
                    elif earliest_fall > FALL_CONTEXT:
                        # Some frames before fall — take what's available
                        norm_ranges = [(0, int(earliest_fall) - FALL_CONTEXT // 2)]
                    else:
                        # Fall at video start — take frames after last fall
                        latest_fall = max(fs_end for _, fs_end in fall_segs)
                        norm_start = latest_fall + FALL_CONTEXT // 2
                        norm_ranges = [(norm_start, norm_start + NORM_WINDOW)]

                    Xn, yn = extract_and_window(
                        data_root, s, camera_ids, detector, tri,
                        norm_ranges, fall_segs,
                        subsample=NORM_SUBSAMPLE, stride=NORM_STRIDE,
                        time_batch_size=args.time_batch,
                        min_quality=args.min_quality,
                    )
                    # Cap normal windows per scenario to keep balance reasonable
                    max_norm = args.max_normal
                    if len(Xn) > max_norm:
                        keep = np.random.RandomState(42 + s).choice(
                            len(Xn), max_norm, replace=False)
                        Xn, yn = Xn[keep], yn[keep]
                    if len(Xn) > 0:
                        X_scenario.append(Xn)
                        y_scenario.append(yn)
            else:
                # No annotated falls — read a chunk as normal
                print("(no falls)", end=" ")
                Xn, yn = extract_and_window(
                    data_root, s, camera_ids, detector, tri,
                    [(0, 600)], fall_segs=None or [],
                    subsample=NORM_SUBSAMPLE, stride=NORM_STRIDE,
                    time_batch_size=args.time_batch,
                    min_quality=args.min_quality,
                )
                if len(Xn) > 0:
                    X_scenario.append(Xn)
                    y_scenario.append(yn)

            if X_scenario:
                X_combined = np.concatenate(X_scenario)
                y_combined = np.concatenate(y_scenario)
            else:
                X_combined = np.empty((0, 10, WINDOW_SIZE))
                y_combined = np.empty((0,))

            n_fall = int(y_combined.sum())
            n_total = len(y_combined)
            n_norm = n_total - n_fall
            print(f"{n_total} windows ({n_fall} fall / {n_norm} norm)")

            if n_total > 0:
                X_all.append(X_combined)
                y_all.append(y_combined)

        if X_all:
            X_out = np.concatenate(X_all)
            y_out = np.concatenate(y_all)
        else:
            X_out = np.empty((0, 10, WINDOW_SIZE))
            y_out = np.empty((0,))

        os.makedirs(out_dir, exist_ok=True)
        np.save(x_path, X_out)
        np.save(y_path, y_out)
        pos = int(y_out.sum()) if len(y_out) > 0 else 0
        elapsed = time.perf_counter() - split_t0
        ratio = f"{y_out.mean():.1%}" if len(y_out) > 0 else "n/a"
        print(f"  Saved {len(X_out)} windows ({pos} falls, {ratio}) → {out_dir}/ "
              f"in {elapsed/60:.1f} min")

    print("\nDone. Next: python scripts/training/train_3d.py")


if __name__ == "__main__":
    main()
