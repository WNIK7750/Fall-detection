"""
Extract 3D features from MCFD for fall classifier training.

Strategy:
  - Use cam1 as the reference timeline.
  - Build legal 40-frame fall/normal candidates from CSV labels first.
  - Extract 3D features only for legal candidates; do not label random
    pre/post context as fall.
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
import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from pipeline.video_reader import MCFDReader, load_annotations, find_fall_segments
from pipeline.pose_inference import PoseDetector
from pipeline.triangulation import Triangulator, KalmanSmoother
from pipeline.features_3d import FloorEstimator, compute_features
from pipeline.calibration import undistort_keypoints

# Extraction parameters
FALL_STRIDE = 1
NORM_STRIDE = 5
WINDOW_SIZE = 40
FALL_MIN_CONSECUTIVE = 25
NORMAL_SAFETY_GAP = 60
EXTRACTION_WARMUP_FRAMES = 30
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
    ordered = sorted((max(0, int(s)), max(0, int(e))) for s, e in ranges if e > s)
    if not ordered:
        return []
    merged = [ordered[0]]
    for start, end in ordered[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def max_consecutive_fall_frames(
    window_start: int,
    fall_segments: list[tuple[int, int]],
    window_size: int = WINDOW_SIZE,
) -> int:
    """Return the longest consecutive fall-label run inside a window.

    CSV fall segment end frames are treated as inclusive because the original
    dataset annotation convention and previous code used `<= end`.
    """
    if not fall_segments:
        return 0
    max_run = 0
    run = 0
    for frame_idx in range(window_start, window_start + window_size):
        is_fall = any(start <= frame_idx <= end for start, end in fall_segments)
        if is_fall:
            run += 1
            max_run = max(max_run, run)
        else:
            run = 0
    return max_run


def overlaps_fall(
    window_start: int,
    fall_segments: list[tuple[int, int]],
    window_size: int = WINDOW_SIZE,
    safety_gap: int = 0,
) -> bool:
    """Whether a window intersects fall labels, optionally expanded by a gap."""
    window_end = window_start + window_size - 1
    for fall_start, fall_end in fall_segments:
        if window_start <= fall_end + safety_gap and window_end >= fall_start - safety_gap:
            return True
    return False


def build_fall_candidate_starts(
    fall_segments: list[tuple[int, int]],
    stride: int = FALL_STRIDE,
) -> list[int]:
    """Build legal fall starts before any feature extraction."""
    starts: list[int] = []
    if not fall_segments:
        return starts
    min_start = max(0, min(start for start, _ in fall_segments) - WINDOW_SIZE + 1)
    max_end = max(end for _, end in fall_segments)
    for start in range(min_start, max_end + 1, stride):
        if max_consecutive_fall_frames(start, fall_segments) >= FALL_MIN_CONSECUTIVE:
            starts.append(start)
    return starts


def build_normal_candidate_starts(
    ranges: list[tuple[int, int]],
    fall_segments: list[tuple[int, int]],
    stride: int = NORM_STRIDE,
) -> list[int]:
    """Build normal starts that contain no fall frames and keep a safety gap."""
    starts: list[int] = []
    for range_start, range_end in merge_frame_ranges(ranges):
        last_start = range_end - WINDOW_SIZE
        if last_start < range_start:
            continue
        for start in range(range_start, last_start + 1, stride):
            if not overlaps_fall(start, fall_segments, safety_gap=NORMAL_SAFETY_GAP):
                starts.append(start)
    return starts


def ranges_for_window_starts(starts: list[int]) -> list[tuple[int, int]]:
    """Convert window starts to merged read ranges with exclusive ends."""
    return merge_frame_ranges(
        (max(0, start - EXTRACTION_WARMUP_FRAMES), start + WINDOW_SIZE)
        for start in starts
    )


def scenario_frame_count(data_root: str, scenario: int, cam_id: int = 1) -> int:
    """Return the reference-camera frame count without decoding the video."""
    path = Path(data_root) / "dataset" / f"chute{scenario:02d}" / f"cam{cam_id}.avi"
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video for frame count: {path}")
        return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()


def build_feature_sequence(skeletons: list[np.ndarray]) -> list[np.ndarray | None]:
    """Compute one feature vector per original frame without compressing gaps."""
    floor_estimator = FloorEstimator()
    features: list[np.ndarray | None] = []
    prev_skel = None
    prev_feature: np.ndarray | None = None
    for skel in skeletons:
        feature = compute_features(skel, prev_skel, floor_estimator=floor_estimator)
        if feature is not None:
            feature[3] = feature[2] - prev_feature[2] if prev_feature is not None else 0.0
            prev_feature = feature
        features.append(feature)
        prev_skel = skel
    return features


def windows_from_features(
    features: list[np.ndarray | None],
    qualities: list[float],
    absolute_start: int,
    candidate_starts: list[int],
    label: int,
    min_quality: float,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Create windows for candidate starts, dropping windows with missing frames."""
    windows: list[np.ndarray] = []
    labels: list[int] = []
    kept_starts: list[int] = []
    for start in candidate_starts:
        local_start = start - absolute_start
        local_end = local_start + WINDOW_SIZE
        if local_start < 0 or local_end > len(features):
            continue
        window_features = features[local_start:local_end]
        if any(feature is None for feature in window_features):
            continue
        if float(np.mean(qualities[local_start:local_end])) < min_quality:
            continue
        window = np.array(window_features, dtype=np.float32).T
        window[3, 0] = 0.0
        window[3, 1:] = window[2, 1:] - window[2, :-1]
        windows.append(window)
        labels.append(label)
        kept_starts.append(start)
    if not windows:
        return (
            np.empty((0, 10, WINDOW_SIZE), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            [],
        )
    return np.array(windows, dtype=np.float32), np.array(labels, dtype=np.int64), kept_starts


def extract_and_window(
    data_root, scenario, camera_ids, detector, tri,
    candidate_starts: list[int],
    label: int,
    time_batch_size: int = TIME_BATCH_SIZE,
    min_quality: float = 0.35,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Extract skeletons and build strict 40-frame windows for candidates."""
    if not candidate_starts:
        return np.empty((0, 10, WINDOW_SIZE)), np.empty((0,), dtype=np.int64), []

    X_parts = []
    y_parts = []
    start_parts: list[int] = []

    frame_ranges = ranges_for_window_starts(candidate_starts)
    for start, end in merge_frame_ranges(frame_ranges):
        smoother = KalmanSmoother()
        range_skeletons = []
        range_qualities: list[float] = []
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
                    skel = smoother.update(raw_skel)
                    range_skeletons.append(skel)
                    range_qualities.append(float(quality["score"]))
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
                    skel = smoother.update(raw_skel)
                    range_skeletons.append(skel)
                    range_qualities.append(float(quality["score"]))

        if not range_skeletons:
            continue

        feature_sequence = build_feature_sequence(range_skeletons)
        range_candidates = [
            candidate
            for candidate in candidate_starts
            if absolute_start <= candidate and candidate + WINDOW_SIZE <= absolute_start + len(feature_sequence)
        ]
        X, y, kept_starts = windows_from_features(
            feature_sequence,
            range_qualities,
            absolute_start=absolute_start,
            candidate_starts=range_candidates,
            label=label,
            min_quality=min_quality,
        )
        if len(X) == 0:
            continue
        X_parts.append(X)
        y_parts.append(y)
        start_parts.extend(kept_starts)

    if not X_parts:
        return np.empty((0, 10, WINDOW_SIZE)), np.empty((0,), dtype=np.int64), []

    return np.concatenate(X_parts), np.concatenate(y_parts), start_parts


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
    parser.add_argument("--seed", type=int, default=20260713,
                        help="Seed used when sampling legal normal candidates")
    parser.add_argument("--min-quality", type=float, default=0.35,
                        help="Minimum skeleton quality for training extraction")
    parser.add_argument("--pose-fallback", action="store_true",
                        help="Enable relaxed YOLO fallback during extraction (not recommended)")
    parser.add_argument("--split-preset", choices=sorted(SPLIT_PRESETS), default="balanced",
                        help="Scenario split preset")
    parser.add_argument("--only-scenarios", type=int, nargs="+", default=None,
                        help="Restrict extraction to specific scenarios for QA")
    parser.add_argument("--output-root", default=str(REPO_ROOT / "features" / "3d"),
                        help="Directory containing train/val/test output folders")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip a split if X.npy and y.npy already exist")
    args = parser.parse_args()

    data_root = str(REPO_ROOT / "dataset" / "mcfd")
    camera_ids = [1, 4, 7]

    splits = {
        split: scenarios
        for split, scenarios in SPLIT_PRESETS[args.split_preset].items()
        if args.only_scenarios is None
        or any(scenario in args.only_scenarios for scenario in scenarios)
    }
    if args.only_scenarios is not None:
        requested = set(args.only_scenarios)
        splits = {
            split: [scenario for scenario in scenarios if scenario in requested]
            for split, scenarios in splits.items()
        }
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
        out_dir = str(Path(args.output_root) / split_name)
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

            ref_frame_count = scenario_frame_count(data_root, s, camera_ids[0])
            fall_candidates = [
                start
                for start in build_fall_candidate_starts(fall_segs)
                if start + WINDOW_SIZE <= ref_frame_count
            ]
            Xf, yf, kept_fall_starts = extract_and_window(
                data_root,
                s,
                camera_ids,
                detector,
                tri,
                candidate_starts=fall_candidates,
                label=1,
                time_batch_size=args.time_batch,
                min_quality=args.min_quality,
            )
            if len(Xf) > 0:
                X_scenario.append(Xf)
                y_scenario.append(yf)

            normal_candidates: list[int] = []
            kept_normal_starts: list[int] = []
            if not args.fast:
                if not fall_segs:
                    print("(no falls)", end=" ")

                normal_candidates = build_normal_candidate_starts(
                    [(0, ref_frame_count)],
                    fall_segs,
                )
                max_norm = args.max_normal
                if len(normal_candidates) > max_norm:
                    normal_candidates = sorted(
                        np.random.RandomState(args.seed + s).choice(
                            normal_candidates,
                            max_norm,
                            replace=False,
                        ).tolist()
                    )

                Xn, yn, kept_normal_starts = extract_and_window(
                    data_root,
                    s,
                    camera_ids,
                    detector,
                    tri,
                    candidate_starts=normal_candidates,
                    label=0,
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
            invalid_fall_labels = [
                start
                for start in kept_fall_starts
                if max_consecutive_fall_frames(start, fall_segs) < FALL_MIN_CONSECUTIVE
            ]
            invalid_normal_labels = [
                start
                for start in kept_normal_starts
                if overlaps_fall(start, fall_segs)
            ]
            if invalid_fall_labels or invalid_normal_labels:
                raise RuntimeError(
                    f"Scenario {s} label validation failed: "
                    f"fall={invalid_fall_labels[:3]} normal={invalid_normal_labels[:3]}"
                )
            print(
                f"{n_total} windows ({n_fall} fall / {n_norm} norm); "
                f"candidates={len(fall_candidates)} fall/{len(normal_candidates)} norm; "
                f"valid={len(kept_fall_starts)} fall/{len(kept_normal_starts)} norm"
            )

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
