"""
Extract 3D features from MCFD for fall classifier training.

Strategy:
  - Fall area: dense windowing around annotated fall segments (stride=2).
  - Normal area: sparse windowing from pre-fall frames (stride=5).
  - Uses undistorted frames for correct 3D triangulation.

Usage:
    python extract_dataset_3d.py              # full extraction
    python extract_dataset_3d.py --fast       # skip normal windows (quick)
"""

import os
import sys
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from pipeline.video_reader import MCFDReader, load_annotations, find_fall_segments
from pipeline.pose_inference import PoseDetector
from pipeline.triangulation import Triangulator, KalmanSmoother
from pipeline.features_3d import extract_window

# Extraction parameters
FALL_SUBSAMPLE = 2     # denser for fall segments
FALL_STRIDE = 2        # more overlapping fall windows
NORM_SUBSAMPLE = 3     # sparser for normal segments
NORM_STRIDE = 5
WINDOW_SIZE = 20
FALL_CONTEXT = 200     # frames before/after fall segment to include


def label_windows(
    X: np.ndarray,
    frame_starts: list[int],
    fall_segments: list[tuple[int, int]],
    subsample: int = FALL_SUBSAMPLE,
    offset: int = 0,
) -> np.ndarray:
    """Label windows: 1 if any overlap with a fall segment."""
    y = np.zeros(len(X), dtype=np.int64)
    win_span = WINDOW_SIZE * subsample  # original frames covered by one window
    for i, start in enumerate(frame_starts):
        end = start + win_span
        for fs_start, fs_end in fall_segments:
            if not (end < fs_start or (start - offset) > fs_end):
                y[i] = 1
                break
    return y


def extract_and_window(
    data_root, scenario, camera_ids, detector, tri,
    frame_ranges, fall_segs, subsample, stride, offset=0,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract skeletons and build windows for given frame ranges."""
    smoother = KalmanSmoother()
    all_skeletons = []

    for start, end in frame_ranges:
        with MCFDReader(data_root, scenario, camera_ids,
                        start_frame=start, end_frame=end) as reader:
            # Time-batched YOLO inference: collect N timesteps, run one GPU call
            time_batch: list[list[np.ndarray]] = []  # [[cam1,cam2,cam3], ...]
            batch_size = 64  # 64 timesteps × 3 views = 192 images per YOLO call

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
                    kps = [flat_kps[t * num_views + v] for v in range(num_views)]
                    skel = tri.triangulate(kps)
                    skel = smoother.update(skel)
                    all_skeletons.append(skel)
                time_batch = []

            # Process remaining frames
            if time_batch:
                flat_frames = [f for t in time_batch for f in t]
                flat_kps = detector.infer_batch(flat_frames)
                num_views = len(camera_ids)
                for t in range(len(time_batch)):
                    kps = [flat_kps[t * num_views + v] for v in range(num_views)]
                    skel = tri.triangulate(kps)
                    skel = smoother.update(skel)
                    all_skeletons.append(skel)

    if not all_skeletons:
        return np.empty((0, WINDOW_SIZE, 10)), np.empty((0,))

    X, indices = extract_window(all_skeletons, window_size=WINDOW_SIZE,
                                stride=stride, subsample=subsample)
    if len(X) == 0:
        return np.empty((0, 10, WINDOW_SIZE)), np.empty((0,))

    y = label_windows(X, indices, fall_segs, subsample, offset)
    return X, y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true",
                        help="Fall segments only (skip normal windows)")
    args = parser.parse_args()

    data_root = os.path.join(os.path.dirname(__file__), "dataset", "mcfd")
    camera_ids = [1, 4, 7]

    splits = {"train": range(1, 19), "val": range(19, 21), "test": range(21, 24)}
    annotations = load_annotations(data_root)

    print("Initializing YOLO + Triangulation...")
    detector = PoseDetector()
    from pipeline.calibration import DEFAULT_PROJECTIONS
    tri = Triangulator(DEFAULT_PROJECTIONS)

    for split_name, scenario_range in splits.items():
        X_all, y_all = [], []

        for s in scenario_range:
            fall_segs = find_fall_segments(annotations, s)
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
                )
                # Only keep fall-labeled windows — context frames are NOT normal
                if len(Xf) > 0:
                    fall_mask = yf == 1
                    if fall_mask.sum() > 0:
                        X_scenario.append(Xf[fall_mask])
                        y_scenario.append(yf[fall_mask])

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
                    )
                    # Cap normal windows per scenario to keep balance reasonable
                    max_norm = 120
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

        out_dir = os.path.join("features", "3d", split_name)
        os.makedirs(out_dir, exist_ok=True)
        np.save(os.path.join(out_dir, "X.npy"), X_out)
        np.save(os.path.join(out_dir, "y.npy"), y_out)
        pos = int(y_out.sum()) if len(y_out) > 0 else 0
        print(f"  Saved {len(X_out)} windows ({pos} falls, {y_out.mean():.1%}) → {out_dir}/")

    print("\nDone. Next: python train_3d.py")


if __name__ == "__main__":
    main()
