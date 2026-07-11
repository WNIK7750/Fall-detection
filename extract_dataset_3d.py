"""
Extract 3D features from MCFD for fall classifier training.

Strategy: process frames around annotated fall segments (with context),
          plus a downsample of normal frames for balance.

Usage:
    python extract_dataset_3d.py              # full extraction
    python extract_dataset_3d.py --fast       # fall segments only (quick)
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


def extract_frames(
    data_root: str,
    scenario: int,
    camera_ids: list[int],
    detector: PoseDetector,
    tri: Triangulator,
    frame_ranges: list[tuple[int, int]],
) -> list[np.ndarray]:
    """Extract 3D skeletons from specific frame ranges."""
    smoother = KalmanSmoother()
    all_skeletons = []

    for start, end in frame_ranges:
        with MCFDReader(data_root, scenario, camera_ids,
                        start_frame=start, end_frame=end) as reader:
            for frames in reader:
                kps = detector.infer_batch(frames)
                skel = tri.triangulate(kps)
                skel = smoother.update(skel)
                all_skeletons.append(skel)

    return all_skeletons


def label_windows(
    X: np.ndarray,
    frame_starts: list[int],
    fall_segments: list[tuple[int, int]],
    offset: int = 0,
) -> np.ndarray:
    """Label windows: 1 if any overlap with a fall segment."""
    y = np.zeros(len(X), dtype=np.int64)
    for i, start in enumerate(frame_starts):
        end = start + 20 * 3  # 20-frame window, subsample=3 → ~60 original frames
        for fs_start, fs_end in fall_segments:
            if not (end < fs_start or (start - offset) > fs_end):
                y[i] = 1
                break
    return y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true", help="Fall segments only (fast)")
    parser.add_argument("--context", type=int, default=150, help="Context frames around fall")
    args = parser.parse_args()

    data_root = os.path.join(os.path.dirname(__file__), "dataset", "mcfd")
    camera_ids = [1, 4, 7]

    splits = {"train": range(1, 18), "val": range(18, 20), "test": range(20, 23)}
    annotations = load_annotations(data_root)

    print("Initializing YOLO + Triangulation...")
    detector = PoseDetector()
    from pipeline.calibration import DEFAULT_PROJECTIONS
    tri = Triangulator(DEFAULT_PROJECTIONS)

    for split_name, scenario_range in splits.items():
        X_all, y_all = [], []
        ctx = args.context

        for s in scenario_range:
            fall_segs = find_fall_segments(annotations, s)
            print(f"  [{split_name}] Scenario {s:2d}: {len(fall_segs)} fall segs "
                  f"@ {[f'{a}-{b}' for a,b in fall_segs[:3]]}...", end=" ", flush=True)

            if args.fast and fall_segs:
                # Only fall segments + context
                ranges = []
                for fs_start, fs_end in fall_segs:
                    ranges.append((max(0, fs_start - ctx), fs_end + ctx))
                skeletons = extract_frames(data_root, s, camera_ids,
                                           detector, tri, ranges)
                offset = ranges[0][0] if ranges else 0
                X, indices = extract_window(skeletons, window_size=20,
                                            stride=5, subsample=3)
                if len(X) > 0:
                    y = label_windows(X, indices, fall_segs, offset)
                    X_all.append(X)
                    y_all.append(y)
            else:
                # Full scenario (or --fast with no fall segs)
                skeletons = extract_frames(data_root, s, camera_ids,
                                           detector, tri, [(0, 800)])
                X, indices = extract_window(skeletons, window_size=20,
                                            stride=5, subsample=3)
                if len(X) > 0:
                    y = label_windows(X, indices, fall_segs)
                    X_all.append(X)
                    y_all.append(y)

            n_fall = int(y_all[-1].sum()) if y_all else 0
            n_total = len(y_all[-1]) if y_all else 0
            print(f"{n_total} windows ({n_fall} falls)")

        if X_all:
            X_out = np.concatenate(X_all)
            y_out = np.concatenate(y_all)
        else:
            X_out = np.empty((0, 10, 20))
            y_out = np.empty((0,))

        out_dir = os.path.join("features", "3d", split_name)
        os.makedirs(out_dir, exist_ok=True)
        np.save(os.path.join(out_dir, "X.npy"), X_out)
        np.save(os.path.join(out_dir, "y.npy"), y_out)
        pos = int(y_out.sum()) if len(y_out) > 0 else 0
        print(f"  Saved {len(X_out)} windows ({pos} falls) → {out_dir}/")

    print("\nDone. Next: python train_3d.py")


if __name__ == "__main__":
    main()
