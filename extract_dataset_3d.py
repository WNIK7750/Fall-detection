"""
Extract 3D features from MCFD for training the 3-class fall classifier.

Pipeline per scenario:
  MCFD videos → YOLOv8n-pose (3-view) → Triangulation → 3D Features → .npy

Split: scenarios 1-17 train, 18-19 val, 20-22 test (~77/9/14%)
Label: any 20-frame window containing a fall frame = class 1 (fall), else 0 (normal).

Usage:
    python extract_dataset_3d.py                # extract all
    python extract_dataset_3d.py --limit 5      # extract first 5 scenarios only
    python extract_dataset_3d.py --skip-train   # skip already-extracted train set
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


def extract_scenario(
    data_root: str,
    scenario: int,
    camera_ids: list[int],
    detector: PoseDetector,
    tri: Triangulator,
    annotations: list[dict],
    max_frames: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract features and labels from one MCFD scenario.

    Returns:
        X: (N, 10, 20) feature windows.
        y: (N,) binary labels (0=normal, 1=fall).
    """
    smoother = KalmanSmoother()
    all_skeletons = []

    with MCFDReader(data_root, scenario, camera_ids) as reader:
        total = reader.total_frames
        if max_frames > 0:
            total = min(total, max_frames)

        for i, frames in enumerate(reader):
            if max_frames > 0 and i >= max_frames:
                break
            kps = detector.infer_batch(frames)
            skel = tri.triangulate(kps)
            skel = smoother.update(skel)
            all_skeletons.append(skel)

    # Extract sliding windows (subsample every 3rd frame: 120fps→40fps)
    X, frame_starts = extract_window(all_skeletons, window_size=20, stride=5, subsample=3)

    if len(X) == 0:
        return np.empty((0, 10, 20)), np.empty((0,))

    # Label windows based on data_tuple3.csv annotations
    fall_segments = find_fall_segments(annotations, scenario)
    y = np.zeros(len(X), dtype=np.int64)

    for i, start_frame in enumerate(frame_starts):
        end_frame = start_frame + 20 * 3  # window covers ~60 original frames
        for fs_start, fs_end in fall_segments:
            if not (end_frame < fs_start or start_frame > fs_end):
                y[i] = 1
                break

    return X, y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="Max scenarios to process")
    parser.add_argument("--max-frames", type=int, default=800, help="Max frames per scenario")
    args = parser.parse_args()

    data_root = os.path.join(os.path.dirname(__file__), "dataset", "mcfd")
    camera_ids = [1, 4, 7]

    # Split: train=1-17, val=18-19, test=20-22
    splits = {"train": range(1, 18), "val": range(18, 20), "test": range(20, 23)}
    scenario_limit = args.limit

    print("Initializing...")
    detector = PoseDetector()
    from pipeline.calibration import DEFAULT_PROJECTIONS
    tri = Triangulator(DEFAULT_PROJECTIONS)
    annotations = load_annotations(data_root)

    for split_name, scenario_range in splits.items():
        X_all, y_all = [], []

        for s in scenario_range:
            if scenario_limit and s > scenario_limit:
                break

            print(f"  [{split_name}] Scenario {s:2d}...", end=" ", flush=True)
            X, y = extract_scenario(data_root, s, camera_ids, detector, tri,
                                    annotations, max_frames=args.max_frames)
            if len(X) > 0:
                X_all.append(X)
                y_all.append(y)
                n_fall = int(y.sum())
                print(f"{len(X)} windows ({n_fall} falls)")
            else:
                print("0 windows (skipped)")

        if X_all:
            X_out = np.concatenate(X_all, axis=0)
            y_out = np.concatenate(y_all, axis=0)
        else:
            X_out = np.empty((0, 10, 20))
            y_out = np.empty((0,))

        out_dir = os.path.join("features", "3d", split_name)
        os.makedirs(out_dir, exist_ok=True)
        np.save(os.path.join(out_dir, "X.npy"), X_out)
        np.save(os.path.join(out_dir, "y.npy"), y_out)
        print(f"  Saved {len(X_out)} windows to {out_dir}/")

    print("\nDone. Run train_3d.py to train the classifier.")


if __name__ == "__main__":
    main()
