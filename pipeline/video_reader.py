"""
Multi-view synchronized video reader for MCFD (Multiple Cameras Fall Dataset).

MCFD structure:
    dataset/mcfd/dataset/chuteXX/cam1.avi ~ cam8.avi

Usage:
    reader = MCFDReader("dataset/mcfd", scenario=1, camera_ids=[1, 4, 7])
    for frames in reader:
        # frames: list of 3 numpy arrays (BGR, H×W×3)
        process(frames)
"""

import os
import csv
from typing import Iterator, Optional

import cv2
import numpy as np


class MCFDReader:
    """Synchronized multi-view video reader for MCFD."""

    def __init__(
        self,
        data_root: str,
        scenario: int,
        camera_ids: list[int],
        target_size: int = 640,
        start_frame: int = 0,
        end_frame: Optional[int] = None,
    ):
        """
        Args:
            data_root: Path to MCFD root (contains dataset/, data_tuple3.csv).
            scenario: Scenario number (1-24).
            camera_ids: Which cameras to read, e.g. [1, 4, 7].
            target_size: Resize frames to target_size × target_size (keep aspect ratio via padding).
            start_frame: First frame to read (0-based).
            end_frame: Last frame to read (exclusive); None = read all.
        """
        self.scenario = scenario
        self.camera_ids = camera_ids
        self.target_size = target_size
        self.num_views = len(camera_ids)

        scenario_dir = os.path.join(data_root, "dataset", f"chute{scenario:02d}")
        if not os.path.isdir(scenario_dir):
            raise FileNotFoundError(f"Scenario directory not found: {scenario_dir}")

        self.caps: list[cv2.VideoCapture] = []
        for cid in camera_ids:
            video_path = os.path.join(scenario_dir, f"cam{cid}.avi")
            if not os.path.isfile(video_path):
                raise FileNotFoundError(f"Video not found: {video_path}")
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open video: {video_path}")
            self.caps.append(cap)

        # Seek to start_frame and determine total frames
        self.total_frames = int(self.caps[0].get(cv2.CAP_PROP_FRAME_COUNT))
        for i, cap in enumerate(self.caps):
            fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if fc != self.total_frames:
                print(f"[WARN] cam{self.camera_ids[i]}: {fc} frames vs cam{self.camera_ids[0]}: {self.total_frames}")
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        self.current_frame = start_frame
        self.end_frame = end_frame if end_frame is not None else self.total_frames
        self.end_frame = min(self.end_frame, self.total_frames)

    @property
    def remaining(self) -> int:
        return max(0, self.end_frame - self.current_frame)

    def _resize_pad(self, frame: np.ndarray) -> np.ndarray:
        """Resize to target_size × target_size keeping aspect ratio, pad with black."""
        h, w = frame.shape[:2]
        scale = self.target_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        pad_h = (self.target_size - new_h) // 2
        pad_w = (self.target_size - new_w) // 2
        padded = np.zeros((self.target_size, self.target_size, 3), dtype=np.uint8)
        padded[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized
        return padded

    def __iter__(self) -> Iterator[list[np.ndarray]]:
        while self.current_frame < self.end_frame:
            frames: list[np.ndarray] = []
            all_ok = True
            for cap in self.caps:
                ok, frame = cap.read()
                if not ok:
                    all_ok = False
                    break
                frame = self._resize_pad(frame)
                frames.append(frame)

            if not all_ok:
                break

            self.current_frame += 1
            yield frames

    def close(self):
        for cap in self.caps:
            cap.release()

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def load_annotations(data_root: str) -> list[dict]:
    """Load data_tuple3.csv annotations.

    Returns list of dicts with keys: chute, cam, start, end, label.
    label: 0 = no fall, 1 = fall.
    """
    csv_path = os.path.join(data_root, "data_tuple3.csv")
    annotations = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            annotations.append({
                "chute": int(float(row["chute"])),
                "cam": int(float(row["cam"])),
                "start": int(float(row["start"])),
                "end": int(float(row["end"])),
                "label": int(float(row["label"])),
            })
    return annotations


def find_fall_segments(
    annotations: list[dict],
    scenario: int,
) -> list[tuple[int, int]]:
    """Return list of (start_frame, end_frame) where label==1 for given scenario."""
    segments = []
    for a in annotations:
        if a["chute"] == scenario and a["label"] == 1:
            segments.append((a["start"], a["end"]))
    return segments


# --- Quick test ---
if __name__ == "__main__":
    root = os.path.join(os.path.dirname(__file__), "..", "dataset", "mcfd")
    root = os.path.abspath(root)

    anns = load_annotations(root)
    falls = find_fall_segments(anns, 1)
    print(f"Scenario 1 fall segments: {falls}")

    with MCFDReader(root, scenario=1, camera_ids=[1, 4, 7]) as reader:
        print(f"Total frames: {reader.total_frames}, {reader.num_views} views")
        for i, frames in enumerate(reader):
            if i >= 5:
                break
            print(f"  Frame {i}: shapes={[f.shape for f in frames]}, dtype={frames[0].dtype}")
    print("MCFDReader test OK")
