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
    """Synchronized multi-view video reader for MCFD.

    Supports per-camera frame offsets for approximate time synchronization.
    Offsets are estimated from data_tuple3.csv fall segment annotations.

    Optional lens undistortion using precomputed cv2 remap maps.

    For YOLO pose extraction, prefer undistort=False and then undistort the
    detected keypoints before triangulation. YOLO is more stable on the natural
    camera image, while triangulation still needs undistorted coordinates.
    """

    def __init__(
        self,
        data_root: str,
        scenario: int,
        camera_ids: list[int],
        target_size: int = 640,
        start_frame: int = 0,
        end_frame: Optional[int] = None,
        offsets: Optional[dict[int, float]] = None,
        ref_cam: int = 1,
        undistort: bool = False,
    ):
        """
        Args:
            data_root: Path to MCFD root (contains dataset/, data_tuple3.csv).
            scenario: Scenario number (1-24).
            camera_ids: Which cameras to read, e.g. [1, 4, 7].
            target_size: Resize frames to target_size × target_size.
            start_frame: First frame to read (0-based, in reference camera coords).
            end_frame: Last frame to read (exclusive); None = read all.
            offsets: dict[cam_id] -> frame offset. Positive = target is ahead.
                     e.g. {5: 20} means cam5 frame 1120 ≈ cam1 frame 1100.
                     If None, auto-compute from annotations.
            ref_cam: Reference camera ID for offsets (default cam1).
            undistort: If True, apply lens undistortion before resize+pad.
                       For YOLO pose, keep this False and use
                       calibration.undistort_keypoints() before triangulation.
        """
        self.scenario = scenario
        self.camera_ids = camera_ids
        self.target_size = target_size
        self.num_views = len(camera_ids)
        self._undistort = undistort
        self._undistort_maps: Optional[tuple[list, list]] = None

        # Initialize undistort maps if requested
        if undistort:
            from pipeline.calibration import get_undistort_maps
            try:
                m1_list, m2_list = get_undistort_maps(camera_ids)
                self._undistort_maps = (m1_list, m2_list)
                print(f"[UNDISTORT] Lens correction enabled for cameras {camera_ids}")
            except Exception as e:
                print(f"[UNDISTORT] WARNING: Failed to init undistort maps: {e}")
                self._undistort = False

        # Compute or use offsets
        if offsets is None:
            from pipeline.sync import compute_offsets
            self._offsets = dict(compute_offsets(data_root, scenario, ref_cam))
        else:
            self._offsets = dict(offsets)  # copy to avoid mutating caller's dict

        scenario_dir = os.path.join(data_root, "dataset", f"chute{scenario:02d}")
        if not os.path.isdir(scenario_dir):
            raise FileNotFoundError(f"Scenario directory not found: {scenario_dir}")

        self.caps: list[cv2.VideoCapture] = []
        self._cam_frame_counts: list[int] = []
        for cid in camera_ids:
            video_path = os.path.join(scenario_dir, f"cam{cid}.avi")
            if not os.path.isfile(video_path):
                raise FileNotFoundError(f"Video not found: {video_path}")
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open video: {video_path}")
            self.caps.append(cap)
            self._cam_frame_counts.append(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))

        # Determine seek positions with offsets.
        #
        # offset[cid] = N means: cam{cid}'s frame K ≈ cam_ref's frame (K - N)
        #   e.g., offset[5] = 20 → cam5 frame 1120 ≈ cam1 frame 1100
        #
        # To read synchronized frames starting at start_frame (ref time):
        #   cam{cid} reads frame: start_frame + offset[cid]
        #
        # Round all offsets to int (from np.median float)
        for cid in camera_ids:
            self._offsets[cid] = int(round(self._offsets.get(cid, 0)))

        all_offs = [self._offsets.get(cid, 0) for cid in camera_ids]
        min_off = min(all_offs)

        # Shift all cameras so the earliest-starting one starts at frame 0
        if min_off < 0:
            shift = -min_off
            start_frame += shift
            # Don't add to offsets — offsets are relative to ref already
            # The shift is absorbed into start_frame

        self.ref_start_frame = start_frame

        self._seek_positions = []
        self._effective_lengths = []
        for i, cid in enumerate(camera_ids):
            # For negative offsets: cam is behind ref, needs fewer seek frames
            # seek_to = max(0, start_frame + offset) — but after shift, start_frame
            # already includes -min_off, so offsets are non-negative.
            off = self._offsets.get(cid, 0)
            seek_to = max(0, start_frame + off)
            self._seek_positions.append(seek_to)
            self._effective_lengths.append(self._cam_frame_counts[i] - seek_to)

            if off != 0:
                print(f"[SYNC] cam{cid} offset={off:+d} → seek frame {seek_to}")

        # Seek all cameras
        for i, cap in enumerate(self.caps):
            cap.set(cv2.CAP_PROP_POS_FRAMES, self._seek_positions[i])

        # Total frames = min effective length across all cameras
        self.total_frames = min(self._effective_lengths)
        self.current_frame = 0  # relative to seek start

        if end_frame is None:
            self.end_frame = self.total_frames
        else:
            # end_frame is in reference-camera absolute coordinates; iteration
            # uses frames relative to ref_start_frame after seeking.
            requested = max(0, end_frame - self.ref_start_frame)
            self.end_frame = min(requested, self.total_frames)

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
            for i, cap in enumerate(self.caps):
                ok, frame = cap.read()
                if not ok:
                    all_ok = False
                    break
                # Undistort before resize (corrects lens distortion for 3D)
                if self._undistort and self._undistort_maps is not None:
                    frame = cv2.remap(
                        frame,
                        self._undistort_maps[0][i],
                        self._undistort_maps[1][i],
                        cv2.INTER_LINEAR,
                    )
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
    cam_id: int | None = None,
) -> list[tuple[int, int]]:
    """Return (start_frame, end_frame) fall labels for a scenario.

    If cam_id is provided, only labels from that camera are returned. This is
    important when downstream code operates in a reference-camera timeline.
    """
    segments = []
    for a in annotations:
        if a["chute"] == scenario and a["label"] == 1:
            if cam_id is not None and a["cam"] != cam_id:
                continue
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
