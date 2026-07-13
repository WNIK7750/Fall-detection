"""Privacy-preserving rolling video buffer for event clips."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class BufferedFrame:
    frame_index: int
    timestamp_sec: float
    image: np.ndarray
    views: list[np.ndarray] | None = None


@dataclass
class ClipExportInfo:
    path: str
    frame_count: int
    fps: float
    start_frame: int
    end_frame: int
    start_timestamp_sec: float
    end_timestamp_sec: float


class RollingClipBuffer:
    """Keep recent frames in memory and export clips only on trigger.

    The buffer does not write continuously to disk. It stores compact BGR
    frames, usually a multi-view collage, and exports only when the event loop
    asks for a suspected/fall evidence clip.
    """

    def __init__(self, max_frames: int = 300, fps: float = 30.0):
        self.max_frames = max_frames
        self.fps = fps
        self._frames: deque[BufferedFrame] = deque(maxlen=max_frames)

    def append(
        self,
        frame_index: int,
        image: np.ndarray,
        timestamp_sec: float | None = None,
        views: list[np.ndarray] | None = None,
    ) -> None:
        if timestamp_sec is None:
            timestamp_sec = frame_index / max(self.fps, 1e-6)
        copied_views = None if views is None else [view.copy() for view in views]
        self._frames.append(BufferedFrame(frame_index, timestamp_sec, image.copy(), copied_views))

    def __len__(self) -> int:
        return len(self._frames)

    def frame_range(self) -> tuple[int, int] | None:
        """Return (min_frame, max_frame) of buffered frames, or None if empty."""
        if not self._frames:
            return None
        indices = [item.frame_index for item in self._frames]
        return (min(indices), max(indices))

    def can_export(self, frames: int) -> bool:
        return len(self._frames) >= frames

    def latest(self, frames: int) -> list[BufferedFrame]:
        if frames <= 0:
            return []
        return list(self._frames)[-frames:]

    def range_by_frame(self, start_frame: int, end_frame: int) -> list[BufferedFrame]:
        """Return buffered frames in inclusive [start_frame, end_frame]."""
        return [
            item for item in self._frames
            if start_frame <= item.frame_index <= end_frame
        ]

    def export_latest_info(
        self,
        output_path: str | Path,
        frames: int = 90,
        fps: float | None = None,
    ) -> ClipExportInfo | None:
        """Export latest N frames and return path plus timing metadata."""
        if not self.can_export(frames):
            return None

        selected = self.latest(frames)
        if not selected:
            return None

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        h, w = selected[0].image.shape[:2]
        output_fps = fps or self.fps
        writer = _open_video_writer(output, output_fps, (w, h))
        try:
            for item in selected:
                frame = item.image
                if frame.shape[:2] != (h, w):
                    frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
                writer.write(frame)
        finally:
            writer.release()

        return ClipExportInfo(
            path=str(output),
            frame_count=len(selected),
            fps=float(output_fps),
            start_frame=int(selected[0].frame_index),
            end_frame=int(selected[-1].frame_index),
            start_timestamp_sec=float(selected[0].timestamp_sec),
            end_timestamp_sec=float(selected[-1].timestamp_sec),
        )

    def export_range_info(
        self,
        output_path: str | Path,
        start_frame: int,
        end_frame: int,
        fps: float | None = None,
    ) -> ClipExportInfo | None:
        """Export buffered frames by inclusive frame range.

        Tolerates gaps (e.g. from video decoding errors). Returns None only
        when fewer than half the expected frames are available.
        """
        selected = self.range_by_frame(start_frame, end_frame)
        expected = end_frame - start_frame + 1
        if len(selected) < max(1, expected // 2):
            return None
        return _export_selected(output_path, selected, fps or self.fps)

    def export_latest(self, output_path: str | Path, frames: int = 90, fps: float | None = None) -> str | None:
        """Export latest N frames to a clip. Returns path or None if unavailable."""
        info = self.export_latest_info(output_path, frames=frames, fps=fps)
        return None if info is None else info.path


def _open_video_writer(output: Path, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
    """Open a stable event-clip writer.

    The dashboard decodes clips server-side into cached JPEG frames, so these
    mp4 files do not need browser-native H264. Avoid probing H264/OpenH264 here
    because missing OpenH264 DLLs produce noisy errors on Windows.
    """
    candidates = [
        ("mp4v", output),
    ]
    for fourcc_name, path in candidates:
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc_name), fps, size)
        if writer.isOpened():
            return writer
        writer.release()
    raise RuntimeError(f"Cannot open output video: {output}")


def _export_selected(
    output_path: str | Path,
    selected: list[BufferedFrame],
    fps: float,
) -> ClipExportInfo | None:
    if not selected:
        return None

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    h, w = selected[0].image.shape[:2]
    writer = _open_video_writer(output, fps, (w, h))
    try:
        for item in selected:
            frame = item.image
            if frame.shape[:2] != (h, w):
                frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
            writer.write(frame)
    finally:
        writer.release()

    return ClipExportInfo(
        path=str(output),
        frame_count=len(selected),
        fps=float(fps),
        start_frame=int(selected[0].frame_index),
        end_frame=int(selected[-1].frame_index),
        start_timestamp_sec=float(selected[0].timestamp_sec),
        end_timestamp_sec=float(selected[-1].timestamp_sec),
    )
