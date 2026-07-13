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

    def append(self, frame_index: int, image: np.ndarray, timestamp_sec: float | None = None) -> None:
        if timestamp_sec is None:
            timestamp_sec = frame_index / max(self.fps, 1e-6)
        self._frames.append(BufferedFrame(frame_index, timestamp_sec, image.copy()))

    def __len__(self) -> int:
        return len(self._frames)

    def can_export(self, frames: int) -> bool:
        return len(self._frames) >= frames

    def latest(self, frames: int) -> list[BufferedFrame]:
        if frames <= 0:
            return []
        return list(self._frames)[-frames:]

    def export_latest(self, output_path: str | Path, frames: int = 90, fps: float | None = None) -> str | None:
        """Export latest N frames to an mp4. Returns path or None if unavailable."""
        if not self.can_export(frames):
            return None

        selected = self.latest(frames)
        if not selected:
            return None

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        h, w = selected[0].image.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output), fourcc, fps or self.fps, (w, h))
        if not writer.isOpened():
            raise RuntimeError(f"Cannot open output video: {output}")
        try:
            for item in selected:
                frame = item.image
                if frame.shape[:2] != (h, w):
                    frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
                writer.write(frame)
        finally:
            writer.release()

        return str(output)
