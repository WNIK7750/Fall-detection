"""Event payload schema helpers for detector-to-monitor alerts."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def relative_clip_path(path: str | None) -> str | None:
    if not path:
        return None
    p = Path(path)
    try:
        return p.relative_to(Path.cwd()).as_posix()
    except ValueError:
        return str(p).replace("\\", "/")


def build_event_payload(
    *,
    client_id: str,
    device_id: str,
    address: str,
    status: str,
    p_fall: float,
    threshold_low: float,
    threshold_high: float,
    clip_path: str | None,
    scenario: int | None = None,
    frame_index: int | None = None,
    track_id: int | None = None,
    quality: float | None = None,
    llm_review: dict[str, Any] | None = None,
    trigger_reason: str = "",
    event_id: str | None = None,
) -> dict[str, Any]:
    """Build the JSON event sent over WebSocket to the monitor server."""
    if llm_review is None:
        llm_review = {
            "status": status,
            "is_fall": status == "fall",
            "confidence": None,
            "reason": trigger_reason or "No LLM review was run.",
        }

    return {
        "event_id": event_id or f"event_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:6]}",
        "client_id": client_id,
        "device_id": device_id,
        "address": address,
        "timestamp": now_iso(),
        "scenario": scenario,
        "frame_index": frame_index,
        "track_id": track_id,
        "status": status,
        "classifier": {
            "p_fall": round(float(p_fall), 4),
            "threshold_low": float(threshold_low),
            "threshold_high": float(threshold_high),
        },
        "quality": None if quality is None else round(float(quality), 4),
        "llm_review": llm_review,
        "clip_path": relative_clip_path(clip_path),
        "trigger_reason": trigger_reason,
    }
