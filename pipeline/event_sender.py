"""WebSocket sender for monitor events."""

from __future__ import annotations

import asyncio
import json
from typing import Any


async def send_event_ws(url: str, payload: dict[str, Any], timeout: float = 10.0) -> dict[str, Any]:
    """Send one event to monitor_server.py and return the ack message."""
    try:
        import websockets
    except ImportError as exc:
        raise RuntimeError("Install websockets or run with --no-send") from exc

    async with websockets.connect(url, open_timeout=timeout, close_timeout=timeout) as ws:
        ready_raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
        await ws.send(json.dumps(payload, ensure_ascii=False))
        ack_raw = await asyncio.wait_for(ws.recv(), timeout=timeout)

    try:
        return json.loads(ack_raw)
    except json.JSONDecodeError:
        return {"type": "raw", "ready": ready_raw, "ack": ack_raw}


def send_event(url: str, payload: dict[str, Any], timeout: float = 10.0) -> dict[str, Any]:
    return asyncio.run(send_event_ws(url, payload, timeout=timeout))
