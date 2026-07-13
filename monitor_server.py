"""
Local community monitoring server for roadshow fall alerts.

This is intentionally local/LAN only: no authentication, TLS, persistence, or
public deployment concerns. It receives detector events over WebSocket and
broadcasts them to staff dashboards.

Run:
    python monitor_server.py --host 127.0.0.1 --port 8765

Dashboard:
    http://127.0.0.1:8765/

Detector WebSocket:
    ws://127.0.0.1:8765/ws/events
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles


BASE_DIR = Path(__file__).resolve().parent
DASHBOARD_PATH = BASE_DIR / "web" / "monitor_dashboard.html"
CLIPS_DIR = BASE_DIR / "clips"

MAX_EVENTS = 200

CLIPS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Fall Detection Community Monitor")
dashboard_clients: set[WebSocket] = set()
events: list[dict[str, Any]] = []


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def normalize_event(payload: dict[str, Any]) -> dict[str, Any]:
    """Normalize detector/demo payloads into dashboard-safe event records."""
    event = dict(payload)
    event.setdefault("event_id", f"event_{uuid4().hex[:12]}")
    event.setdefault("timestamp", now_iso())
    event.setdefault("status", "suspected")
    event.setdefault("client_id", "unknown-client")
    event.setdefault("device_id", "unknown-device")
    event.setdefault("address", "local demo")

    classifier = event.get("classifier")
    if not isinstance(classifier, dict):
        classifier = {}
    classifier.setdefault("p_fall", event.pop("p_fall", None))
    classifier.setdefault("threshold_low", 0.30)
    classifier.setdefault("threshold_high", 0.40)
    event["classifier"] = classifier

    llm_review = event.get("llm_review")
    if not isinstance(llm_review, dict):
        llm_review = {
            "status": event.get("status"),
            "is_fall": event.get("status") == "fall",
            "confidence": None,
            "reason": "No LLM review attached.",
        }
    event["llm_review"] = llm_review

    clip_path = event.get("clip_path")
    if clip_path:
        event["clip_url"] = clip_path_to_url(str(clip_path))
    else:
        event.setdefault("clip_url", None)

    event.setdefault("handled", False)
    return event


def clip_path_to_url(clip_path: str) -> str | None:
    """Expose clips under /clips when the path points into the local clips dir."""
    normalized = clip_path.replace("\\", "/")
    if normalized.startswith("clips/"):
        return "/" + normalized

    path = Path(clip_path)
    try:
        rel = path.resolve().relative_to(CLIPS_DIR.resolve())
        return "/clips/" + rel.as_posix()
    except (OSError, ValueError):
        return None


async def broadcast(message: dict[str, Any]) -> None:
    stale: list[WebSocket] = []
    encoded = json.dumps(message, ensure_ascii=False)
    for ws in list(dashboard_clients):
        try:
            await ws.send_text(encoded)
        except Exception:
            stale.append(ws)
    for ws in stale:
        dashboard_clients.discard(ws)


async def add_event(payload: dict[str, Any]) -> dict[str, Any]:
    incoming = normalize_event(payload)
    existing = next((event for event in events if event.get("event_id") == incoming.get("event_id")), None)
    if existing is None:
        event = incoming
        events.insert(0, event)
        del events[MAX_EVENTS:]
    else:
        handled = existing.get("handled", False)
        handled_at = existing.get("handled_at")
        first_timestamp = existing.get("timestamp")
        updated_at = incoming.get("timestamp", now_iso())
        existing.update(incoming)
        if first_timestamp:
            existing["timestamp"] = first_timestamp
        existing["updated_at"] = updated_at
        existing["handled"] = handled
        if handled_at:
            existing["handled_at"] = handled_at
        event = existing
    await broadcast({"type": "event", "event": event})
    return event


@app.get("/", response_class=HTMLResponse)
async def dashboard() -> HTMLResponse:
    if not DASHBOARD_PATH.exists():
        return HTMLResponse("<h1>Dashboard file missing</h1>", status_code=500)
    return HTMLResponse(DASHBOARD_PATH.read_text(encoding="utf-8"))


@app.get("/api/events")
async def list_events() -> JSONResponse:
    return JSONResponse({"events": events})


@app.post("/api/events")
async def post_event(payload: dict[str, Any]) -> JSONResponse:
    event = await add_event(payload)
    return JSONResponse({"ok": True, "event": event})


@app.post("/api/events/{event_id}/handled")
async def mark_handled(event_id: str) -> JSONResponse:
    for event in events:
        if event.get("event_id") == event_id:
            event["handled"] = True
            event["handled_at"] = now_iso()
            await broadcast({"type": "handled", "event_id": event_id, "event": event})
            return JSONResponse({"ok": True, "event": event})
    return JSONResponse({"ok": False, "error": "event not found"}, status_code=404)


@app.get("/favicon.ico")
async def favicon() -> Response:
    return Response(status_code=204)


@app.websocket("/ws/dashboard")
async def dashboard_socket(ws: WebSocket) -> None:
    await ws.accept()
    dashboard_clients.add(ws)
    await ws.send_text(json.dumps({"type": "snapshot", "events": events}, ensure_ascii=False))
    try:
        while True:
            raw = await ws.receive_text()
            try:
                message = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if message.get("type") == "handled":
                event_id = str(message.get("event_id", ""))
                for event in events:
                    if event.get("event_id") == event_id:
                        event["handled"] = True
                        event["handled_at"] = now_iso()
                        await broadcast({"type": "handled", "event_id": event_id, "event": event})
                        break
    except WebSocketDisconnect:
        pass
    finally:
        dashboard_clients.discard(ws)


@app.websocket("/ws/events")
async def event_socket(ws: WebSocket) -> None:
    await ws.accept()
    await ws.send_text(json.dumps({"type": "ready", "message": "event socket connected"}))
    try:
        while True:
            raw = await ws.receive_text()
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError as exc:
                await ws.send_text(json.dumps({"type": "error", "error": f"invalid json: {exc}"}))
                continue
            event = await add_event(payload)
            await ws.send_text(json.dumps({"type": "ack", "event_id": event["event_id"]}, ensure_ascii=False))
    except WebSocketDisconnect:
        pass


app.mount("/clips", StaticFiles(directory=str(CLIPS_DIR), check_dir=False), name="clips")


def main() -> None:
    parser = argparse.ArgumentParser(description="Local fall-event monitoring server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("PYTHONUTF8", "1")
    if sys.platform.startswith("win"):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    uvicorn.run("monitor_server:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
