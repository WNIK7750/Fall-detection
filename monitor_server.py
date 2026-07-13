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
import base64
from contextlib import asynccontextmanager
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import cv2

# Uvicorn/asyncio can otherwise use the Proactor loop on Windows, which logs
# noisy ConnectionResetError traces when browsers abort video range requests.
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles


BASE_DIR = Path(__file__).resolve().parent
DASHBOARD_PATH = BASE_DIR / "web" / "monitor_dashboard.html"
CLIPS_DIR = BASE_DIR / "clips"

MAX_EVENTS = 200

CLIPS_DIR.mkdir(parents=True, exist_ok=True)

dashboard_clients: set[WebSocket] = set()
event_clients: set[WebSocket] = set()
events: list[dict[str, Any]] = []


async def close_websockets() -> None:
    clients = list(dashboard_clients | event_clients)
    dashboard_clients.clear()
    event_clients.clear()
    if not clients:
        return
    await asyncio.gather(
        *(ws.close(code=1001, reason="server shutdown") for ws in clients),
        return_exceptions=True,
    )


@asynccontextmanager
async def lifespan(_app: FastAPI):
    try:
        yield
    finally:
        await close_websockets()


app = FastAPI(title="Fall Detection Community Monitor", lifespan=lifespan)


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def parse_iso(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        return None


def apply_due_handled(event: dict[str, Any]) -> None:
    due_at = parse_iso(event.get("handle_due_at"))
    if not due_at or event.get("handled"):
        return
    if datetime.now(timezone.utc).astimezone() >= due_at:
        event["handled"] = True
        event["handled_at"] = due_at.isoformat(timespec="seconds")
        event.pop("handle_requested_at", None)
        event.pop("handle_due_at", None)


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
    classifier.setdefault("threshold_low", 0.40)
    classifier.setdefault("threshold_high", 1.0)
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
    raw_clip_url = clip_path_to_url(str(clip_path)) if clip_path else None
    event["_clip_url_raw"] = raw_clip_url
    event.setdefault("video_released", event.get("status") == "fall")
    event["clip_url"] = raw_clip_url if event.get("video_released") else None
    event["clip_stream_url"] = None

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
        handle_requested_at = existing.get("handle_requested_at")
        handle_due_at = existing.get("handle_due_at")
        video_released = existing.get("video_released", False)
        incoming_video_released = incoming.get("video_released", False)
        video_requested_at = existing.get("video_requested_at")
        video_release_at = existing.get("video_release_at")
        first_timestamp = existing.get("timestamp")
        updated_at = incoming.get("timestamp", now_iso())
        existing.update(incoming)
        if first_timestamp:
            existing["timestamp"] = first_timestamp
        existing["updated_at"] = updated_at
        existing["handled"] = handled
        if handle_requested_at:
            existing["handle_requested_at"] = handle_requested_at
        if handle_due_at:
            existing["handle_due_at"] = handle_due_at
        existing["video_released"] = bool(
            video_released
            or incoming_video_released
            or existing.get("status") == "fall"
        )
        if video_requested_at:
            existing["video_requested_at"] = video_requested_at
        if video_release_at:
            existing["video_release_at"] = video_release_at
        raw_clip_url = existing.get("_clip_url_raw")
        existing["clip_url"] = raw_clip_url if existing.get("video_released") else None
        existing["clip_stream_url"] = None
        if handled_at:
            existing["handled_at"] = handled_at
        event = existing
    apply_due_handled(event)
    await broadcast({"type": "event", "event": event})
    return event


@app.get("/", response_class=HTMLResponse)
async def dashboard() -> HTMLResponse:
    if not DASHBOARD_PATH.exists():
        return HTMLResponse("<h1>Dashboard file missing</h1>", status_code=500)
    return HTMLResponse(DASHBOARD_PATH.read_text(encoding="utf-8"))


@app.get("/api/events")
async def list_events() -> JSONResponse:
    for event in events:
        apply_due_handled(event)
    return JSONResponse({"events": events})


@app.post("/api/events")
async def post_event(payload: dict[str, Any]) -> JSONResponse:
    event = await add_event(payload)
    return JSONResponse({"ok": True, "event": event})


@app.post("/api/events/{event_id}/handled")
async def mark_handled(event_id: str) -> JSONResponse:
    for event in events:
        if event.get("event_id") == event_id:
            apply_due_handled(event)
            due_at = parse_iso(event.get("handle_due_at"))
            if due_at and not event.get("handled"):
                event.pop("handle_requested_at", None)
                event.pop("handle_due_at", None)
            elif not event.get("handled"):
                now = datetime.now(timezone.utc).astimezone()
                event["handle_requested_at"] = now.isoformat(timespec="seconds")
                event["handle_due_at"] = (now + timedelta(seconds=30)).isoformat(timespec="seconds")
            await broadcast({"type": "handled", "event_id": event_id, "event": event})
            return JSONResponse({"ok": True, "event": event})
    return JSONResponse({"ok": False, "error": "event not found"}, status_code=404)


@app.post("/api/events/{event_id}/request-video")
async def request_video(event_id: str) -> JSONResponse:
    for event in events:
        if event.get("event_id") == event_id:
            if not event.get("_clip_url_raw"):
                return JSONResponse({"ok": False, "error": "clip not available"}, status_code=404)
            now = datetime.now(timezone.utc).astimezone()
            release_at = now + timedelta(seconds=30)
            event["video_requested_at"] = now.isoformat(timespec="seconds")
            event["video_release_at"] = release_at.isoformat(timespec="seconds")
            await broadcast({"type": "event", "event": event})
            return JSONResponse({"ok": True, "event": event})
    return JSONResponse({"ok": False, "error": "event not found"}, status_code=404)


@app.post("/api/events/{event_id}/release-video")
async def release_video(event_id: str) -> JSONResponse:
    for event in events:
        if event.get("event_id") == event_id:
            raw_clip_url = event.get("_clip_url_raw")
            if not raw_clip_url:
                return JSONResponse({"ok": False, "error": "clip not available"}, status_code=404)
            event["video_released"] = True
            event["clip_url"] = raw_clip_url
            event["clip_stream_url"] = None
            await broadcast({"type": "event", "event": event})
            return JSONResponse({"ok": True, "event": event})
    return JSONResponse({"ok": False, "error": "event not found"}, status_code=404)


def event_clip_path(event_id: str) -> Path | None:
    for event in events:
        if event.get("event_id") != event_id:
            continue
        clip_path = event.get("clip_path")
        if not clip_path:
            return None
        path = Path(str(clip_path))
        if not path.is_absolute():
            path = BASE_DIR / path
        try:
            path.resolve().relative_to(CLIPS_DIR.resolve())
        except (OSError, ValueError):
            return None
        return path
    return None


def clip_meta_for_event(event: dict[str, Any], path: Path) -> dict[str, Any]:
    meta = event.get("clip_meta")
    if isinstance(meta, dict) and meta.get("frame_count"):
        return meta

    cap = cv2.VideoCapture(str(path))
    try:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    finally:
        cap.release()
    return {
        "frame_count": frame_count,
        "fps": fps,
        "start_frame": None,
        "end_frame": None,
        "start_timestamp_sec": None,
        "end_timestamp_sec": None,
        "duration_sec": round(frame_count / max(fps, 1e-6), 3) if frame_count else None,
    }


@app.get("/api/events/{event_id}/clip-meta")
async def clip_meta(event_id: str) -> JSONResponse:
    event = next((item for item in events if item.get("event_id") == event_id), None)
    if not event or not event.get("video_released"):
        return JSONResponse({"ok": False, "error": "video not released"}, status_code=403)
    path = event_clip_path(event_id)
    if path is None or not path.exists():
        return JSONResponse({"ok": False, "error": "clip not found"}, status_code=404)
    return JSONResponse({"ok": True, "meta": clip_meta_for_event(event, path)})


@app.get("/api/events/{event_id}/clip-frames")
async def clip_frames(event_id: str) -> JSONResponse:
    event = next((item for item in events if item.get("event_id") == event_id), None)
    if not event or not event.get("video_released"):
        return JSONResponse({"ok": False, "error": "video not released"}, status_code=403)
    path = event_clip_path(event_id)
    if path is None or not path.exists():
        return JSONResponse({"ok": False, "error": "clip not found"}, status_code=404)

    cap = cv2.VideoCapture(str(path))
    frames: list[str] = []
    try:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
            if ok:
                frames.append(base64.b64encode(encoded.tobytes()).decode("ascii"))
    finally:
        cap.release()
    if not frames:
        return JSONResponse({"ok": False, "error": "clip has no readable frames"}, status_code=404)
    return JSONResponse({
        "ok": True,
        "meta": clip_meta_for_event(event, path),
        "frames": frames,
        "mime": "image/jpeg",
    })


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
                        apply_due_handled(event)
                        if not event.get("handled"):
                            now = datetime.now(timezone.utc).astimezone()
                            event["handle_requested_at"] = now.isoformat(timespec="seconds")
                            event["handle_due_at"] = (now + timedelta(seconds=30)).isoformat(timespec="seconds")
                        await broadcast({"type": "handled", "event_id": event_id, "event": event})
                        break
    except WebSocketDisconnect:
        pass
    finally:
        dashboard_clients.discard(ws)


@app.websocket("/ws/events")
async def event_socket(ws: WebSocket) -> None:
    await ws.accept()
    event_clients.add(ws)
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
    finally:
        event_clients.discard(ws)


app.mount("/clips", StaticFiles(directory=str(CLIPS_DIR), check_dir=False), name="clips")


def main() -> None:
    parser = argparse.ArgumentParser(description="Local fall-event monitoring server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("PYTHONUTF8", "1")
    uvicorn_app = "monitor_server:app" if args.reload else app
    uvicorn.run(
        uvicorn_app,
        host=args.host,
        port=args.port,
        reload=args.reload,
        loop="asyncio",
        timeout_graceful_shutdown=2,
    )


if __name__ == "__main__":
    main()
