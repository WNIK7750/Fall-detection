"""
Send a sample fall event to the local monitor server over WebSocket.

This uses only the Python standard library so it can run even when no WebSocket
client package is installed.

Run:
    python scripts/demo/demo_event_sender.py --clip clips/s01_1050_90.mp4
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import socket
import ssl
import struct
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse
from uuid import uuid4


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def build_sample_event(args: argparse.Namespace) -> dict:
    status = args.status
    llm_status = args.llm_status or status
    is_fall = llm_status == "fall"
    return {
        "event_id": args.event_id or f"demo_{uuid4().hex[:12]}",
        "client_id": args.client_id,
        "device_id": args.device_id,
        "address": args.address,
        "timestamp": now_iso(),
        "scenario": args.scenario,
        "frame_index": args.frame_index,
        "status": status,
        "classifier": {
            "p_fall": args.p_fall,
            "threshold_low": 0.30,
            "threshold_high": 0.40,
        },
        "llm_review": {
            "status": llm_status,
            "is_fall": is_fall,
            "confidence": args.llm_confidence,
            "reason": args.reason,
        },
        "clip_path": normalize_clip_path(args.clip),
    }


def normalize_clip_path(path: str | None) -> str | None:
    if not path:
        return None
    p = Path(path)
    try:
        return p.relative_to(Path.cwd()).as_posix()
    except ValueError:
        return str(p).replace("\\", "/")


def websocket_send_text(url: str, text: str, timeout: float = 10.0) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in {"ws", "wss"}:
        raise ValueError("URL must start with ws:// or wss://")

    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or (443 if parsed.scheme == "wss" else 80)
    path = parsed.path or "/"
    if parsed.query:
        path += "?" + parsed.query

    raw_sock = socket.create_connection((host, port), timeout=timeout)
    sock: socket.socket | ssl.SSLSocket
    if parsed.scheme == "wss":
        sock = ssl.create_default_context().wrap_socket(raw_sock, server_hostname=host)
    else:
        sock = raw_sock
    sock.settimeout(timeout)

    key = base64.b64encode(os.urandom(16)).decode("ascii")
    host_header = f"{host}:{port}"
    request = (
        f"GET {path} HTTP/1.1\r\n"
        f"Host: {host_header}\r\n"
        "Upgrade: websocket\r\n"
        "Connection: Upgrade\r\n"
        f"Sec-WebSocket-Key: {key}\r\n"
        "Sec-WebSocket-Version: 13\r\n"
        "\r\n"
    )
    sock.sendall(request.encode("ascii"))
    response = recv_until(sock, b"\r\n\r\n")
    if b" 101 " not in response.split(b"\r\n", 1)[0]:
        raise RuntimeError(f"WebSocket handshake failed: {response[:200]!r}")

    accept_expected = base64.b64encode(
        hashlib.sha1((key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode("ascii")).digest()
    ).decode("ascii")
    if accept_expected.encode("ascii") not in response:
        raise RuntimeError("WebSocket handshake returned an invalid accept key")

    ready = websocket_recv_text(sock)
    websocket_send_frame(sock, text.encode("utf-8"))
    ack = websocket_recv_text(sock)
    websocket_close(sock)
    return ack or ready


def recv_until(sock: socket.socket | ssl.SSLSocket, marker: bytes) -> bytes:
    data = bytearray()
    while marker not in data:
        chunk = sock.recv(4096)
        if not chunk:
            break
        data.extend(chunk)
    return bytes(data)


def websocket_send_frame(sock: socket.socket | ssl.SSLSocket, payload: bytes) -> None:
    mask = os.urandom(4)
    header = bytearray([0x81])
    length = len(payload)
    if length < 126:
        header.append(0x80 | length)
    elif length <= 0xFFFF:
        header.append(0x80 | 126)
        header.extend(struct.pack("!H", length))
    else:
        header.append(0x80 | 127)
        header.extend(struct.pack("!Q", length))
    masked = bytes(byte ^ mask[i % 4] for i, byte in enumerate(payload))
    sock.sendall(bytes(header) + mask + masked)


def websocket_recv_text(sock: socket.socket | ssl.SSLSocket) -> str:
    first = sock.recv(2)
    if len(first) < 2:
        return ""
    opcode = first[0] & 0x0F
    length = first[1] & 0x7F
    if length == 126:
        length = struct.unpack("!H", recv_exact(sock, 2))[0]
    elif length == 127:
        length = struct.unpack("!Q", recv_exact(sock, 8))[0]
    masked = bool(first[1] & 0x80)
    mask = recv_exact(sock, 4) if masked else b""
    payload = recv_exact(sock, length)
    if masked:
        payload = bytes(byte ^ mask[i % 4] for i, byte in enumerate(payload))
    if opcode == 0x8:
        return ""
    return payload.decode("utf-8", errors="replace")


def recv_exact(sock: socket.socket | ssl.SSLSocket, length: int) -> bytes:
    data = bytearray()
    while len(data) < length:
        chunk = sock.recv(length - len(data))
        if not chunk:
            raise RuntimeError("Socket closed while receiving data")
        data.extend(chunk)
    return bytes(data)


def websocket_close(sock: socket.socket | ssl.SSLSocket) -> None:
    try:
        sock.sendall(b"\x88\x80" + os.urandom(4))
    finally:
        sock.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Send a demo fall event over WebSocket")
    parser.add_argument("--url", default="ws://127.0.0.1:8765/ws/events")
    parser.add_argument("--client-id", default="demo-client-01")
    parser.add_argument("--device-id", default="mcfd-camset-1-4-7")
    parser.add_argument("--address", default="Demo Community / Building A / Room 101")
    parser.add_argument("--scenario", type=int, default=1)
    parser.add_argument("--frame-index", type=int, default=1050)
    parser.add_argument("--status", choices=["suspected", "fall"], default="fall")
    parser.add_argument("--p-fall", type=float, default=0.421)
    parser.add_argument("--llm-status", choices=["normal", "suspected", "fall"], default=None)
    parser.add_argument("--llm-confidence", type=float, default=0.95)
    parser.add_argument("--reason", default="Demo event: person is lying horizontally on the floor mat.")
    parser.add_argument("--clip", default="clips/s01_1050_90.mp4")
    parser.add_argument("--event-id", default=None)
    args = parser.parse_args()

    event = build_sample_event(args)
    ack = websocket_send_text(args.url, json.dumps(event, ensure_ascii=False))
    print(ack)


if __name__ == "__main__":
    main()
