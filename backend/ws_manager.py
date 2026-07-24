"""Camera-aware WebSocket fan-out with optional binary JPEG transport."""
from __future__ import annotations

import asyncio
import base64
import binascii
from dataclasses import dataclass, field
from typing import Any

from fastapi import WebSocket


@dataclass
class _Connection:
    websocket: WebSocket
    camera_id: str | None
    include_frame: bool
    binary_frames: bool
    send_lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class ConnectionManager:
    """Publish only relevant frames and avoid Base64 for capable clients."""

    def __init__(self, send_timeout_seconds: float = 0.5):
        self._connections: list[_Connection] = []
        self.send_timeout_seconds = max(0.05, float(send_timeout_seconds))

    async def connect(
        self,
        ws: WebSocket,
        *,
        camera_id: str | None = None,
        include_frame: bool = True,
        binary_frames: bool = False,
    ):
        await ws.accept()
        self._connections.append(_Connection(
            websocket=ws,
            camera_id=(camera_id or "").strip() or None,
            include_frame=bool(include_frame),
            binary_frames=bool(binary_frames),
        ))

    def disconnect(self, ws: WebSocket):
        self._connections = [
            connection
            for connection in self._connections
            if connection.websocket is not ws
        ]

    @staticmethod
    def _camera_id(data: dict[str, Any]) -> str:
        return str(data.get("camera_id") or "cam_default")

    async def _send(
        self,
        connection: _Connection,
        data: dict[str, Any],
        jpeg_bytes: bytes | None,
    ) -> None:
        payload = data
        binary_payload = None
        if not connection.include_frame:
            payload = dict(data)
            payload["frame_jpeg_b64"] = ""
            payload["frame_transport"] = "metadata-only"
        elif connection.binary_frames:
            payload = dict(data)
            encoded = payload.get("frame_jpeg_b64")
            binary_payload = jpeg_bytes
            if binary_payload is None and encoded:
                try:
                    binary_payload = base64.b64decode(encoded, validate=True)
                except (binascii.Error, ValueError, TypeError):
                    binary_payload = None
            payload["frame_jpeg_b64"] = ""
            payload["frame_transport"] = (
                "binary-jpeg" if binary_payload else "metadata-only"
            )
        elif jpeg_bytes is not None and not data.get("frame_jpeg_b64"):
            # Compatibility is intentionally lazy: the current panel uses
            # binary JPEG, so normal operation never allocates Base64.  Only
            # an older JSON-only subscriber pays this conversion cost.
            payload = dict(data)
            payload["frame_jpeg_b64"] = base64.b64encode(jpeg_bytes).decode()
            payload["frame_transport"] = "base64-jpeg"

        async with connection.send_lock:
            await connection.websocket.send_json(payload)
            if binary_payload is not None:
                await connection.websocket.send_bytes(binary_payload)

    async def broadcast_frame(
        self,
        data: dict[str, Any],
        jpeg_bytes: bytes | None = None,
    ) -> None:
        camera_id = self._camera_id(data)
        targets = [
            connection
            for connection in list(self._connections)
            if connection.camera_id is None or connection.camera_id == camera_id
        ]
        if not targets:
            return

        async def bounded_send(connection: _Connection):
            await asyncio.wait_for(
                self._send(connection, data, jpeg_bytes),
                timeout=self.send_timeout_seconds,
            )

        results = await asyncio.gather(
            *(bounded_send(connection) for connection in targets),
            return_exceptions=True,
        )
        dead = [
            connection.websocket
            for connection, result in zip(targets, results)
            if isinstance(result, BaseException)
        ]
        for websocket in dead:
            self.disconnect(websocket)

    async def broadcast_json(self, data: dict):
        """Backward-compatible JSON-only publisher."""
        await self.broadcast_frame(data)

    @property
    def client_count(self) -> int:
        return len(self._connections)
