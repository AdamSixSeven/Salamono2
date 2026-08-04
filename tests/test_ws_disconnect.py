from types import SimpleNamespace

import pytest

from backend.routes.ws import websocket_live


class _RecordingManager:
    def __init__(self):
        self.connected = []
        self.disconnected = []

    async def connect(self, websocket, **options):
        self.connected.append((websocket, options))

    def disconnect(self, websocket):
        self.disconnected.append(websocket)


class _DisconnectingWebSocket:
    def __init__(self, manager):
        self.app = SimpleNamespace(state=SimpleNamespace(ws_manager=manager))
        self.query_params = {
            "camera_id": "demo_upload",
            "binary": "true",
        }
        self.receive_calls = 0

    async def receive(self):
        self.receive_calls += 1
        if self.receive_calls > 1:
            raise AssertionError("receive() called after websocket.disconnect")
        return {"type": "websocket.disconnect", "code": 1000}


@pytest.mark.asyncio
async def test_live_websocket_stops_receiving_after_disconnect_message():
    manager = _RecordingManager()
    websocket = _DisconnectingWebSocket(manager)

    await websocket_live(websocket)

    assert websocket.receive_calls == 1
    assert manager.connected == [(
        websocket,
        {
            "camera_id": "demo_upload",
            "include_frame": True,
            "binary_frames": True,
        },
    )]
    assert manager.disconnected == [websocket]
