import base64

import pytest

from backend.ws_manager import ConnectionManager


class FakeWebSocket:
    def __init__(self):
        self.accepted = False
        self.sent = []

    async def accept(self):
        self.accepted = True

    async def send_json(self, data):
        self.sent.append(("json", data))

    async def send_bytes(self, data):
        self.sent.append(("bytes", data))


@pytest.mark.asyncio
async def test_camera_subscription_filters_before_sending():
    manager = ConnectionManager()
    phone = FakeWebSocket()
    other = FakeWebSocket()
    await manager.connect(phone, camera_id="cam_phone_1")
    await manager.connect(other, camera_id="cam_phone_2")

    await manager.broadcast_frame({
        "camera_id": "cam_phone_1",
        "frame_jpeg_b64": "abc",
    })

    assert phone.accepted is True
    assert len(phone.sent) == 1
    assert other.sent == []


@pytest.mark.asyncio
async def test_metadata_only_subscription_never_receives_frame_payload():
    manager = ConnectionManager()
    client = FakeWebSocket()
    await manager.connect(
        client,
        camera_id="cam_phone_1",
        include_frame=False,
    )

    await manager.broadcast_frame({
        "camera_id": "cam_phone_1",
        "frame_jpeg_b64": base64.b64encode(b"jpeg").decode(),
    }, b"jpeg")

    assert client.sent == [(
        "json",
        {
            "camera_id": "cam_phone_1",
            "frame_jpeg_b64": "",
            "frame_transport": "metadata-only",
        },
    )]


@pytest.mark.asyncio
async def test_binary_subscription_gets_metadata_then_raw_jpeg():
    manager = ConnectionManager()
    client = FakeWebSocket()
    await manager.connect(
        client,
        camera_id="cam_phone_1",
        binary_frames=True,
    )

    await manager.broadcast_frame({
        "camera_id": "cam_phone_1",
        "frame_jpeg_b64": "unused",
        "frame_id": 7,
    }, b"\xff\xd8jpeg\xff\xd9")

    assert client.sent[0][0] == "json"
    assert client.sent[0][1]["frame_jpeg_b64"] == ""
    assert client.sent[0][1]["frame_transport"] == "binary-jpeg"
    assert client.sent[1] == ("bytes", b"\xff\xd8jpeg\xff\xd9")


@pytest.mark.asyncio
async def test_base64_is_created_only_for_legacy_json_frame_client():
    manager = ConnectionManager()
    legacy = FakeWebSocket()
    await manager.connect(legacy, camera_id="cam_phone_1")

    jpeg = b"\xff\xd8legacy\xff\xd9"
    await manager.broadcast_frame({
        "camera_id": "cam_phone_1",
        "frame_jpeg_b64": "",
    }, jpeg)

    assert legacy.sent == [(
        "json",
        {
            "camera_id": "cam_phone_1",
            "frame_jpeg_b64": base64.b64encode(jpeg).decode(),
            "frame_transport": "base64-jpeg",
        },
    )]
