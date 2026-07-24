"""Phone-pairing QR endpoint — /api/pair-qr returns an SVG for a valid URL."""
import os
import sys

import pytest
from httpx import AsyncClient, ASGITransport

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.main import app


@pytest.mark.asyncio
async def test_pair_qr_returns_svg():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get(
            "/api/pair-qr",
            params={"target": "https://example.up.railway.app/phone/capture.html?server=https://example.up.railway.app"},
        )
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("image/svg+xml")
    assert b"<svg" in resp.content


@pytest.mark.asyncio
@pytest.mark.parametrize("target", ["not-a-url", "ftp://x/y", "javascript:alert(1)", ""])
async def test_pair_qr_rejects_non_http(target):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/pair-qr", params={"target": target})
    # empty target -> 422 (missing required), others -> 400 (bad scheme)
    assert resp.status_code in (400, 422)


@pytest.mark.asyncio
async def test_pair_qr_rejects_overlong_target():
    transport = ASGITransport(app=app)
    long_target = "https://x/" + "a" * 600
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/pair-qr", params={"target": long_target})
    assert resp.status_code == 400
