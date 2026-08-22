"""Phone-pairing QR endpoint.

The panel's "Podłącz telefon" button shows a QR that the on-site worker
scans to open /phone/capture.html on their phone. Pairing is stateless —
the phone just needs the capture URL pointing back at this server — so the
QR simply encodes that URL. Rendered as SVG (crisp at any size, no Pillow
dependency needed).
"""
from __future__ import annotations

from urllib.parse import urlparse

import qrcode
import qrcode.image.svg
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response

router = APIRouter()

# The capture URL (origin + /phone/capture.html?server=origin) is short;
# cap generously to reject anything that isn't a plausible URL.
MAX_TARGET_LEN = 512


@router.get("/pair-qr")
def pair_qr(target: str = Query(..., description="http(s) URL to encode")) -> Response:
    """Return an SVG QR code encoding ``target`` (the phone capture URL)."""
    if len(target) > MAX_TARGET_LEN:
        raise HTTPException(400, "target too long")
    parsed = urlparse(target)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise HTTPException(400, "target must be an http(s) URL")

    qr = qrcode.QRCode(
        version=None,                                    # auto-size to content
        error_correction=qrcode.constants.ERROR_CORRECT_M,
        box_size=10,
        border=2,
    )
    qr.add_data(target)
    qr.make(fit=True)
    img = qr.make_image(image_factory=qrcode.image.svg.SvgPathImage)

    return Response(
        content=img.to_string(),
        media_type="image/svg+xml",
        headers={"Cache-Control": "no-store"},
    )
