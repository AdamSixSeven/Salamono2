from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_worker_marker_frontend_uses_png_preview_and_download():
    html = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
    javascript = (ROOT / "frontend" / "app.js").read_text(encoding="utf-8")
    stylesheet = (ROOT / "frontend" / "style.css").read_text(encoding="utf-8")

    assert 'id="workerQrPreview"' in html
    assert 'id="workerQrPreviewImg"' in html
    assert 'id="workerQrDownloadBtn"' in html
    assert 'id="workerQrOpenBtn"' in html
    assert "Generuj PNG" in html

    assert 'return "/api/worker-tag.png?"' in javascript
    assert "function generateWorkerTagPng()" in javascript
    assert 'workerQrOpenBtn.addEventListener("click", generateWorkerTagPng)' in javascript
    assert "showWorkerQrPreview(workerId)" in javascript
    assert "/api/worker-qr" not in javascript

    assert ".px-worker-qr-preview" in stylesheet
    assert ".px-worker-qr-image-wrap img" in stylesheet
    assert "image-rendering: pixelated" in stylesheet
