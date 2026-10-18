"""Local HTTP API. One image in, scores and a debug image out."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from .analyze import analyze_bytes
from .errors import GatewayError, MissingKey, Refuse
from .spec import MAX_UPLOAD_BYTES

WEB_DIR = Path(__file__).resolve().parent / "web"

app = FastAPI(title="薄肌检测器", version="0.1.0")


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.get("/")
def index():
    page = WEB_DIR / "index.html"
    if not page.is_file():
        raise HTTPException(status_code=404, detail="页面还没放上来")
    return FileResponse(page)


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)) -> dict:
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail={"ok": False, "reason": "too_large", "detail": "照片超过 15MB"},
        )
    try:
        return analyze_bytes(data).response()
    except Refuse as exc:
        raise HTTPException(
            status_code=422,
            detail={"ok": False, "reason": exc.reason, "detail": exc.detail},
        ) from exc
    except MissingKey as exc:
        raise HTTPException(
            status_code=500,
            detail={"ok": False, "reason": "missing_api_key", "detail": str(exc)},
        ) from exc
    except GatewayError as exc:
        raise HTTPException(
            status_code=502,
            detail={"ok": False, "reason": "gateway_error", "detail": str(exc)},
        ) from exc
