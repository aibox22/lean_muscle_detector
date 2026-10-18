"""Disk cache for gateway payloads, keyed by the image bytes and the call."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


class Cache:
    def __init__(self, root: Path):
        self.root = root

    def _path(self, kind: str, digest: str) -> Path:
        return self.root / kind / f"{digest}.json"

    def get(self, kind: str, digest: str) -> dict | None:
        path = self._path(kind, digest)
        if not path.is_file():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def put(self, kind: str, digest: str, payload: dict) -> None:
        path = self._path(kind, digest)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")


def digest(image_jpeg: bytes, *parts: str) -> str:
    hasher = hashlib.sha256(image_jpeg)
    for part in parts:
        hasher.update(b"\0")
        hasher.update(part.encode("utf-8"))
    return hasher.hexdigest()
