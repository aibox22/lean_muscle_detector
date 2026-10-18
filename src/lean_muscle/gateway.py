"""One gateway client. Pose and segmentation return image payloads, not video."""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path

import cv2
import numpy as np
from openai import OpenAI

from .cache import Cache, digest
from .errors import GatewayError, MissingKey
from .spec import (
    GATEWAY_BASE_URL,
    MAX_IMAGE_SIDE,
    POSE_MODEL,
    SEGMENT_MODEL,
    SEGMENT_PROMPT,
)


def load_api_key(start: Path | None = None) -> str:
    preset = (os.getenv("VLMRUN_API_KEY") or "").strip()
    if preset:
        return preset
    roots = [Path.cwd()]
    if start is not None:
        roots.append(start)
    here = Path(__file__).resolve()
    roots.extend([here.parents[2], here.parents[1]])
    seen: set[Path] = set()
    for root in roots:
        if root in seen:
            continue
        seen.add(root)
        env_file = root / ".env"
        if not env_file.is_file():
            continue
        for line in env_file.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            name, value = stripped.split("=", 1)
            if name.strip() == "VLMRUN_API_KEY":
                key = value.strip().strip('"').strip("'")
                if key:
                    return key
    raise MissingKey(
        "没有 VLMRUN_API_KEY。在环境变量或项目 .env 里写上，"
        "密钥在 https://app.vlm.run 。"
    )


def make_client(api_key: str, timeout: float = 180.0) -> OpenAI:
    return OpenAI(base_url=GATEWAY_BASE_URL, api_key=api_key, timeout=timeout)


def limit_long_side(image: np.ndarray, max_side: int = MAX_IMAGE_SIDE) -> np.ndarray:
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= max_side:
        return image
    scale = max_side / longest
    return cv2.resize(
        image,
        (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
        interpolation=cv2.INTER_AREA,
    )


def encode_jpeg(image: np.ndarray, quality: int = 90) -> bytes:
    ok, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise GatewayError("照片转成 JPEG 失败")
    return buffer.tobytes()


def data_url(image_jpeg: bytes, mime: str = "image/jpeg") -> str:
    encoded = base64.b64encode(image_jpeg).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def unwrap(payload: dict | str, expected: str) -> dict:
    node = payload
    if isinstance(node, str):
        node = json.loads(node)
    if not isinstance(node, dict):
        raise GatewayError(f"网关返回的不是 JSON 对象：{str(payload)[:200]}")
    if node.get("object") == expected:
        return node
    content = node.get("content")
    if isinstance(content, str):
        content = json.loads(content)
    if isinstance(content, dict) and content.get("object") == expected:
        return content
    raise GatewayError(f"期望 {expected}，得到 {str(payload)[:300]}")


def _complete(client: OpenAI, *, model: str, image_jpeg: bytes, extra_body: dict) -> dict:
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": [{
                    "type": "image_url",
                    "image_url": {"url": data_url(image_jpeg)},
                }],
            }],
            response_format={"type": "json_object"},
            extra_body=extra_body,
        )
    except Exception as exc:  # noqa: BLE001 — the SDK raises a wide tree; the route maps it
        raise GatewayError(f"网关调用失败：{exc}") from exc
    content = response.choices[0].message.content
    if not content:
        raise GatewayError("网关返回了空内容")
    try:
        return json.loads(content)
    except json.JSONDecodeError as exc:
        raise GatewayError(f"网关返回的不是 JSON：{content[:200]}") from exc


def fetch_pose(client: OpenAI, image_jpeg: bytes, cache: Cache | None) -> dict:
    key = digest(image_jpeg, "pose", POSE_MODEL)
    if cache is not None:
        cached = cache.get("pose", key)
        if cached is not None:
            return cached
    payload = _complete(
        client,
        model=POSE_MODEL,
        image_jpeg=image_jpeg,
        extra_body={"method": "pose", "precision": 4},
    )
    parsed = unwrap(payload, "img.pose.kpts")
    if cache is not None:
        cache.put("pose", key, parsed)
    return parsed


def fetch_segment(client: OpenAI, image_jpeg: bytes, cache: Cache | None) -> dict:
    key = digest(image_jpeg, "segment", SEGMENT_MODEL, SEGMENT_PROMPT)
    if cache is not None:
        cached = cache.get("segment", key)
        if cached is not None:
            return cached
    payload = _complete(
        client,
        model=SEGMENT_MODEL,
        image_jpeg=image_jpeg,
        extra_body={
            "method": "segment",
            "method_params": {
                "prompt": SEGMENT_PROMPT,
                "mask_format": "png",
            },
        },
    )
    parsed = unwrap(payload, "img.segment.masks")
    if cache is not None:
        cache.put("segment", key, parsed)
    return parsed


def decode_label_map(mask: dict | None, width: int, height: int) -> np.ndarray:
    if not mask or not mask.get("data"):
        raise GatewayError("分割结果没有标签图")
    raw = mask["data"]
    if "," in raw and raw.strip().startswith("data:"):
        raw = raw.split(",", 1)[1]
    try:
        decoded = base64.b64decode(raw)
    except Exception as exc:  # noqa: BLE001
        raise GatewayError("分割标签图不是合法的 base64") from exc
    image = cv2.imdecode(np.frombuffer(decoded, np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise GatewayError("分割标签图解不开")
    if image.ndim == 3:
        image = image[:, :, 0]
    if image.shape[0] != height or image.shape[1] != width:
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_NEAREST)
    return image
