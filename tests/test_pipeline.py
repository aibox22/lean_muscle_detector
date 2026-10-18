import base64

import cv2
import numpy as np
from fastapi.testclient import TestClient

from lean_muscle.analyze import Analysis, analyze_arrays
from lean_muscle.app import app
from lean_muscle.errors import Refuse
from lean_muscle.spec import KPT_NAMES


def _mask() -> np.ndarray:
    mask = np.zeros((320, 240), np.uint8)
    mask[40:300, 90:170] = 1
    mask[55:95, 70:190] = 1
    mask[120:210, 36:64] = 1
    mask[120:210, 176:204] = 1
    return mask


def _pose(width: int, height: int) -> dict:
    pixels = {
        "left_shoulder": (85, 75),
        "right_shoulder": (175, 75),
        "left_elbow": (15, 255),
        "right_elbow": (205, 255),
        "left_hip": (110, 260),
        "right_hip": (150, 260),
    }
    kpts = [[0.0, 0.0] for _ in KPT_NAMES]
    scores = [0.0 for _ in KPT_NAMES]
    for name, (x, y) in pixels.items():
        index = KPT_NAMES.index(name)
        kpts[index] = [x / width, y / height]
        scores[index] = 0.95
    return {
        "object": "img.pose.kpts",
        "kpts_labels": list(KPT_NAMES),
        "items": [{
            "bbox_xywh": [0.05, 0.05, 0.9, 0.9],
            "label": "person",
            "kpts_xy": kpts,
            "kpts_score": scores,
        }],
    }


def _segment(mask: np.ndarray) -> dict:
    ok, buffer = cv2.imencode(".png", mask)
    assert ok
    encoded = base64.b64encode(buffer.tobytes()).decode()
    return {
        "object": "img.segment.masks",
        "items": [{
            "bbox_xywh": [0.0, 0.0, 1.0, 1.0],
            "label": "person",
            "score": 0.99,
            "area": 0.2,
            "instance_id": 1,
        }],
        "mask": {
            "format": "png",
            "height": int(mask.shape[0]),
            "width": int(mask.shape[1]),
            "data": f"data:image/png;base64,{encoded}",
        },
    }


def test_pipeline_scores_a_synthetic_torso_without_the_gateway():
    mask = _mask()
    image = np.full((320, 240, 3), 180, np.uint8)
    result = analyze_arrays(
        image,
        _pose(240, 320),
        _segment(mask),
        sentence="肩腰 1.50，形状有了，腰还没跟上。",
    )
    scores = result.body["scores"]
    assert scores["shoulder"] == 98.0
    assert set(scores) == {"abs", "chest", "shoulder", "back", "arm", "total"}
    assert "1.50" in result.body["sentence"]
    assert result.body["measures"]["shoulder_waist_ratio"] == pytest_approx_ratio(result)
    assert result.body["measures"]["arms_visible"] is True
    assert result.body["measures"]["chest_waist_ratio"] is not None
    assert result.debug_png[:8] == b"\x89PNG\r\n\x1a\n"


def pytest_approx_ratio(result):
    ratio = result.body["measures"]["shoulder_waist_ratio"]
    assert abs(ratio - 1.5) < 0.08
    return ratio


def test_arms_outside_the_frame_are_still_scored():
    mask = _mask()
    mask[120:210, 36:64] = 0
    mask[120:210, 176:204] = 0
    pose = _pose(240, 320)
    pose["items"][0]["kpts_xy"][KPT_NAMES.index("left_elbow")] = [-0.35, 0.8]
    pose["items"][0]["kpts_xy"][KPT_NAMES.index("right_elbow")] = [1.45, 0.8]
    image = np.full((320, 240, 3), 180, np.uint8)
    result = analyze_arrays(
        image,
        pose,
        _segment(mask),
        sentence="肩腰比1.50，手臂都不漏，可能是太细了。",
    )
    assert result.body["scores"]["arm"] == 42.0
    assert result.body["measures"]["arms_visible"] is False
    assert result.body["measures"]["arm_scores"] == [42.0, 42.0]
    assert result.debug_png[:8] == b"\x89PNG\r\n\x1a\n"


def test_out_of_frame_joint_refuses_before_a_sentence():
    mask = _mask()
    pose = _pose(240, 320)
    pose["items"][0]["kpts_xy"][KPT_NAMES.index("left_hip")] = [-0.1, 0.8]
    image = np.full((320, 240, 3), 180, np.uint8)
    try:
        analyze_arrays(image, pose, _segment(mask))
    except Refuse as exc:
        assert exc.reason == "out_of_frame"
    else:
        raise AssertionError("expected a refusal")


def test_http_returns_scores_and_a_debug_image(monkeypatch):
    def fake(_data: bytes) -> Analysis:
        return Analysis(
            body={
                "scores": {
                    "abs": 40.0, "chest": 54.0, "shoulder": 98.0, "back": 48.0, "arm": 70.0, "total": 62.0,
                },
                "measures": {"shoulder_waist_ratio": 1.5, "arm_waist_ratio": 0.2, "abs_grooves": 0},
                "sentence": "肩腰 1.50，形状有了，腰还没跟上。",
                "llm_error": None,
            },
            debug_png=b"\x89PNG\r\n\x1a\n",
        )

    monkeypatch.setattr("lean_muscle.app.analyze_bytes", fake)
    client = TestClient(app)
    response = client.post("/analyze", files={"file": ("a.jpg", b"not-an-image", "image/jpeg")})
    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert body["calibrated"] is False
    assert body["sentence"].startswith("肩腰 1.50")
    assert body["debug_image_png_base64"]


def test_http_refusal_stays_a_422(monkeypatch):
    def fake(_data: bytes) -> Analysis:
        raise Refuse("out_of_frame", "左髋出画了")

    monkeypatch.setattr("lean_muscle.app.analyze_bytes", fake)
    client = TestClient(app)
    response = client.post("/analyze", files={"file": ("a.jpg", b"x", "image/jpeg")})
    assert response.status_code == 422
    assert response.json()["detail"]["reason"] == "out_of_frame"


def test_index_is_served():
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert "薄肌检测器" in response.text
