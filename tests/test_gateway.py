import json

import pytest

from lean_muscle.errors import GatewayError
from lean_muscle.gateway import unwrap


def test_unwrap_accepts_a_bare_image_payload_and_a_content_envelope():
    bare = {"object": "img.pose.kpts", "items": []}
    wrapped = {"model": "usyd-community/vitpose-plus-large", "content": bare}
    assert unwrap(bare, "img.pose.kpts")["object"] == "img.pose.kpts"
    assert unwrap(wrapped, "img.pose.kpts")["items"] == []
    assert unwrap(json.dumps(wrapped), "img.pose.kpts")["object"] == "img.pose.kpts"


def test_unwrap_rejects_a_video_pose_payload():
    payload = {"content": {"object": "vid.pose.kpts", "items": []}}
    with pytest.raises(GatewayError):
        unwrap(payload, "img.pose.kpts")
