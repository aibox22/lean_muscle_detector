"""One photo in, scores and a debug image out. The summary is optional."""

from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .cache import Cache
from .errors import Refuse
from .gateway import (
    decode_label_map,
    encode_jpeg,
    fetch_pose,
    fetch_segment,
    limit_long_side,
    load_api_key,
    make_client,
)
from .geometry import instance_mask, measure, read_joints, select_person
from .kernels import abdomen_response, chest_muscle_lines
from .render import render_debug
from .score import (
    abs_score,
    arm_part_score,
    back_score,
    chest_score,
    combine_arm_scores,
    pack_scores,
    shoulder_score,
)
from .summarize import write_sentence


def _pack_line(line: tuple[int, int, int, int]) -> dict[str, int]:
    x0, y0, x1, y1 = line
    return {"x0": int(x0), "y0": int(y0), "x1": int(x1), "y1": int(y1)}


@dataclass
class Analysis:
    body: dict
    debug_png: bytes

    def response(self) -> dict:
        payload = dict(self.body)
        payload["ok"] = True
        payload["calibrated"] = False
        payload["debug_image_png_base64"] = base64.b64encode(self.debug_png).decode("ascii")
        return payload


def decode_image(data: bytes) -> np.ndarray:
    if not data:
        raise Refuse("bad_image", "空文件")
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise Refuse("bad_image", "读不出这张图，请用 JPEG、PNG 或 WebP")
    return limit_long_side(image)


def analyze_arrays(
    image: np.ndarray,
    pose_payload: dict,
    segment_payload: dict,
    *,
    sentence: str | None = None,
    llm_error: str | None = None,
) -> Analysis:
    height, width = image.shape[:2]
    person = select_person(list(pose_payload.get("items") or []))
    joints = read_joints(person, width, height, pose_payload.get("kpts_labels"))
    label_map = decode_label_map(segment_payload.get("mask"), width, height)
    mask = instance_mask(label_map, list(segment_payload.get("items") or []), person.get("bbox_xywh") or [0, 0, 1, 1])
    measured = measure(mask, joints)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    center_x = float((joints.left_shoulder[0] + joints.right_shoulder[0]) / 2.0)
    shoulder_y = float((joints.left_shoulder[1] + joints.right_shoulder[1]) / 2.0)
    hip_y = float((joints.left_hip[1] + joints.right_hip[1]) / 2.0)
    grooves, lr_ratio, grid, _response, abs_lines = abdomen_response(
        gray, mask, shoulder_y, hip_y, center_x, measured.waist_width,
    )
    chest_width = measured.chest_width or measured.shoulder_width
    chest_edge_count, chest_lines = chest_muscle_lines(
        gray, mask, shoulder_y, hip_y, center_x, chest_width,
    )
    left_arm_value = arm_part_score(
        None if measured.left_arm is None else measured.left_arm.width,
        measured.waist_width,
    )
    right_arm_value = arm_part_score(
        None if measured.right_arm is None else measured.right_arm.width,
        measured.waist_width,
    )
    arm_value = combine_arm_scores(
        left_arm_value,
        right_arm_value,
        both_measured=measured.arms_visible,
        ratio=measured.arm_waist_ratio,
        asymmetry=measured.arm_asymmetry,
    )
    chest_value = 40.0 if measured.chest_waist_ratio is None else chest_score(measured.chest_waist_ratio, chest_edge_count)
    back_value = 40.0 if measured.lat_waist_ratio is None else back_score(measured.lat_waist_ratio)
    scores = pack_scores(
        abs_score(grooves, lr_ratio),
        chest_value,
        shoulder_score(measured.shoulder_waist_ratio),
        back_value,
        arm_value,
    )
    measures = {
        "shoulder_waist_ratio": round(measured.shoulder_waist_ratio, 4),
        "shoulder_width_px": round(measured.shoulder_width, 1),
        "waist_width_px": round(measured.waist_width, 1),
        "chest_waist_ratio": None if measured.chest_waist_ratio is None else round(measured.chest_waist_ratio, 4),
        "lat_waist_ratio": None if measured.lat_waist_ratio is None else round(measured.lat_waist_ratio, 4),
        "chest_edges": int(chest_edge_count),
        "arms_visible": measured.arms_visible,
        "arm_scores": [round(left_arm_value, 1), round(right_arm_value, 1)],
        "arm_width_px": None if measured.arm_widths is None else [round(v, 1) for v in measured.arm_widths],
        "arm_waist_ratio": None if measured.arm_waist_ratio is None else round(measured.arm_waist_ratio, 4),
        "arm_asymmetry": None if measured.arm_asymmetry is None else round(measured.arm_asymmetry, 4),
        "abs_grooves": int(grooves),
        "abs_lines": [_pack_line(line) for line in abs_lines],
        "chest_lines": [_pack_line(line) for line in chest_lines],
        "abs_lr_ratio": round(lr_ratio, 4),
        "abs_grid": grid,
    }
    debug = render_debug(
        image, mask, measured, joints, scores,
        (left_arm_value, right_arm_value),
        grid, grooves, sentence,
        abs_lines, chest_lines,
    )
    return Analysis(
        body={
            "scores": scores,
            "measures": measures,
            "sentence": sentence,
            "llm_error": llm_error,
        },
        debug_png=debug,
    )


def analyze_bytes(
    data: bytes,
    *,
    cache_dir: Path | None = None,
    summarize: bool = True,
    client=None,
) -> Analysis:
    image = decode_image(data)
    jpeg = encode_jpeg(image)
    if client is None:
        client = make_client(load_api_key())
    cache = Cache(cache_dir) if cache_dir is not None else Cache(Path.cwd() / ".cache" / "gateway")
    pose_payload = fetch_pose(client, jpeg, cache)
    segment_payload = fetch_segment(client, jpeg, cache)
    # Score before the sentence. A geometry refusal must not call the model.
    draft = analyze_arrays(image, pose_payload, segment_payload)
    sentence = None
    llm_error = None
    if summarize:
        sentence, llm_error = write_sentence(client, draft.body["measures"], draft.body["scores"])
        if sentence != draft.body["sentence"]:
            draft = analyze_arrays(
                image, pose_payload, segment_payload,
                sentence=sentence, llm_error=llm_error,
            )
    return draft
