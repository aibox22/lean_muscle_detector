"""Debug image: contour, shoulder, waist, arm chords, scores, kernel grid."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .geometry import Joints, Measures

_FONT_CANDIDATES = (
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "/System/Library/Fonts/STHeiti Light.ttc",
    "/Library/Fonts/Arial Unicode.ttf",
)


def _font(size: int) -> ImageFont.ImageFont:
    for path in _FONT_CANDIDATES:
        if Path(path).is_file():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def clip_segment(
    p0: np.ndarray,
    p1: np.ndarray,
    width: int,
    height: int,
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    """Cohen–Sutherland clip. Coordinates may lie outside the photo."""
    x0, y0 = float(p0[0]), float(p0[1])
    x1, y1 = float(p1[0]), float(p1[1])
    xmax, ymax = float(width - 1), float(height - 1)
    left, right, bottom, top = 1, 2, 4, 8

    def code(x: float, y: float) -> int:
        bits = 0
        if x < 0:
            bits |= left
        elif x > xmax:
            bits |= right
        if y < 0:
            bits |= top
        elif y > ymax:
            bits |= bottom
        return bits

    c0, c1 = code(x0, y0), code(x1, y1)
    for _ in range(12):
        if c0 == 0 and c1 == 0:
            return (int(round(x0)), int(round(y0))), (int(round(x1)), int(round(y1)))
        if c0 & c1:
            return None
        bits = c0 or c1
        if bits & top:
            x = x0 + (x1 - x0) * (0.0 - y0) / (y1 - y0)
            y = 0.0
        elif bits & bottom:
            x = x0 + (x1 - x0) * (ymax - y0) / (y1 - y0)
            y = ymax
        elif bits & right:
            y = y0 + (y1 - y0) * (xmax - x0) / (x1 - x0)
            x = xmax
        else:
            y = y0 + (y1 - y0) * (0.0 - x0) / (x1 - x0)
            x = 0.0
        if bits == c0:
            x0, y0, c0 = x, y, code(x, y)
        else:
            x1, y1, c1 = x, y, code(x, y)
    return None


def arm_guides(
    joints: Joints,
    measures: Measures,
    width: int,
    height: int,
) -> list[tuple[str, tuple[tuple[int, int], tuple[int, int]]]]:
    """A line for every arm that has no thickness chord, clipped to the photo."""
    guides: list[tuple[str, tuple[tuple[int, int], tuple[int, int]]]] = []
    pairs = (
        ("左臂", joints.left_shoulder, joints.left_elbow, measures.left_arm, -1.0),
        ("右臂", joints.right_shoulder, joints.right_elbow, measures.right_arm, 1.0),
    )
    for label, shoulder, elbow, segment, outward in pairs:
        if segment is not None:
            continue
        if elbow is None:
            aim = shoulder + np.array([outward * 80.0, 50.0])
        else:
            aim = elbow
        clipped = clip_segment(shoulder, aim, width, height)
        if clipped is not None:
            guides.append((label, clipped))
    return guides


def _dashed(draw: ImageDraw.ImageDraw, p0: tuple[int, int], p1: tuple[int, int], fill: tuple[int, int, int]) -> None:
    start = np.array(p0, dtype=np.float64)
    end = np.array(p1, dtype=np.float64)
    span = end - start
    length = float(np.linalg.norm(span))
    if length < 1:
        return
    direction = span / length
    cursor = 0.0
    while cursor < length:
        nxt = min(length, cursor + 8.0)
        a = start + direction * cursor
        b = start + direction * nxt
        draw.line([(float(a[0]), float(a[1])), (float(b[0]), float(b[1]))], fill=fill, width=2)
        cursor = nxt + 6.0


def _paint_hidden_arms(
    photo: Image.Image,
    guides: list[tuple[str, tuple[tuple[int, int], tuple[int, int]]]],
    arm_scores: tuple[float, float],
) -> None:
    if not guides:
        return
    draw = ImageDraw.Draw(photo)
    font = _font(16)
    color = (160, 50, 180)
    score_for = {"左臂": arm_scores[0], "右臂": arm_scores[1]}
    for label, (p0, p1) in guides:
        _dashed(draw, p0, p1, color)
        score = score_for[label]
        text = f"{label} {score:.0f}"
        x = min(max(p1[0] + 4, 4), photo.width - 78)
        y = min(max(p1[1] - 18, 4), photo.height - 22)
        draw.text((x, y), text, fill=color, font=font)


def _draw_measures(
    image: np.ndarray,
    mask: np.ndarray,
    measures: Measures,
    abs_lines: list[tuple[int, int, int, int]],
    chest_lines: list[tuple[int, int, int, int]],
) -> np.ndarray:
    canvas = image.copy()
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(canvas, contours, -1, (80, 220, 80), 2, cv2.LINE_AA)
    shoulder = measures.shoulder
    waist = measures.waist
    cv2.line(canvas, (shoulder.x0, shoulder.y), (shoulder.x1, shoulder.y), (0, 220, 255), 2, cv2.LINE_AA)
    cv2.circle(canvas, (shoulder.x0, shoulder.y), 4, (255, 180, 0), -1, cv2.LINE_AA)
    cv2.circle(canvas, (shoulder.x1, shoulder.y), 4, (255, 180, 0), -1, cv2.LINE_AA)
    cv2.line(canvas, (waist.x0, waist.y), (waist.x1, waist.y), (40, 40, 220), 2, cv2.LINE_AA)
    if measures.chest is not None:
        cv2.line(
            canvas,
            (measures.chest.x0, measures.chest.y),
            (measures.chest.x1, measures.chest.y),
            (40, 140, 230),
            2,
            cv2.LINE_AA,
        )
    if measures.lat is not None:
        cv2.line(
            canvas,
            (measures.lat.x0, measures.lat.y),
            (measures.lat.x1, measures.lat.y),
            (40, 180, 160),
            2,
            cv2.LINE_AA,
        )
    for segment in (measures.left_arm, measures.right_arm):
        if segment is None:
            continue
        p0 = tuple(int(round(v)) for v in segment.p0)
        p1 = tuple(int(round(v)) for v in segment.p1)
        cv2.line(canvas, p0, p1, (180, 60, 200), 2, cv2.LINE_AA)
    for x0, y0, x1, y1 in abs_lines:
        cv2.line(canvas, (x0, y0), (x1, y1), (255, 80, 220), 2, cv2.LINE_AA)
    for x0, y0, x1, y1 in chest_lines:
        cv2.line(canvas, (x0, y0), (x1, y1), (40, 40, 230), 2, cv2.LINE_AA)
    return canvas


def _grid_image(grid: list[list[int]], cell: int = 28) -> np.ndarray:
    rows = len(grid) or 1
    cols = len(grid[0]) if grid else 1
    values = np.array(grid or [[0]], dtype=np.float32)
    heat = np.zeros((rows * cell, cols * cell), np.uint8)
    span = max(float(np.max(np.abs(values))), 1.0)
    for row in range(rows):
        for col in range(cols):
            # Map -span..span onto 0..255, zero at mid gray.
            level = int(np.clip(128 + values[row, col] / span * 127, 0, 255))
            heat[row * cell:(row + 1) * cell, col * cell:(col + 1) * cell] = level
    color = cv2.applyColorMap(heat, cv2.COLORMAP_VIRIDIS)
    for row in range(rows):
        for col in range(cols):
            text = str(int(values[row, col]))
            origin = (col * cell + 4, row * cell + cell - 8)
            cv2.putText(color, text, origin, cv2.FONT_HERSHEY_PLAIN, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
    return color


def _wrap(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, width: int) -> list[str]:
    lines: list[str] = []
    current = ""
    for char in text:
        trial = current + char
        if current and draw.textlength(trial, font=font) > width:
            lines.append(current)
            current = char
        else:
            current = trial
    if current:
        lines.append(current)
    return lines or [""]


def _paint_muscle_labels(
    photo: Image.Image,
    abs_lines: list[tuple[int, int, int, int]],
    chest_lines: list[tuple[int, int, int, int]],
) -> None:
    draw = ImageDraw.Draw(photo)
    font = _font(15)
    marks = ((abs_lines, "腹肌", (210, 40, 170)), (chest_lines, "胸肌", (210, 30, 30)))
    for lines, text, color in marks:
        if not lines:
            continue
        x0, y0, x1, y1 = lines[0]
        x = min(max(max(x0, x1) + 4, 4), photo.width - 40)
        y = min(max(min(y0, y1) - 16, 4), photo.height - 20)
        draw.text((x, y), text, fill=color, font=font)


def render_debug(
    image: np.ndarray,
    mask: np.ndarray,
    measures: Measures,
    joints: Joints,
    scores: dict,
    arm_scores: tuple[float, float],
    grid: list[list[int]],
    grooves: int,
    sentence: str | None,
    abs_lines: list[tuple[int, int, int, int]] | None = None,
    chest_lines: list[tuple[int, int, int, int]] | None = None,
) -> bytes:
    abs_lines = abs_lines or []
    chest_lines = chest_lines or []
    overlay = _draw_measures(image, mask, measures, abs_lines, chest_lines)
    photo = Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    _paint_muscle_labels(photo, abs_lines, chest_lines)
    _paint_hidden_arms(
        photo,
        arm_guides(joints, measures, image.shape[1], image.shape[0]),
        arm_scores,
    )
    panel_w = 320
    title = _font(28)
    body = _font(18)
    small = _font(15)
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    sentence_lines = _wrap(probe, sentence, small, panel_w - 32) if sentence else []

    grid_bgr = _grid_image(grid)
    grid_image = Image.fromarray(cv2.cvtColor(grid_bgr, cv2.COLOR_BGR2RGB))
    max_grid = panel_w - 32
    scale = min(1.0, max_grid / max(grid_image.width, 1))
    if scale != 1.0:
        grid_image = grid_image.resize(
            (max(1, int(grid_image.width * scale)), max(1, int(grid_image.height * scale))),
            Image.Resampling.NEAREST,
        )

    text_h = 42 + 26 * 5 + 8 + 26 + 24 + 22 * len(sentence_lines) + 12
    panel_h = text_h + grid_image.height + 16
    height = max(photo.height, panel_h)
    canvas = Image.new("RGB", (photo.width + panel_w, height), (248, 246, 242))
    canvas.paste(photo, (0, 0))
    draw = ImageDraw.Draw(canvas)
    x = photo.width + 16
    y = 16
    draw.text((x, y), f"薄肌 {scores['total']:.1f}", fill=(20, 20, 20), font=title)
    y += 42
    for label, value, color in (
        ("腹肌", scores["abs"], (37, 99, 235)),
        ("胸肌", scores["chest"], (180, 70, 40)),
        ("肩腰", scores["shoulder"], (234, 88, 12)),
        ("背肌", scores["back"], (14, 116, 144)),
        ("手臂", scores["arm"], (22, 163, 74)),
    ):
        draw.text((x, y), f"{label}  {value:.1f}", fill=color, font=body)
        y += 26
    y += 8
    draw.text((x, y), f"肩腰比 {measures.shoulder_waist_ratio:.2f}", fill=(60, 60, 60), font=body)
    y += 26
    if measures.arms_visible and measures.arm_waist_ratio is not None:
        arm_line = f"臂腰比 {measures.arm_waist_ratio:.2f}  沟 {grooves}"
    else:
        arm_line = f"左臂 {arm_scores[0]:.0f}  右臂 {arm_scores[1]:.0f}  沟 {grooves}"
    draw.text((x, y), arm_line, fill=(60, 60, 60), font=small)
    y += 24
    for line in sentence_lines:
        draw.text((x, y), line, fill=(20, 20, 20), font=small)
        y += 22
    y += 12
    canvas.paste(grid_image, (x, y))
    ok, encoded = cv2.imencode(".png", cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError("调试图编码失败")
    return encoded.tobytes()
