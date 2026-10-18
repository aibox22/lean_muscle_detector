"""Shoulder, waist, and arm widths on a person mask. Pixels in, ratios out."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .errors import GatewayError, Refuse
from .spec import (
    ARM_STUCK_FRACTION,
    ELBOW_JOINTS,
    JOINT_LABELS,
    KPT_NAMES,
    MIN_KPT_SCORE,
    SECOND_PERSON_AREA_RATIO,
    TORSO_FRAME_MARGIN,
    TORSO_JOINTS,
)


@dataclass
class Joints:
    """Shoulders and hips are required. Elbows stay even when they leave the frame."""

    left_shoulder: np.ndarray
    right_shoulder: np.ndarray
    left_hip: np.ndarray
    right_hip: np.ndarray
    left_elbow: np.ndarray | None = None
    right_elbow: np.ndarray | None = None

    def by_name(self, name: str) -> np.ndarray:
        return getattr(self, name)


@dataclass
class Chord:
    x0: int
    x1: int
    y: int

    @property
    def width(self) -> float:
        return float(abs(self.x1 - self.x0) + 1)


@dataclass
class Segment:
    p0: np.ndarray
    p1: np.ndarray

    @property
    def width(self) -> float:
        return float(np.linalg.norm(self.p1 - self.p0))


@dataclass
class Measures:
    shoulder: Chord
    waist: Chord
    chest: Chord | None
    lat: Chord | None
    left_arm: Segment | None
    right_arm: Segment | None
    shoulder_width: float
    waist_width: float
    chest_width: float | None
    lat_width: float | None
    arm_widths: tuple[float, float] | None

    @property
    def shoulder_waist_ratio(self) -> float:
        return self.shoulder_width / self.waist_width

    @property
    def chest_waist_ratio(self) -> float | None:
        if not self.chest_width:
            return None
        return self.chest_width / self.waist_width

    @property
    def lat_waist_ratio(self) -> float | None:
        if not self.lat_width:
            return None
        return self.lat_width / self.waist_width

    @property
    def arms_visible(self) -> bool:
        return self.arm_widths is not None

    @property
    def arm_waist_ratio(self) -> float | None:
        if not self.arm_widths:
            return None
        return float(np.mean(self.arm_widths) / self.waist_width)

    @property
    def arm_asymmetry(self) -> float | None:
        if not self.arm_widths:
            return None
        left, right = self.arm_widths
        mean = (left + right) / 2.0
        if mean <= 1e-6:
            return 0.0
        return abs(left - right) / mean


def _box_area(box) -> float:
    return float(box[2]) * float(box[3])


def select_person(items: list[dict]) -> dict:
    if not items:
        raise Refuse("no_person", "画面里没有人")
    ranked = sorted(items, key=lambda item: _box_area(item.get("bbox_xywh") or [0, 0, 0, 0]), reverse=True)
    if len(ranked) >= 2:
        largest = _box_area(ranked[0].get("bbox_xywh") or [0, 0, 0, 0])
        second = _box_area(ranked[1].get("bbox_xywh") or [0, 0, 0, 0])
        if largest <= 0 or second > SECOND_PERSON_AREA_RATIO * largest:
            raise Refuse("multiple_people", "画面里有两个大小接近的人")
    return ranked[0]


def read_joints(item: dict, width: int, height: int, labels: list[str] | None) -> Joints:
    if labels is not None and list(labels) != KPT_NAMES:
        raise GatewayError(
            f"关节点顺序不是 COCO-17：{list(labels)}"
        )
    kpts = item.get("kpts_xy") or []
    scores = item.get("kpts_score") or []
    if len(kpts) != len(KPT_NAMES) or len(scores) != len(KPT_NAMES):
        raise GatewayError(
            f"期望 17 个关节点，得到坐标 {len(kpts)}、置信度 {len(scores)}"
        )

    index = {name: i for i, name in enumerate(KPT_NAMES)}

    def read_one(name: str, margin: float, optional: bool) -> np.ndarray | None:
        i = index[name]
        x, y = float(kpts[i][0]), float(kpts[i][1])
        score = float(scores[i])
        label = JOINT_LABELS[name]
        missing = score < MIN_KPT_SCORE or (x == 0.0 and y == 0.0)
        outside = not (-margin <= x <= 1.0 + margin and -margin <= y <= 1.0 + margin)
        if missing:
            if optional:
                return None
            raise Refuse("low_confidence", f"{label}没看清")
        # Elbows outside the photo still aim the arm mark and still get a score.
        if not optional and outside:
            raise Refuse("out_of_frame", f"{label}出画了")
        if not optional:
            x = min(1.0, max(0.0, x))
            y = min(1.0, max(0.0, y))
        return np.array([x * width, y * height], dtype=np.float64)

    torso = {name: read_one(name, TORSO_FRAME_MARGIN, optional=False) for name in TORSO_JOINTS}
    elbows = {name: read_one(name, 0.0, optional=True) for name in ELBOW_JOINTS}
    points = {**torso, **elbows}
    assert points["left_shoulder"] is not None and points["right_hip"] is not None

    span = (
        points["left_hip"][1] + points["right_hip"][1]
        - points["left_shoulder"][1] - points["right_shoulder"][1]
    )
    if span <= 20:
        raise Refuse("low_confidence", "躯干太短，肩和髋不可靠")
    return Joints(**points)


def _nearest_on_mask(mask: np.ndarray, x: float, y: float, radius: int = 8) -> tuple[int, int] | None:
    height, width = mask.shape
    xi = int(np.clip(round(x), 0, width - 1))
    yi = int(np.clip(round(y), 0, height - 1))
    if mask[yi, xi]:
        return xi, yi
    for r in range(1, radius + 1):
        y0, y1 = max(0, yi - r), min(height, yi + r + 1)
        x0, x1 = max(0, xi - r), min(width, xi + r + 1)
        patch = mask[y0:y1, x0:x1]
        if not patch.any():
            continue
        ys, xs = np.nonzero(patch)
        dist = (ys - (yi - y0)) ** 2 + (xs - (xi - x0)) ** 2
        k = int(np.argmin(dist))
        return x0 + int(xs[k]), y0 + int(ys[k])
    return None


def _walk(mask: np.ndarray, x: int, y: int, direction: int, max_steps: int) -> tuple[int, bool]:
    """Walk horizontally while the mask holds. The bool is True when the cap is still on mask."""
    cursor = x
    width = mask.shape[1]
    for _ in range(max_steps):
        nxt = cursor + direction
        if nxt < 0 or nxt >= width or not mask[y, nxt]:
            return cursor, False
        cursor = nxt
    nxt = cursor + direction
    capped = 0 <= nxt < width and bool(mask[y, nxt])
    return cursor, capped


def _central_chord(mask: np.ndarray, y: int, center_x: int) -> Chord | None:
    height, width = mask.shape
    if y < 0 or y >= height:
        return None
    if not mask[y].any():
        return None
    xs = np.flatnonzero(mask[y])
    # The run that contains the torso center. If the center is a hole, use the nearest run.
    if mask[y, center_x]:
        anchor = center_x
    else:
        anchor = int(xs[np.argmin(np.abs(xs - center_x))])
    left = anchor
    while left > 0 and mask[y, left - 1]:
        left -= 1
    right = anchor
    while right < width - 1 and mask[y, right + 1]:
        right += 1
    return Chord(left, right, y)


def _shoulder_chord(mask: np.ndarray, joints: Joints, torso: float) -> Chord:
    pair = sorted(
        [joints.left_shoulder, joints.right_shoulder],
        key=lambda p: float(p[0]),
    )
    image_left, image_right = pair
    span = float(image_right[0] - image_left[0])
    if span < 8:
        raise Refuse("low_confidence", "两肩太近")
    max_steps = max(4, int(round(0.35 * span)))
    y_start = int(round(min(image_left[1], image_right[1])))
    step = max(1, int(round(0.02 * torso)))
    uncapped: list[Chord] = []
    capped: list[Chord] = []
    for dy in range(0, max(1, int(round(0.12 * torso))) + 1, step):
        y = y_start + dy
        if y >= mask.shape[0]:
            break
        left = _nearest_on_mask(mask, image_left[0], y)
        right = _nearest_on_mask(mask, image_right[0], y)
        if left is None or right is None or left[1] != y or right[1] != y:
            if not mask[y].any():
                continue
            # Stay on this row. A snap onto another row would mix two widths.
            if not mask[y, int(np.clip(round(image_left[0]), 0, mask.shape[1] - 1))]:
                continue
            if not mask[y, int(np.clip(round(image_right[0]), 0, mask.shape[1] - 1))]:
                continue
            left = (int(np.clip(round(image_left[0]), 0, mask.shape[1] - 1)), y)
            right = (int(np.clip(round(image_right[0]), 0, mask.shape[1] - 1)), y)
        left_x, left_cap = _walk(mask, left[0], y, -1, max_steps)
        right_x, right_cap = _walk(mask, right[0], y, 1, max_steps)
        chord = Chord(min(left_x, right_x), max(left_x, right_x), y)
        if left_cap or right_cap:
            capped.append(chord)
        else:
            uncapped.append(chord)
    if uncapped:
        return max(uncapped, key=lambda chord: chord.width)
    if capped:
        return min(capped, key=lambda chord: chord.width)
    raise Refuse("no_mask", "肩高处没有人体轮廓")


def _waist_chord(mask: np.ndarray, joints: Joints, torso: float, shoulder_width: float) -> Chord:
    shoulder_y = float(min(joints.left_shoulder[1], joints.right_shoulder[1]))
    center = int(round((joints.left_shoulder[0] + joints.right_shoulder[0]) / 2.0))
    center = int(np.clip(center, 0, mask.shape[1] - 1))
    # Below the ribs and above the hip joints. Higher than this is the chest.
    y0 = int(round(shoulder_y + 0.55 * torso))
    y1 = int(round(shoulder_y + 0.88 * torso))
    y0 = max(0, min(y0, mask.shape[0] - 1))
    y1 = max(y0 + 1, min(y1, mask.shape[0]))
    best: Chord | None = None
    floor = 0.15 * shoulder_width
    for y in range(y0, y1):
        chord = _central_chord(mask, y, center)
        if chord is None or chord.width < floor:
            continue
        if best is None or chord.width < best.width:
            best = chord
    if best is None:
        raise Refuse("no_mask", "找不到腰部最窄处")
    return best


def _band_chord(
    mask: np.ndarray,
    joints: Joints,
    torso: float,
    start: float,
    end: float,
    *,
    widest: bool,
) -> Chord | None:
    """Widest or narrowest central chord between two fractions of the torso."""
    shoulder_y = float(min(joints.left_shoulder[1], joints.right_shoulder[1]))
    center = int(round((joints.left_shoulder[0] + joints.right_shoulder[0]) / 2.0))
    center = int(np.clip(center, 0, mask.shape[1] - 1))
    y0 = int(round(shoulder_y + start * torso))
    y1 = int(round(shoulder_y + end * torso))
    y0 = max(0, min(y0, mask.shape[0] - 1))
    y1 = max(y0 + 1, min(y1, mask.shape[0]))
    best: Chord | None = None
    for y in range(y0, y1):
        chord = _central_chord(mask, y, center)
        if chord is None or chord.width < 4:
            continue
        if best is None:
            best = chord
        elif widest and chord.width > best.width:
            best = chord
        elif not widest and chord.width < best.width:
            best = chord
    return best


def _normal(origin: np.ndarray, end: np.ndarray) -> np.ndarray | None:
    direction = end - origin
    length = float(np.linalg.norm(direction))
    if length < 1:
        return None
    return np.array([-direction[1], direction[0]], dtype=np.float64) / length


def _chord_along(mask: np.ndarray, origin: np.ndarray, normal: np.ndarray, half_cap: float) -> tuple[Segment, bool]:
    point = _nearest_on_mask(mask, float(origin[0]), float(origin[1]), radius=6)
    if point is None:
        raise Refuse("low_confidence", "上臂不在人体轮廓里")
    start = np.array(point, dtype=np.float64)

    def walk(sign: float) -> tuple[np.ndarray, bool]:
        cursor = start.copy()
        capped = False
        steps = int(max(1, round(half_cap)))
        for _ in range(steps):
            nxt = cursor + sign * normal
            xi, yi = int(round(nxt[0])), int(round(nxt[1]))
            if yi < 0 or xi < 0 or yi >= mask.shape[0] or xi >= mask.shape[1] or not mask[yi, xi]:
                return cursor, False
            cursor = nxt
        probe = cursor + sign * normal
        xi, yi = int(round(probe[0])), int(round(probe[1]))
        if 0 <= yi < mask.shape[0] and 0 <= xi < mask.shape[1] and mask[yi, xi]:
            capped = True
        return cursor, capped

    negative, cap_neg = walk(-1.0)
    positive, cap_pos = walk(1.0)
    return Segment(negative, positive), cap_neg or cap_pos


def _arm_segment(mask: np.ndarray, shoulder: np.ndarray, elbow: np.ndarray | None, shoulder_width: float) -> Segment | None:
    """Upper-arm thickness, or None when the arm is out of frame or stuck to the torso."""
    if elbow is None:
        return None
    normal = _normal(shoulder, elbow)
    if normal is None:
        return None
    half_cap = ARM_STUCK_FRACTION * shoulder_width / 2.0
    usable: list[tuple[float, Segment]] = []
    for t in np.linspace(0.40, 0.60, 9):
        point = shoulder + float(t) * (elbow - shoulder)
        snapped = _nearest_on_mask(mask, float(point[0]), float(point[1]), radius=4)
        if snapped is None:
            continue
        if abs(snapped[0] - point[0]) + abs(snapped[1] - point[1]) > 6:
            continue
        try:
            segment, capped = _chord_along(mask, np.array(snapped, dtype=np.float64), normal, half_cap)
        except Refuse:
            continue
        if capped or segment.width > ARM_STUCK_FRACTION * shoulder_width:
            continue
        usable.append((abs(float(t) - 0.5), segment))
    if not usable:
        return None
    usable.sort(key=lambda item: item[0])
    return usable[0][1]


def measure(mask: np.ndarray, joints: Joints) -> Measures:
    torso = float(
        (joints.left_hip[1] + joints.right_hip[1]) / 2.0
        - (joints.left_shoulder[1] + joints.right_shoulder[1]) / 2.0
    )
    shoulder = _shoulder_chord(mask, joints, torso)
    waist = _waist_chord(mask, joints, torso, shoulder.width)
    if waist.width <= 1:
        raise Refuse("no_mask", "腰宽不可用")
    chest = _band_chord(mask, joints, torso, 0.12, 0.38, widest=True)
    lat = _band_chord(mask, joints, torso, 0.38, 0.54, widest=True)
    left_arm = _arm_segment(mask, joints.left_shoulder, joints.left_elbow, shoulder.width)
    right_arm = _arm_segment(mask, joints.right_shoulder, joints.right_elbow, shoulder.width)
    arm_widths = None
    if left_arm is not None and right_arm is not None:
        arm_widths = (left_arm.width, right_arm.width)
    return Measures(
        shoulder=shoulder,
        waist=waist,
        chest=chest,
        lat=lat,
        left_arm=left_arm,
        right_arm=right_arm,
        shoulder_width=shoulder.width,
        waist_width=waist.width,
        chest_width=None if chest is None else chest.width,
        lat_width=None if lat is None else lat.width,
        arm_widths=arm_widths,
    )


def instance_mask(label_map: np.ndarray, items: list[dict], bbox_xywh: list[float]) -> np.ndarray:
    """The person instance that overlaps the pose box. Label pixels are instance ids."""
    if label_map.ndim == 3:
        label_map = label_map[:, :, 0]
    height, width = label_map.shape
    x, y, bw, bh = [float(v) for v in bbox_xywh]
    x0 = int(np.clip(x * width, 0, width - 1))
    y0 = int(np.clip(y * height, 0, height - 1))
    x1 = int(np.clip((x + bw) * width, x0 + 1, width))
    y1 = int(np.clip((y + bh) * height, y0 + 1, height))
    window = label_map[y0:y1, x0:x1]
    best_id = None
    best_overlap = 0
    known = {int(item["instance_id"]) for item in items if item.get("instance_id") is not None}
    candidates = known or {int(v) for v in np.unique(window) if int(v) != 0}
    for instance_id in candidates:
        overlap = int(np.count_nonzero(window == instance_id))
        if overlap > best_overlap:
            best_id = instance_id
            best_overlap = overlap
    if best_id is None or best_overlap < 10:
        raise Refuse("no_mask", "人体轮廓没有盖住这个人")
    return label_map == best_id
