"""Map raw proportions onto the uncalibrated 0–100 scores."""

from __future__ import annotations

from .spec import (
    ABS_GROOVE_SCORES,
    ABS_LR_FACTOR,
    ABS_LR_FULL,
    ABS_LR_HALF,
    ARM_ASYMMETRY_PENALTIES,
    ARM_HIDDEN_SCORE,
    ARM_KNOTS,
    BACK_KNOTS,
    CHEST_DEFINITION_SCORES,
    CHEST_KNOTS,
    SHOULDER_KNOTS,
    WEIGHT_ABS,
    WEIGHT_ARM,
    WEIGHT_BACK,
    WEIGHT_CHEST,
    WEIGHT_SHOULDER,
)


def piecewise(value: float, knots: tuple[tuple[float, float], ...]) -> float:
    """Linear between knots. Outside the first and last knot, clamp."""
    points = sorted(knots)
    if value <= points[0][0]:
        return points[0][1]
    if value >= points[-1][0]:
        return points[-1][1]
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if x0 <= value <= x1:
            span = x1 - x0
            if span == 0:
                return y1
            t = (value - x0) / span
            return y0 + t * (y1 - y0)
    return points[-1][1]


def shoulder_score(ratio: float) -> float:
    return piecewise(ratio, SHOULDER_KNOTS)


def chest_score(ratio: float, edges: int) -> float:
    width = piecewise(ratio, CHEST_KNOTS)
    count = max(0, min(int(edges), 3))
    definition = CHEST_DEFINITION_SCORES[count]
    return 0.7 * width + 0.3 * definition


def back_score(ratio: float) -> float:
    return piecewise(ratio, BACK_KNOTS)


def hidden_arm_score() -> float:
    return ARM_HIDDEN_SCORE


def arm_part_score(width: float | None, waist: float) -> float:
    """One arm. No width means it was not in frame, which is still a score."""
    if width is None or waist <= 1:
        return hidden_arm_score()
    return arm_score(width / waist, 0.0)


def combine_arm_scores(
    left: float,
    right: float,
    *,
    both_measured: bool,
    ratio: float | None,
    asymmetry: float | None,
) -> float:
    if both_measured and ratio is not None and asymmetry is not None:
        return arm_score(ratio, asymmetry)
    return (left + right) / 2.0


def arm_score(ratio: float, asymmetry: float) -> float:
    base = piecewise(ratio, ARM_KNOTS)
    penalty = 0.0
    for threshold, points in ARM_ASYMMETRY_PENALTIES:
        if asymmetry > threshold:
            penalty = points
            break
    return max(0.0, base - penalty)


def abs_score(grooves: int, lr_ratio: float) -> float:
    count = max(0, min(int(grooves), 4))
    base = ABS_GROOVE_SCORES[count]
    if count == 0:
        return base
    if lr_ratio >= ABS_LR_FULL:
        factor = 1.0
    elif lr_ratio <= ABS_LR_HALF:
        factor = ABS_LR_FACTOR
    else:
        span = ABS_LR_FULL - ABS_LR_HALF
        factor = ABS_LR_FACTOR + (lr_ratio - ABS_LR_HALF) / span * (1.0 - ABS_LR_FACTOR)
    return base * factor


def total_score(
    abs_value: float,
    chest_value: float,
    shoulder_value: float,
    back_value: float,
    arm_value: float,
) -> float:
    return (
        WEIGHT_ABS * abs_value
        + WEIGHT_CHEST * chest_value
        + WEIGHT_SHOULDER * shoulder_value
        + WEIGHT_BACK * back_value
        + WEIGHT_ARM * arm_value
    )


def round1(value: float) -> float:
    return round(float(value), 1)


def pack_scores(
    abs_value: float,
    chest_value: float,
    shoulder_value: float,
    back_value: float,
    arm_value: float,
) -> dict[str, float]:
    """Round the parts first, then the total, so the displayed weights add up."""
    parts = {
        "abs": round1(abs_value),
        "chest": round1(chest_value),
        "shoulder": round1(shoulder_value),
        "back": round1(back_value),
        "arm": round1(arm_value),
    }
    parts["total"] = round1(total_score(
        parts["abs"], parts["chest"], parts["shoulder"], parts["back"], parts["arm"],
    ))
    return parts
