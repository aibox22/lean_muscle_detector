import pytest

from lean_muscle.score import (
    abs_score,
    arm_part_score,
    arm_score,
    back_score,
    chest_score,
    combine_arm_scores,
    hidden_arm_score,
    pack_scores,
    piecewise,
    shoulder_score,
)
from lean_muscle.spec import ARM_KNOTS, SHOULDER_KNOTS, WEIGHT_ABS, WEIGHT_ARM, WEIGHT_BACK, WEIGHT_CHEST, WEIGHT_SHOULDER


def test_piecewise_clamps_and_interpolates():
    assert piecewise(1.0, SHOULDER_KNOTS) == 60
    assert piecewise(1.05, SHOULDER_KNOTS) == 60
    assert piecewise(1.125, SHOULDER_KNOTS) == pytest.approx(70)
    assert piecewise(1.275, SHOULDER_KNOTS) == pytest.approx(86)
    assert piecewise(1.425, SHOULDER_KNOTS) == pytest.approx(95)
    assert piecewise(1.8, SHOULDER_KNOTS) == 98
    assert piecewise(0.21, ARM_KNOTS) == 80
    assert piecewise(0.27, ARM_KNOTS) == 91
    assert piecewise(0.34, ARM_KNOTS) == 83.5


def test_arm_asymmetry_penalty_uses_the_larger_band():
    assert arm_score(0.27, 0.0) == 91
    assert arm_score(0.27, 0.2) == 76
    assert arm_score(0.27, 0.3) == 66
    assert arm_score(0.10, 0.4) == 45


def test_abs_zero_grooves_skip_the_light_penalty():
    assert abs_score(0, 0.1) == 40
    assert abs_score(3, 0.9) == 88
    assert abs_score(3, 0.4) == 88 * 0.6
    assert abs_score(5, 1.0) == 95
    mid = abs_score(3, 0.6)
    assert mid == 88 * 0.8


def test_total_is_recomputed_from_rounded_parts():
    packed = pack_scores(70.44, 80.04, 86.04, 72.04, 83.46)
    assert packed["abs"] == 70.4
    assert packed["chest"] == 80.0
    assert packed["shoulder"] == 86.0
    assert packed["back"] == 72.0
    assert packed["arm"] == 83.5
    expected = (
        WEIGHT_ABS * 70.4
        + WEIGHT_CHEST * 80.0
        + WEIGHT_SHOULDER * 86.0
        + WEIGHT_BACK * 72.0
        + WEIGHT_ARM * 83.5
    )
    assert packed["total"] == round(expected, 1)


def test_shoulder_endpoints():
    assert shoulder_score(1.05) == 60
    assert shoulder_score(1.50) == 98


def test_chest_mixes_width_and_definition():
    assert chest_score(1.20, 2) == pytest.approx(0.7 * 80 + 0.3 * 80)
    assert chest_score(1.50, 0) == pytest.approx(0.7 * 97 + 0.3 * 40)
    assert back_score(1.02) == 48


def test_hidden_arm_still_scores():
    assert arm_part_score(None, 80) == 42
    assert combine_arm_scores(42, 42, both_measured=False, ratio=None, asymmetry=None) == 42
    assert combine_arm_scores(80, 42, both_measured=False, ratio=None, asymmetry=None) == 61
    assert hidden_arm_score() == 42
