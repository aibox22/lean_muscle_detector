import numpy as np
import pytest

from lean_muscle.errors import Refuse
from lean_muscle.geometry import Joints, measure, read_joints, select_person
from lean_muscle.render import arm_guides
from lean_muscle.spec import KPT_NAMES


def _person_mask():
    mask = np.zeros((320, 240), dtype=bool)
    mask[40:300, 90:170] = True
    mask[55:95, 70:190] = True
    mask[120:210, 36:64] = True
    mask[120:210, 176:204] = True
    return mask


def _joints():
    return Joints(
        left_shoulder=np.array([85.0, 75.0]),
        right_shoulder=np.array([175.0, 75.0]),
        left_elbow=np.array([15.0, 255.0]),
        right_elbow=np.array([205.0, 255.0]),
        left_hip=np.array([110.0, 260.0]),
        right_hip=np.array([150.0, 260.0]),
    )


def test_separated_arms_measure_shoulder_over_waist():
    measured = measure(_person_mask(), _joints())
    assert measured.shoulder_waist_ratio == pytest.approx(120 / 80, abs=0.08)
    assert measured.arm_widths[0] == pytest.approx(28, abs=4)
    assert measured.arm_widths[1] == pytest.approx(28, abs=4)
    assert measured.arm_asymmetry < 0.15


def test_merged_arms_are_scored_as_hidden():
    mask = np.zeros((320, 240), dtype=bool)
    mask[40:300, 20:220] = True
    measured = measure(mask, _joints())
    assert measured.arms_visible is False
    assert measured.arm_widths is None
    assert measured.waist_width > 1


def test_missing_arms_still_get_a_guide():
    mask = np.zeros((320, 240), dtype=bool)
    mask[40:300, 90:170] = True
    mask[55:95, 70:190] = True
    measured = measure(mask, _joints())
    assert measured.arms_visible is False
    guides = arm_guides(_joints(), measured, 240, 320)
    assert [label for label, _line in guides] == ["左臂", "右臂"]
    for _label, (p0, p1) in guides:
        for x, y in (p0, p1):
            assert 0 <= x < 240
            assert 0 <= y < 320
    items = [
        {"bbox_xywh": [0.1, 0.1, 0.4, 0.8]},
        {"bbox_xywh": [0.5, 0.1, 0.35, 0.7]},
    ]
    with pytest.raises(Refuse) as caught:
        select_person(items)
    assert caught.value.reason == "multiple_people"


def test_a_small_bystander_is_ignored():
    items = [
        {"bbox_xywh": [0.2, 0.1, 0.5, 0.8], "label": "big"},
        {"bbox_xywh": [0.0, 0.0, 0.1, 0.2], "label": "small"},
    ]
    assert select_person(items)["label"] == "big"


def _pose_item(points: dict[str, tuple[float, float]]) -> dict:
    kpts = [[0.0, 0.0] for _ in KPT_NAMES]
    scores = [0.0 for _ in KPT_NAMES]
    for name, (x, y) in points.items():
        index = KPT_NAMES.index(name)
        kpts[index] = [x, y]
        scores[index] = 0.95
    return {"kpts_xy": kpts, "kpts_score": scores}


def test_elbow_just_past_the_edge_is_kept():
    item = _pose_item({
        "left_shoulder": (0.40, 0.30),
        "right_shoulder": (0.60, 0.30),
        "left_elbow": (0.20, 0.50),
        "right_elbow": (1.12, 0.48),
        "left_hip": (0.45, 0.70),
        "right_hip": (0.55, 0.70),
    })
    joints = read_joints(item, 200, 400, list(KPT_NAMES))
    assert joints.right_elbow[0] == pytest.approx(1.12 * 200)


def test_elbow_far_outside_is_kept_for_the_mark():
    item = _pose_item({
        "left_shoulder": (0.40, 0.30),
        "right_shoulder": (0.60, 0.30),
        "left_elbow": (0.20, 0.50),
        "right_elbow": (1.45, 0.48),
        "left_hip": (0.45, 0.70),
        "right_hip": (0.55, 0.70),
    })
    joints = read_joints(item, 200, 400, list(KPT_NAMES))
    assert joints.right_elbow[0] == pytest.approx(1.45 * 200)
    assert joints.left_elbow is not None


def test_labels_constant_matches_coco_17():
    assert len(KPT_NAMES) == 17
    assert KPT_NAMES[5:13] == [
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
    ]
