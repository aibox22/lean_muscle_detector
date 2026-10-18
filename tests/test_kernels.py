import numpy as np

from lean_muscle.kernels import abdomen_response, chest_muscle_lines, count_peaks


def test_count_peaks_keeps_separated_grooves():
    profile = np.zeros(100)
    profile[20] = 5
    profile[50] = 4
    profile[80] = 6
    assert count_peaks(profile) == 3


def test_count_peaks_merges_a_double_edge():
    profile = np.zeros(100)
    profile[40] = 5
    profile[42] = 4
    assert count_peaks(profile) == 1


def test_three_dark_lines_count_as_grooves():
    gray = np.full((160, 100), 210, np.uint8)
    mask = np.ones_like(gray, dtype=bool)
    for y in (85, 105, 125):
        gray[y - 1:y + 2, 20:80] = 30
    grooves, lr_ratio, grid, _response, lines = abdomen_response(
        gray, mask, shoulder_y=10, hip_y=150, center_x=50, waist_width=70,
    )
    assert grooves == 3
    assert len(lines) >= 3
    ys = sorted(line[1] for line in lines if line[1] == line[3])
    assert len(ys) == 3
    for found, expected in zip(ys, (85, 105, 125)):
        assert abs(found - expected) <= 4
    assert lr_ratio > 0.7
    assert len(grid) == 10 and len(grid[0]) == 10


def test_chest_split_and_lower_border_are_lines():
    gray = np.full((160, 120), 200, np.uint8)
    mask = np.ones_like(gray, dtype=bool)
    gray[20:70, 58:62] = 25
    gray[62:66, 25:95] = 25
    count, lines = chest_muscle_lines(
        gray, mask, shoulder_y=8, hip_y=150, center_x=60, chest_width=70,
    )
    assert count == len(lines)
    assert count >= 2
    vertical = [line for line in lines if line[0] == line[2]]
    horizontal = [line for line in lines if line[1] == line[3]]
    assert vertical and abs(vertical[0][0] - 60) <= 6
    assert horizontal and abs(horizontal[0][1] - 64) <= 8
