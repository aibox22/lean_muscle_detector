"""Horizontal-kernel response over the abdomen, and a coarse signed grid."""

from __future__ import annotations

import cv2
import numpy as np

# Light above, dark below: a tendinous groove reads as a local horizontal edge.
_KERNEL = np.array(
    [[1, 2, 1],
     [0, 0, 0],
     [-1, -2, -1]],
    dtype=np.float32,
)


def peak_indices(profile: np.ndarray, min_gap: int | None = None) -> list[int]:
    """Local maxima far enough apart to be separate lines, not one edge twice."""
    if profile.size < 5:
        return []
    magnitude = np.abs(profile.astype(np.float64))
    window = max(3, int(profile.size / 20) | 1)
    kernel = np.ones(window, dtype=np.float64) / window
    smooth = np.convolve(magnitude, kernel, mode="same")
    peak = float(smooth.max()) if smooth.size else 0.0
    if peak < 1e-3:
        return []
    threshold = 0.45 * peak
    min_dist = max(2, int(0.08 * len(smooth))) if min_gap is None else max(2, int(min_gap))
    found: list[int] = []
    for index in range(1, len(smooth) - 1):
        if smooth[index] < threshold:
            continue
        if smooth[index] < smooth[index - 1] or smooth[index] < smooth[index + 1]:
            continue
        if found and index - found[-1] < min_dist:
            if smooth[index] > smooth[found[-1]]:
                found[-1] = index
            continue
        found.append(index)
    return found


def count_peaks(profile: np.ndarray) -> int:
    return len(peak_indices(profile))


def _central_span(row: np.ndarray, anchor: int) -> tuple[int, int] | None:
    if row.size == 0 or not row.any():
        return None
    anchor = int(np.clip(anchor, 0, row.size - 1))
    if not row[anchor]:
        xs = np.flatnonzero(row)
        anchor = int(xs[np.argmin(np.abs(xs - anchor))])
    left = anchor
    while left > 0 and row[left - 1]:
        left -= 1
    right = anchor
    while right < row.size - 1 and row[right + 1]:
        right += 1
    if right - left < 6:
        return None
    return left, right


Line = tuple[int, int, int, int]


def block_means(response: np.ndarray, rows: int = 10, cols: int = 10) -> np.ndarray:
    height, width = response.shape
    grid = np.zeros((rows, cols), dtype=np.float64)
    for row in range(rows):
        y0 = int(round(row * height / rows))
        y1 = int(round((row + 1) * height / rows))
        for col in range(cols):
            x0 = int(round(col * width / cols))
            x1 = int(round((col + 1) * width / cols))
            cell = response[y0:y1, x0:x1]
            grid[row, col] = float(cell.mean()) if cell.size else 0.0
    return grid


def scale_grid(grid: np.ndarray) -> list[list[int]]:
    """Signed integers, 95th percentile of |response| mapped to 8."""
    flat = np.abs(grid).ravel()
    scale = float(np.percentile(flat, 95)) if flat.size else 1.0
    if scale < 1e-6:
        scale = 1.0
    shown = np.clip(np.rint(grid / scale * 8.0), -8, 8).astype(int)
    return shown.tolist()


def _horizontal_lines(
    mask: np.ndarray,
    indices: list[int],
    y0: int,
    x0: int,
    x1: int,
    center_x: float,
) -> list[Line]:
    lines: list[Line] = []
    for index in indices:
        y = y0 + index
        if y < 0 or y >= mask.shape[0]:
            continue
        span = _central_span(mask[y, x0:x1], int(round(center_x)) - x0)
        if span is None:
            continue
        left, right = span
        lines.append((x0 + left, y, x0 + right, y))
    return lines


def _vertical_lines(
    mask: np.ndarray,
    indices: list[int],
    x0: int,
    y0: int,
    y1: int,
    center_y: float,
) -> list[Line]:
    lines: list[Line] = []
    for index in indices:
        x = x0 + index
        if x < 0 or x >= mask.shape[1]:
            continue
        span = _central_span(mask[y0:y1, x], int(round(center_y)) - y0)
        if span is None:
            continue
        top, bottom = span
        lines.append((x, y0 + top, x, y0 + bottom))
    return lines


def abdomen_response(
    gray: np.ndarray,
    mask: np.ndarray,
    shoulder_y: float,
    hip_y: float,
    center_x: float,
    waist_width: float,
) -> tuple[int, float, list[list[int]], np.ndarray, list[Line]]:
    """Groove count, left/right ratio, grid, response, and the groove lines."""
    empty: list[Line] = []
    torso = hip_y - shoulder_y
    if torso <= 1:
        return 0, 1.0, scale_grid(np.zeros((10, 10))), np.zeros((1, 1), np.float32), empty
    y0 = int(np.clip(round(shoulder_y + 0.45 * torso), 0, gray.shape[0] - 2))
    y1 = int(np.clip(round(shoulder_y + 0.95 * torso), y0 + 2, gray.shape[0]))
    half = max(4.0, 0.45 * waist_width)
    x0 = int(np.clip(round(center_x - half), 0, gray.shape[1] - 2))
    x1 = int(np.clip(round(center_x + half), x0 + 2, gray.shape[1]))

    roi = gray[y0:y1, x0:x1].astype(np.float32)
    roi_mask = mask[y0:y1, x0:x1]
    if roi.size == 0 or int(roi_mask.sum()) < 20:
        return 0, 1.0, scale_grid(np.zeros((10, 10))), np.zeros((1, 1), np.float32), empty

    local = roi - cv2.GaussianBlur(roi, (0, 0), 11)
    response = cv2.filter2D(local, cv2.CV_32F, _KERNEL)
    response = response.copy()
    response[~roi_mask] = 0.0

    profile = response.mean(axis=1)
    gap = max(10, int(round(0.08 * torso)))
    indices = peak_indices(profile, min_gap=gap)
    mid = response.shape[1] // 2
    left = float(np.abs(response[:, :mid]).mean()) if mid else 0.0
    right = float(np.abs(response[:, mid:]).mean()) if response.shape[1] - mid else 0.0
    larger = max(left, right)
    lr_ratio = (min(left, right) / larger) if larger > 1e-6 else 1.0
    grid = scale_grid(block_means(response))
    lines = _horizontal_lines(mask, indices, y0, x0, x1, center_x)
    # The linea alba is the vertical split, not another groove.
    vertical = cv2.filter2D(local, cv2.CV_32F, _VERTICAL)
    vertical = vertical.copy()
    vertical[~roi_mask] = 0.0
    center_peaks = [
        index for index in peak_indices(np.abs(vertical).mean(axis=0), min_gap=gap)
        if abs((x0 + index) - center_x) <= 0.18 * (x1 - x0)
    ]
    if center_peaks:
        nearest = min(center_peaks, key=lambda index: abs((x0 + index) - center_x))
        lines.extend(_vertical_lines(mask, [nearest], x0, y0, y1, (y0 + y1) / 2.0))
    return len(indices), lr_ratio, grid, response, lines


_VERTICAL = np.array(
    [[1, 0, -1],
     [2, 0, -2],
     [1, 0, -1]],
    dtype=np.float32,
)


def chest_muscle_lines(
    gray: np.ndarray,
    mask: np.ndarray,
    shoulder_y: float,
    hip_y: float,
    center_x: float,
    chest_width: float,
) -> tuple[int, list[Line]]:
    """Pec split and the lower pec border. The count is how many lines were drawn."""
    torso = hip_y - shoulder_y
    if torso <= 1:
        return 0, []
    y0 = int(np.clip(round(shoulder_y + 0.10 * torso), 0, gray.shape[0] - 2))
    y1 = int(np.clip(round(shoulder_y + 0.42 * torso), y0 + 2, gray.shape[0]))
    half = max(8.0, 0.55 * chest_width)
    x0 = int(np.clip(round(center_x - half), 0, gray.shape[1] - 2))
    x1 = int(np.clip(round(center_x + half), x0 + 2, gray.shape[1]))
    roi = gray[y0:y1, x0:x1].astype(np.float32)
    roi_mask = mask[y0:y1, x0:x1]
    if roi.size == 0 or int(roi_mask.sum()) < 20:
        return 0, []
    local = roi - cv2.GaussianBlur(roi, (0, 0), 9)
    gap = max(10, int(round(0.08 * torso)))
    vertical = cv2.filter2D(local, cv2.CV_32F, _VERTICAL)
    vertical = vertical.copy()
    vertical[~roi_mask] = 0.0
    upright = _vertical_lines(
        mask,
        peak_indices(np.abs(vertical).mean(axis=0), min_gap=gap),
        x0,
        y0,
        y1,
        (y0 + y1) / 2.0,
    )
    # The lower border sits under the pecs, not up at the collarbone.
    split = y0 + int(0.45 * (y1 - y0))
    lower = local[split - y0:, :]
    lower_mask = roi_mask[split - y0:, :]
    horizontal = cv2.filter2D(lower, cv2.CV_32F, _KERNEL)
    horizontal = horizontal.copy()
    horizontal[~lower_mask] = 0.0
    under = _horizontal_lines(
        mask,
        peak_indices(horizontal.mean(axis=1), min_gap=gap),
        split,
        x0,
        x1,
        center_x,
    )
    lines = upright[:3] + under[:2]
    return len(lines), lines
