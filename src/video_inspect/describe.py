"""Controlled-vocabulary verbal descriptors for local-change events.

The vocabulary is fixed and small on purpose: "yellow blob, top-right" is a
measurement translated to words, not a semantic label. "tennis ball" is the
reader's job, not this module's.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

GRID_COLS = 4
GRID_ROWS = 3
COL_LETTERS = "ABCD"

_HUE_BUCKETS = [
    (0, "ROSSO"), (15, "ARANCIONE"), (35, "GIALLO"), (75, "VERDE"),
    (150, "CIANO"), (200, "BLU"), (260, "VIOLA"), (320, "MAGENTA"), (360, "ROSSO"),
]


def hue_name(hue_deg: float) -> str:
    for upper, name in _HUE_BUCKETS:
        if hue_deg <= upper:
            return name
    return "ROSSO"


def spatial_cell_id(row: int, col: int, rows: int, cols: int) -> str:
    """Map a (row, col) in an arbitrary rows x cols grid to the fixed
    GRID_ROWS x GRID_COLS addressing scheme used for --cell in `zoom`.
    """
    frac_row = row / max(1, rows - 1) if rows > 1 else 0.0
    frac_col = col / max(1, cols - 1) if cols > 1 else 0.0
    out_row = min(GRID_ROWS - 1, int(frac_row * GRID_ROWS))
    out_col = min(GRID_COLS - 1, int(frac_col * GRID_COLS))
    return f"{COL_LETTERS[out_col]}{out_row + 1}"


_DIR_BY_POSITION = [
    ["NW", "N", "N", "NE"],
    ["W", "C", "C", "E"],
    ["SW", "S", "S", "SE"],
]


def direction_word(row: int, col: int, rows: int, cols: int) -> str:
    frac_row = row / max(1, rows - 1) if rows > 1 else 0.0
    frac_col = col / max(1, cols - 1) if cols > 1 else 0.0
    # int(frac * n) already lands in [0, n-1] for frac in [0, 1); only the
    # frac == 1.0 edge (rightmost/bottommost cell) needs the clamp, since
    # int(1.0 * n) == n is one past the last valid index. An earlier
    # version also remapped column 3 down to column 2 "to be safe", which
    # instead made the table's rightmost column (NE/E/SE) unreachable.
    r = min(2, int(frac_row * 3))
    c = min(3, int(frac_col * 4))
    return _DIR_BY_POSITION[r][c]


def area_bucket(area_frac: float) -> str:
    pct = area_frac * 100
    if pct < 0.5:
        return "TINY"
    if pct < 3:
        return "SMALL"
    if pct < 15:
        return "MEDIUM"
    return "LARGE"


@dataclass
class LocalEventDescriptor:
    cell_id: str
    direction: str
    area_bucket: str
    dominant_hue: str | None
    bbox_norm: tuple[float, float, float, float]


def describe_local_event(
    peak_rc: tuple[int, int],
    grid_shape: tuple[int, int],
    area_frac: float,
    frame_bgr: np.ndarray | None,
) -> LocalEventDescriptor:
    import cv2

    row, col = peak_rc
    rows, cols = grid_shape
    cell_id = spatial_cell_id(row, col, rows, cols)
    direction = direction_word(row, col, rows, cols)
    bucket = area_bucket(area_frac)

    x0, y0 = col / cols, row / rows
    w, h = 1.0 / cols, 1.0 / rows
    bbox = (x0, y0, w, h)

    dominant_hue = None
    if frame_bgr is not None:
        fh, fw = frame_bgr.shape[:2]
        px0, py0 = int(x0 * fw), int(y0 * fh)
        px1, py1 = int((x0 + w) * fw), int((y0 + h) * fh)
        patch = frame_bgr[max(0, py0) : max(1, py1), max(0, px0) : max(1, px1)]
        if patch.size:
            hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
            sat = hsv[:, :, 1]
            val = hsv[:, :, 2]
            mask = (sat > 60) & (val > 60)
            if mask.any():
                mean_hue = float(hsv[:, :, 0][mask].mean()) * 2  # OpenCV hue 0-180 -> degrees
                dominant_hue = hue_name(mean_hue)

    return LocalEventDescriptor(cell_id, direction, bucket, dominant_hue, bbox)


def verbal_caption(descriptor: LocalEventDescriptor | None) -> str:
    """Kept deliberately short: a longer phrase (motion word + area-bucket +
    hue + cell) overflows the caption band on a 4-column sheet at a legible
    font size. Direction + hue + cell is the information that is hardest to
    get from the thumbnail itself at a glance; area is visible directly in
    the image, so it is dropped here (still recorded in metrics.json/zones).
    """
    if descriptor is None:
        return ""
    parts = [descriptor.direction]
    if descriptor.dominant_hue:
        parts.append(descriptor.dominant_hue.lower())
    parts.append(descriptor.cell_id)
    return " ".join(parts)
