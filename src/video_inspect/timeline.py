"""A single temporal gestalt image: change/local-change curves plus markers
for every selected frame ID. Deliberately NOT a motion-history image or a
slit-scan: those need a legend the viewer must already know, and a plain
curve over time is more directly readable without one.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from .env_check import load_font

WIDTH = 1440
HEIGHT = 260
MARGIN_L = 60
MARGIN_R = 20
MARGIN_T = 30
MARGIN_B = 40
GLOBAL_COLOR = (70, 110, 200)
LOCAL_COLOR = (210, 90, 40)
MARKER_COLOR = (30, 30, 30)
BG = (255, 255, 255)
GRID_COLOR = (225, 225, 225)


def render_timeline(
    scores: list,  # list[sampler.FrameScore]
    selected: list,  # list[sampler.SelectedFrame]
    out_path: Path,
    *,
    font_path: Path,
    duration_s: float | None = None,
) -> Path:
    img = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(img)
    font = load_font(font_path, 12)
    title_font = load_font(font_path, 14)

    plot_w = WIDTH - MARGIN_L - MARGIN_R
    plot_h = HEIGHT - MARGIN_T - MARGIN_B
    plot_x0, plot_y0 = MARGIN_L, MARGIN_T
    plot_x1, plot_y1 = plot_x0 + plot_w, plot_y0 + plot_h

    draw.text((MARGIN_L, 8), "change-energy timeline (blue=global, orange=local)", fill=(0, 0, 0), font=title_font)

    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = plot_y1 - frac * plot_h
        draw.line([(plot_x0, y), (plot_x1, y)], fill=GRID_COLOR, width=1)
        draw.text((4, y - 6), f"{frac:.2f}", fill=(120, 120, 120), font=font)

    draw.rectangle([plot_x0, plot_y0, plot_x1, plot_y1], outline=(180, 180, 180), width=1)

    if not scores:
        img.save(out_path, "PNG")
        return out_path

    # default=1.0 covers frame-directory input with no --assume-fps, where
    # every timestamp is -1 (unknown): without it, max() over the empty
    # filtered sequence raises ValueError and the whole run crashes instead
    # of degrading to "all markers at x=0", which is what unknown
    # timestamps actually are.
    max_t = duration_s or max((s.timestamp_s for s in scores if s.timestamp_s >= 0), default=1.0) or 1.0

    def xy(t: float, v: float) -> tuple[float, float]:
        x = plot_x0 + (max(0.0, t) / max_t) * plot_w
        y = plot_y1 - max(0.0, min(1.0, v)) * plot_h
        return x, y

    global_pts = [xy(s.timestamp_s, s.global_change_fraction) for s in scores if s.timestamp_s >= 0]
    local_pts = [xy(s.timestamp_s, s.local_energy) for s in scores if s.timestamp_s >= 0]
    if len(global_pts) > 1:
        draw.line(global_pts, fill=GLOBAL_COLOR, width=2)
    if len(local_pts) > 1:
        draw.line(local_pts, fill=LOCAL_COLOR, width=2)

    for i, sel in enumerate(selected):
        x, _ = xy(sel.timestamp_s, 0)
        draw.line([(x, plot_y0), (x, plot_y1)], fill=MARKER_COLOR, width=1)
        label = f"F{i+1:03d}"
        draw.text((x + 2, plot_y1 + 4), label, fill=MARKER_COLOR, font=font)

    for frac in (0.0, 0.5, 1.0):
        t = frac * max_t
        x, _ = xy(t, 0)
        draw.text((x - 10, plot_y1 + 20), f"{t:.1f}s", fill=(120, 120, 120), font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, "PNG")
    return out_path
