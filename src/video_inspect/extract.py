"""Frame extraction with real per-frame timestamps.

Uses ffmpeg's `showinfo` filter instead of frame_index/fps arithmetic, so
variable-frame-rate sources (macOS screen recordings, re-muxed clips) get
correct timestamps. `showinfo` runs in the same filter chain as the scale,
so the Nth line of its stderr log corresponds exactly to the Nth output
file — one decode pass, no drift between what was measured and what was
written to disk.
"""
from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from . import provenance

PTS_RE = re.compile(rb"pts_time:([0-9.]+)")


@dataclass
class AnalysisFrame:
    index: int  # 0-based position in decode order
    timestamp_s: float
    path: Path


def extract_analysis_frames(
    source: Path, out_dir: Path, *, width: int = 320, max_frames: int | None = None
) -> list[AnalysisFrame]:
    """Decode every frame once at low resolution, capturing true PTS via showinfo."""
    out_dir.mkdir(parents=True, exist_ok=True)
    provenance.require_tool("ffmpeg")
    vf = f"showinfo,scale='min({width}\\,iw)':-2:flags=area"
    cmd = [
        "ffmpeg",
        "-y",
        "-v",
        "info",
        "-i",
        str(source),
        "-vf",
        vf,
        "-vsync",
        "0",
        "-q:v",
        "3",
    ]
    if max_frames is not None:
        # Must precede the output path: ffmpeg binds an output option to
        # the next output URL that follows it, not to one that already
        # came before it, so appending this after the output pattern was a
        # silent no-op (--max-analysis-frames had no effect).
        cmd += ["-frames:v", str(max_frames)]
    cmd.append(str(out_dir / "f_%06d.jpg"))
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stderr = proc.stderr
    timestamps = [float(m.group(1)) for m in PTS_RE.finditer(stderr)]
    files = sorted(out_dir.glob("f_*.jpg"))
    if not files:
        provenance.fail(
            f"ffmpeg produced no frames for {source}: "
            f"{stderr.decode('utf-8', 'replace')[-2000:]}"
        )
    if len(timestamps) != len(files):
        # Fall back to fps-derived timestamps rather than crash; this is the
        # exact case a report.md warning must surface (see manifest.warnings).
        timestamps = None
    frames = []
    for idx, fpath in enumerate(files):
        ts = timestamps[idx] if timestamps else None
        frames.append(AnalysisFrame(index=idx, timestamp_s=ts if ts is not None else -1.0, path=fpath))
    return frames


def extract_frames_by_index(
    source: Path,
    frame_indices: list[int],
    out_dir: Path,
    *,
    width: int | None = None,
    prefix: str = "sel",
) -> dict[int, Path]:
    """Re-decode the source once, keeping only the requested 0-based frame
    indices, at (optionally) a different resolution than the analysis pass.
    Uses ffmpeg's `select` filter on frame number `n`, which is exact and
    avoids the inaccuracy of timestamp-based seeking near keyframes.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    if not frame_indices:
        return {}
    ordered = sorted(set(frame_indices))
    select_expr = "+".join(f"eq(n\\,{i})" for i in ordered)
    filters = [f"select='{select_expr}'"]
    if width:
        filters.append(f"scale='min({width}\\,iw)':-2:flags=area")
    vf = ",".join(filters)
    pattern = out_dir / f"{prefix}_%06d.jpg"
    cmd = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-i",
        str(source),
        "-vf",
        vf,
        "-vsync",
        "0",
        "-q:v",
        "2",
        str(pattern),
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        provenance.fail(
            f"ffmpeg frame selection failed: {proc.stderr.decode('utf-8', 'replace')}"
        )
    produced = sorted(out_dir.glob(f"{prefix}_*.jpg"))
    if len(produced) != len(ordered):
        provenance.fail(
            f"expected {len(ordered)} selected frames, ffmpeg wrote {len(produced)}"
        )
    return dict(zip(ordered, produced))


def extract_crop_burst(
    source: Path,
    center_time_s: float,
    bbox_norm: tuple[float, float, float, float],
    src_width: int,
    src_height: int,
    out_dir: Path,
    *,
    window_s: float = 1.0,
    frame_count: int = 9,
    padding_frac: float = 0.25,
    prefix: str = "zoom",
) -> list[tuple[float, Path]]:
    """Crop a padded region around bbox_norm from the ORIGINAL source across
    a short time window, at native resolution. Never upsamples a thumbnail:
    this always re-decodes the source video.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    x, y, w, h = bbox_norm
    pad_w = w * padding_frac
    pad_h = h * padding_frac
    x0 = max(0.0, x - pad_w)
    y0 = max(0.0, y - pad_h)
    x1 = min(1.0, x + w + pad_w)
    y1 = min(1.0, y + h + pad_h)
    crop_w = max(2, int(round((x1 - x0) * src_width)))
    crop_h = max(2, int(round((y1 - y0) * src_height)))
    crop_x = int(round(x0 * src_width))
    crop_y = int(round(y0 * src_height))
    # even dimensions keep most codecs/filters happy
    crop_w -= crop_w % 2
    crop_h -= crop_h % 2

    start = max(0.0, center_time_s - window_s)
    duration = window_s * 2

    vf = f"crop={crop_w}:{crop_h}:{crop_x}:{crop_y},showinfo"
    pattern = out_dir / f"{prefix}_all_%06d.jpg"
    cmd = [
        "ffmpeg",
        "-y",
        "-v",
        "info",
        "-ss",
        f"{start:.3f}",
        "-i",
        str(source),
        "-t",
        f"{duration:.3f}",
        "-vf",
        vf,
        "-vsync",
        "0",
        "-q:v",
        "2",
        str(pattern),
    ]
    # No -frames:v cap here: it would take the FIRST frame_count frames
    # after the seek (a tiny sliver at the start of the window, e.g. 5
    # frames at 30fps = 0.17s), not frame_count frames spread across the
    # requested window, so a burst near the start of a wide window could
    # miss an event happening later in that same window. Decode every
    # frame in the window instead, then subsample evenly in Python.
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    timestamps_all = [float(m.group(1)) + start for m in PTS_RE.finditer(proc.stderr)]
    files_all = sorted(out_dir.glob(f"{prefix}_all_*.jpg"))
    if not files_all:
        provenance.fail(
            f"zoom crop produced no frames: {proc.stderr.decode('utf-8', 'replace')[-1500:]}"
        )
    if len(timestamps_all) != len(files_all):
        timestamps_all = [start + i / 30.0 for i in range(len(files_all))]  # best-effort fallback

    n = min(frame_count, len(files_all))
    if n <= 1:
        pick_idx = [0]
    else:
        pick_idx = sorted({round(i * (len(files_all) - 1) / (n - 1)) for i in range(n)})

    result = []
    for rank, idx in enumerate(pick_idx, start=1):
        src = files_all[idx]
        dest = out_dir / f"{prefix}_{rank:06d}.jpg"
        src.rename(dest)
        result.append((timestamps_all[idx], dest))
    for leftover in out_dir.glob(f"{prefix}_all_*.jpg"):
        leftover.unlink()
    return result
