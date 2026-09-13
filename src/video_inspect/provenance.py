"""Source probing, hashing and tool-version provenance.

Deterministic evidence starts here: every run records exactly what it read,
with what tools, so a claim like "F007 is at 12.4s" can be checked against
the actual decoded stream, not against an assumption like frame_index/fps
(which is wrong for variable-frame-rate captures, e.g. macOS screen
recordings).
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

VERSION = "0.1.0"


def fail(message: str) -> None:
    raise SystemExit(f"video-inspect: {message}")


def require_tool(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        fail(f"required tool not found on PATH: {name}")
    return path


def run(cmd: list[str], *, capture: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE,
        check=False,
    )


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ffmpeg_version() -> str:
    require_tool("ffmpeg")
    out = run(["ffmpeg", "-version"]).stdout.decode("utf-8", "replace")
    return out.splitlines()[0].strip() if out else "unknown"


def opencv_version() -> str:
    try:
        import cv2

        return cv2.__version__
    except ImportError:
        return "not-installed"


def pillow_version() -> str:
    try:
        import PIL

        return PIL.__version__
    except ImportError:
        return "not-installed"


@dataclass
class SourceInfo:
    kind: str  # "video" | "frame_dir"
    path: str
    sha256: str | None
    duration_s: float | None
    fps: float | None
    width: int
    height: int
    frame_count_estimate: int | None
    codec_name: str | None
    is_vfr: bool = False
    frame_hashes: dict = field(default_factory=dict)


def probe_video(path: Path) -> SourceInfo:
    require_tool("ffprobe")
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate,avg_frame_rate,duration,nb_frames,codec_name",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(path),
    ]
    proc = run(cmd)
    if proc.returncode != 0:
        fail(f"ffprobe failed on {path}: {proc.stderr.decode('utf-8', 'replace')}")
    data = json.loads(proc.stdout.decode("utf-8", "replace"))
    streams = data.get("streams") or []
    if not streams:
        fail(f"no video stream found in {path}")
    st = streams[0]

    def parse_rate(rate: str | None) -> float | None:
        if not rate or rate in ("0/0", "N/A"):
            return None
        num, _, den = rate.partition("/")
        try:
            num_f, den_f = float(num), float(den or 1)
            return num_f / den_f if den_f else None
        except ValueError:
            return None

    r_rate = parse_rate(st.get("r_frame_rate"))
    avg_rate = parse_rate(st.get("avg_frame_rate"))
    duration = None
    for src in (st.get("duration"), (data.get("format") or {}).get("duration")):
        if src not in (None, "N/A"):
            try:
                duration = float(src)
                break
            except ValueError:
                continue

    nb_frames = None
    if st.get("nb_frames") not in (None, "N/A"):
        try:
            nb_frames = int(st["nb_frames"])
        except ValueError:
            nb_frames = None

    is_vfr = bool(r_rate and avg_rate and abs(r_rate - avg_rate) > 0.01)

    return SourceInfo(
        kind="video",
        path=str(path),
        sha256=sha256_file(path),
        duration_s=duration,
        fps=avg_rate or r_rate,
        width=int(st.get("width") or 0),
        height=int(st.get("height") or 0),
        frame_count_estimate=nb_frames,
        codec_name=st.get("codec_name"),
        is_vfr=is_vfr,
    )


def probe_frame_dir(path: Path) -> SourceInfo:
    files = sorted(
        [*path.glob("*.png"), *path.glob("*.jpg"), *path.glob("*.jpeg")]
    )
    if not files:
        fail(f"no PNG/JPEG frames found in {path}")
    try:
        import cv2

        sample = cv2.imread(str(files[0]))
        height, width = sample.shape[:2] if sample is not None else (0, 0)
    except ImportError:
        width = height = 0

    combined = hashlib.sha256()
    for f in files:
        combined.update(sha256_file(f).encode())
    return SourceInfo(
        kind="frame_dir",
        path=str(path),
        sha256=combined.hexdigest(),
        duration_s=None,
        fps=None,
        width=width,
        height=height,
        frame_count_estimate=len(files),
        codec_name=None,
        is_vfr=False,
    )


def probe_source(path: Path) -> SourceInfo:
    if path.is_dir():
        return probe_frame_dir(path)
    return probe_video(path)


def tool_versions() -> dict:
    return {
        "python": sys.version.split()[0],
        "ffmpeg": ffmpeg_version(),
        "opencv": opencv_version(),
        "pillow": pillow_version(),
        "video_inspect": VERSION,
    }
