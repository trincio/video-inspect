#!/usr/bin/env python3
"""Generate a small corpus of synthetic test videos with known ground truth.

Enough to exercise the main failure mode this tool cares about: losing a
small, short, or local event. All videos are 640x360 @ 30fps, built with
ffmpeg lavfi sources plus a hand-drawn sprite, so every event's exact
time/position is known and recorded in ground_truth.json next to each
video.
"""
from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).parent
SYN = HERE / "synthetic"
SYN.mkdir(exist_ok=True)
W, H = 640, 360
FPS = 30
FONT_PATH = HERE.parent / "fonts" / "DejaVuSansMono.ttf"


def run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise SystemExit(f"command failed: {' '.join(cmd)}\n{proc.stderr.decode('utf-8', 'replace')}")


def gen_hard_cut() -> dict:
    out = SYN / "hard_cut.mp4"
    run(
        [
            "ffmpeg", "-y", "-v", "error",
            "-f", "lavfi", "-i", f"color=c=red:s={W}x{H}:d=2:r={FPS}",
            "-f", "lavfi", "-i", f"color=c=blue:s={W}x{H}:d=2:r={FPS}",
            "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0[v]",
            "-map", "[v]", "-pix_fmt", "yuv420p", str(out),
        ]
    )
    return {
        "file": out.name,
        "description": "hard cut red->blue at t=2.0s, no other change",
        "events": [{"type": "hard_cut", "start_s": 2.0, "end_s": 2.0}],
        "duration_s": 4.0,
    }


def _text_sprite(text: str, path: Path, *, font_size: int = 48) -> tuple[int, int]:
    font = ImageFont.truetype(str(FONT_PATH), font_size)
    tmp = Image.new("RGBA", (10, 10))
    bbox = ImageDraw.Draw(tmp).textbbox((0, 0), text, font=font)
    w, h = bbox[2] - bbox[0] + 8, bbox[3] - bbox[1] + 8
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    ImageDraw.Draw(img).text((4 - bbox[0], 4 - bbox[1]), text, font=font, fill=(255, 255, 255, 255))
    img.save(path)
    return w, h


def gen_static_text_change() -> dict:
    out = SYN / "static_text_change.mp4"
    change_t = 2.5
    duration = 5.0
    sprite_a = SYN / "_text_a.png"
    sprite_b = SYN / "_text_b.png"
    wa, ha = _text_sprite("STATO A", sprite_a)
    wb, hb = _text_sprite("STATO B", sprite_b)
    xa, ya = (W - wa) // 2, (H - ha) // 2
    xb, yb = (W - wb) // 2, (H - hb) // 2
    run(
        [
            "ffmpeg", "-y", "-v", "error",
            "-f", "lavfi", "-i", f"color=c=0x202020:s={W}x{H}:d={duration}:r={FPS}",
            "-i", str(sprite_a), "-i", str(sprite_b),
            "-filter_complex",
            f"[0:v][1:v]overlay=x={xa}:y={ya}:enable='lt(t\\,{change_t})'[v1];"
            f"[v1][2:v]overlay=x={xb}:y={yb}:enable='gte(t\\,{change_t})'[v]",
            "-map", "[v]", "-pix_fmt", "yuv420p", str(out),
        ]
    )
    return {
        "file": out.name,
        "description": "background static; text changes STATO A -> STATO B at t=2.5s",
        "events": [{"type": "static_text_change", "start_s": change_t, "end_s": change_t}],
        "duration_s": duration,
    }


def gen_slow_dissolve() -> dict:
    out = SYN / "slow_dissolve.mp4"
    seg1, trans, seg2 = 2.0, 1.5, 2.0
    duration = seg1 + trans + seg2
    run(
        [
            "ffmpeg", "-y", "-v", "error",
            "-f", "lavfi", "-i", f"color=c=0x104060:s={W}x{H}:d={seg1 + trans}:r={FPS}",
            "-f", "lavfi", "-i", f"color=c=0x804010:s={W}x{H}:d={trans + seg2}:r={FPS}",
            "-filter_complex", f"[0:v][1:v]xfade=transition=fade:duration={trans}:offset={seg1}[v]",
            "-map", "[v]", "-pix_fmt", "yuv420p", str(out),
        ]
    )
    return {
        "file": out.name,
        "description": f"slow crossfade dark-blue -> dark-orange, {trans}s starting at t={seg1}s",
        "events": [{"type": "slow_dissolve", "start_s": seg1, "end_s": seg1 + trans}],
        "duration_s": duration,
    }


def gen_tiny_yellow_ball() -> dict:
    out = SYN / "tiny_yellow_ball.mp4"
    duration = 5.0
    radius = 6
    sprite_path = SYN / "_ball_sprite.png"
    size = radius * 2 + 4
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.ellipse([2, 2, 2 + radius * 2, 2 + radius * 2], fill=(255, 220, 0, 255))
    img.save(sprite_path)

    # Ball crosses from bottom-left to top-right, entering at t=2.0s and
    # exiting at t=3.0s; static frame otherwise. Position is linear in t.
    x0, y0 = int(0.05 * W), int(0.85 * H)
    x1, y1 = int(0.85 * W), int(0.10 * H)
    t_in, t_out = 2.0, 3.0

    x_expr = f"if(between(t\\,{t_in}\\,{t_out})\\,{x0}+({x1}-{x0})*(t-{t_in})/({t_out}-{t_in})\\,-100)"
    y_expr = f"if(between(t\\,{t_in}\\,{t_out})\\,{y0}+({y1}-{y0})*(t-{t_in})/({t_out}-{t_in})\\,-100)"

    run(
        [
            "ffmpeg", "-y", "-v", "error",
            "-f", "lavfi", "-i", f"color=c=0x303030:s={W}x{H}:d={duration}:r={FPS}",
            "-i", str(sprite_path),
            "-filter_complex", f"[0:v][1:v]overlay=x='{x_expr}':y='{y_expr}'[v]",
            "-map", "[v]", "-pix_fmt", "yuv420p", str(out),
        ]
    )

    def pos_at(t: float) -> tuple[float, float]:
        frac = (t - t_in) / (t_out - t_in)
        return x0 + (x1 - x0) * frac, y0 + (y1 - y0) * frac

    area_frac = (math.pi * radius * radius) / (W * H)
    return {
        "file": out.name,
        "description": f"tiny yellow ball (r={radius}px, area_frac={area_frac:.5f}) crosses bottom-left to top-right between t={t_in}s and t={t_out}s",
        "events": [
            {
                "type": "tiny_object",
                "start_s": t_in,
                "end_s": t_out,
                "radius_px": radius,
                "area_frac": area_frac,
                "path": {"x0": x0, "y0": y0, "x1": x1, "y1": y1},
            }
        ],
        "duration_s": duration,
    }


def gen_one_frame_flash() -> dict:
    out = SYN / "one_frame_flash.mp4"
    duration = 4.0
    flash_frame_t = 2.0
    run(
        [
            "ffmpeg", "-y", "-v", "error",
            "-f", "lavfi", "-i", f"color=c=0x202020:s={W}x{H}:d={duration}:r={FPS}",
            "-vf", f"drawbox=x=0:y=0:w={W}:h={H}:color=white@1.0:t=fill:enable='between(t\\,{flash_frame_t}\\,{flash_frame_t + 1/FPS})'",
            "-pix_fmt", "yuv420p", str(out),
        ]
    )
    return {
        "file": out.name,
        "description": f"single white flash frame at t={flash_frame_t}s, otherwise static",
        "events": [{"type": "one_frame_flash", "start_s": flash_frame_t, "end_s": flash_frame_t + 1 / FPS}],
        "duration_s": duration,
    }


def gen_long_rare_event() -> dict:
    """Stress test for uniform coverage: a 20s mostly-static video where the
    only event is a 0.4s ball crossing near t=13.7s. At a small budget,
    uniform anchor spacing (~20/7 =~2.9s here) is wide enough that a scene-
    /local-aware sampler should meaningfully outperform uniform-only on
    event_recall, unlike the short 5s tiny_yellow_ball case where uniform
    density alone already guarantees a hit.
    """
    out = SYN / "long_rare_event.mp4"
    duration = 20.0
    radius = 5
    sprite_path = SYN / "_ball_sprite_small.png"
    size = radius * 2 + 4
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    ImageDraw.Draw(img).ellipse([2, 2, 2 + radius * 2, 2 + radius * 2], fill=(255, 220, 0, 255))
    img.save(sprite_path)

    t_in, t_out = 13.7, 14.1
    x0, y0 = int(0.10 * W), int(0.80 * H)
    x1, y1 = int(0.75 * W), int(0.15 * H)
    x_expr = f"if(between(t\\,{t_in}\\,{t_out})\\,{x0}+({x1}-{x0})*(t-{t_in})/({t_out}-{t_in})\\,-100)"
    y_expr = f"if(between(t\\,{t_in}\\,{t_out})\\,{y0}+({y1}-{y0})*(t-{t_in})/({t_out}-{t_in})\\,-100)"

    run(
        [
            "ffmpeg", "-y", "-v", "error",
            "-f", "lavfi", "-i", f"color=c=0x282828:s={W}x{H}:d={duration}:r={FPS}",
            "-i", str(sprite_path),
            "-filter_complex", f"[0:v][1:v]overlay=x='{x_expr}':y='{y_expr}'[v]",
            "-map", "[v]", "-pix_fmt", "yuv420p", str(out),
        ]
    )
    area_frac = (math.pi * radius * radius) / (W * H)
    return {
        "file": out.name,
        "description": f"20s mostly-static video; tiny ball (r={radius}px, area_frac={area_frac:.5f}) visible only for {t_out - t_in:.1f}s around t={t_in}s",
        "events": [
            {
                "type": "tiny_object",
                "start_s": t_in,
                "end_s": t_out,
                "radius_px": radius,
                "area_frac": area_frac,
                "path": {"x0": x0, "y0": y0, "x1": x1, "y1": y1},
            }
        ],
        "duration_s": duration,
    }


def main() -> None:
    generators = [
        gen_hard_cut,
        gen_static_text_change,
        gen_slow_dissolve,
        gen_tiny_yellow_ball,
        gen_one_frame_flash,
        gen_long_rare_event,
    ]
    ground_truth = {}
    for gen in generators:
        info = gen()
        ground_truth[info["file"]] = info
        print(f"generated {info['file']}: {info['description']}")
    (SYN / "ground_truth.json").write_text(json.dumps(ground_truth, indent=2))
    print(f"wrote {SYN / 'ground_truth.json'}")


if __name__ == "__main__":
    main()
