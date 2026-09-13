from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

from . import describe, env_check, extract, manifest as manifest_mod, provenance, sampler, sheet, timeline

# Resolved lazily (not at import time): find_bundled_font() is safe to call
# even when the font is missing, and doctor/other commands must be able to
# report "missing" instead of crashing on import.
DEFAULT_FONT = env_check.find_bundled_font()

REASON_CODE = {
    "uniform": "U",
    "uniform_fill": "U",
    "scene": "S",
    "local_change": "L",
    "all": "A",
}


def _reason_codes(reasons: list[str]) -> list[str]:
    seen = []
    for r in reasons:
        code = REASON_CODE.get(r, "?")
        if code not in seen:
            seen.append(code)
    return seen


def _validate_inspect_args(args: argparse.Namespace) -> None:
    if args.budget < 1:
        provenance.fail(f"--budget must be >= 1, got {args.budget}")
    if args.uniform_reserved is not None:
        if args.uniform_reserved < 0:
            provenance.fail(f"--uniform-reserved must be >= 0, got {args.uniform_reserved}")
        if args.uniform_reserved > args.budget:
            provenance.fail(
                f"--uniform-reserved ({args.uniform_reserved}) cannot exceed --budget ({args.budget})"
            )
    if args.rows < 1 or args.cols < 1:
        provenance.fail(f"--rows and --cols must be >= 1, got rows={args.rows} cols={args.cols}")
    if args.per_page > args.rows * args.cols:
        provenance.fail(
            f"--per-page ({args.per_page}) exceeds --rows * --cols ({args.rows * args.cols}); "
            "extra cells would be drawn outside the canvas. Increase --rows/--cols or lower --per-page."
        )
    if args.per_page < 1:
        provenance.fail(f"--per-page must be >= 1, got {args.per_page}")
    if args.analysis_width < 16 or args.thumb_width < 16:
        provenance.fail("--analysis-width and --thumb-width must be at least 16px")


def _should_describe(sel: sampler.SelectedFrame, min_caption_energy: float) -> bool:
    """Whether a selected frame has earned a zone/color caption, as opposed
    to just its reason code. Kept as its own function so the exact
    condition is unit-testable: a prior version gated this on
    local_area_frac > 0, which is mathematically equivalent to requiring
    local_energy >= 0.5 (area_frac's hard threshold and local_energy's
    normalization share the same --local-threshold and the same max-cell
    delta) — far too strict for smooth, anti-aliased content, where a real
    local event's peak delta can legitimately stay under a full threshold
    crossing for every single frame of a video.
    """
    return (
        "local_change" in sel.reasons
        and sel.local_peak_rc is not None
        and sel.local_grid_shape is not None
        and sel.local_energy >= min_caption_energy
    )


def cmd_inspect(args: argparse.Namespace) -> int:
    _validate_inspect_args(args)
    source_path = Path(args.source).resolve()
    if not source_path.exists():
        provenance.fail(f"source not found: {source_path}")
    out_dir = Path(args.output).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = out_dir / "frames"
    # Re-running `inspect` into the same --output (e.g. sweeping --budget
    # or --sampler while comparing) must not leave stale F0xx.jpg from a
    # prior, larger selection lying around next to the new ones.
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir()

    warnings: list[str] = list(env_check.optional_warnings(DEFAULT_FONT))
    for w in warnings:
        print(f"video-inspect: {w}", file=sys.stderr)
    src_info = provenance.probe_source(source_path)

    tmp_analysis = Path(tempfile.mkdtemp(prefix="video-inspect-analysis-"))
    tmp_thumbs: Path | None = None
    try:
        if src_info.kind == "video":
            analysis_frames = extract.extract_analysis_frames(
                source_path, tmp_analysis, width=args.analysis_width, max_frames=args.max_analysis_frames
            )
            if analysis_frames and analysis_frames[0].timestamp_s < 0:
                warnings.append(
                    "ffmpeg showinfo did not emit pts_time for every frame; timestamps may be unavailable"
                )
        else:
            files = sorted([*source_path.glob("*.png"), *source_path.glob("*.jpg"), *source_path.glob("*.jpeg")])
            if args.assume_fps:
                ts = lambda i: i / args.assume_fps
            else:
                warnings.append(
                    "frame directory input without --assume-fps: timestamps are unavailable (-1), "
                    "only frame index and position_pct are reliable"
                )
                ts = lambda i: -1.0
            analysis_frames = [
                extract.AnalysisFrame(index=i, timestamp_s=ts(i), path=f) for i, f in enumerate(files)
            ]

        if not analysis_frames:
            provenance.fail("no frames decoded from source")

        scores = sampler.score_frames(
            analysis_frames,
            global_delta_threshold=args.global_threshold,
            local_delta_threshold=args.local_threshold,
        )

        if args.sampler == "uniform":
            resolved_uniform_reserved = args.budget
            selected = sampler.hybrid_select(scores, budget=args.budget, uniform_reserved=resolved_uniform_reserved)
        elif args.sampler == "scene":
            resolved_uniform_reserved = 0
            # local_weight=0: rank candidates by global_change_fraction only,
            # so "scene" is actually scene/global-change-only, not a relabeled
            # hybrid ranking with uniform_reserved=0.
            selected = sampler.hybrid_select(
                scores, budget=args.budget, uniform_reserved=resolved_uniform_reserved, local_weight=0.0
            )
        else:
            resolved_uniform_reserved = (
                args.uniform_reserved if args.uniform_reserved is not None else max(2, args.budget // 3)
            )
            selected = sampler.hybrid_select(
                scores, budget=args.budget, uniform_reserved=resolved_uniform_reserved, min_event_spacing_s=args.min_event_spacing
            )

        selected_indices = [s.index for s in selected]

        if src_info.kind == "video":
            # Extract into a fresh temp dir, never directly into frames_dir:
            # frames_dir persists across re-runs of the same --output (e.g.
            # a budget/sampler sweep reusing a directory), and a fixed
            # "sel_*" prefix there would pick up stale files from a prior
            # run with a different selection count. Left alive (via
            # tmp_thumbs, cleaned in the outer finally) until the per-frame
            # loop below copies each one straight to its final F0xx.jpg
            # name, so no intermediate sel_*.jpg name is ever left behind
            # alongside the final one.
            tmp_thumbs = Path(tempfile.mkdtemp(prefix="video-inspect-thumbs-"))
            thumb_paths = extract.extract_frames_by_index(
                source_path, selected_indices, tmp_thumbs, width=args.thumb_width
            )
            ordered = sorted(selected_indices)
            index_to_path = {idx: thumb_paths[idx] for idx in ordered}
        else:
            index_to_path = {s.index: analysis_frames[s.index].path for s in selected}

        import cv2

        total_frames = len(scores)
        duration_s = src_info.duration_s
        cells: list[sheet.CellData] = []
        per_frame_metrics = []
        zones = []
        frame_manifest_entries = []

        for i, sel in enumerate(selected, start=1):
            frame_id = f"F{i:03d}"
            src_thumb = index_to_path.get(sel.index)
            dest_name = f"{frame_id}.jpg"
            dest_path = frames_dir / dest_name
            if src_thumb and src_thumb != dest_path:
                shutil.copy(src_thumb, dest_path)

            if duration_s and sel.timestamp_s >= 0:
                position_pct = (sel.timestamp_s / duration_s) * 100
            else:
                position_pct = (sel.index / max(1, total_frames - 1)) * 100

            descriptor = None
            if _should_describe(sel, args.min_caption_energy):
                frame_bgr = cv2.imread(str(dest_path)) if dest_path.exists() else None
                descriptor = describe.describe_local_event(
                    sel.local_peak_rc, sel.local_grid_shape, sel.local_area_frac, frame_bgr
                )
                zones.append(
                    {
                        "id": f"{frame_id}-R1",
                        "frame_id": frame_id,
                        "bbox_norm": list(descriptor.bbox_norm),
                        "area_pct": sel.local_area_frac * 100,
                        "direction": descriptor.direction,
                        "dominant_hue": descriptor.dominant_hue,
                        "cell": descriptor.cell_id,
                    }
                )

            cells.append(
                sheet.CellData(
                    frame_id=frame_id,
                    image_path=dest_path,
                    timestamp_s=sel.timestamp_s,
                    position_pct=position_pct,
                    reason_codes=_reason_codes(sel.reasons),
                    verbal=describe.verbal_caption(descriptor) if descriptor else None,
                    cell_addr=descriptor.cell_id if descriptor else None,
                )
            )
            per_frame_metrics.append(
                {
                    "frame_id": frame_id,
                    "frame_index": sel.index,
                    "timestamp_s": sel.timestamp_s,
                    "global_change_fraction": sel.global_change_fraction,
                    "local_energy": sel.local_energy,
                    "local_area_frac": sel.local_area_frac,
                    "reasons": sel.reasons,
                }
            )
            frame_manifest_entries.append(
                {
                    "id": frame_id,
                    "frame_index": sel.index,
                    "timestamp_s": sel.timestamp_s,
                    "position_pct": position_pct,
                    "file": str(dest_path.relative_to(out_dir)),
                    "reasons": sel.reasons,
                    "zones": [z["id"] for z in zones if z["frame_id"] == frame_id],
                }
            )

        res_str = f"{src_info.width}x{src_info.height}" if src_info.width else "?"
        fps_str = f"{src_info.fps:.2f}" if src_info.fps else "?"
        dur_str = f"{duration_s:.2f}s" if duration_s else "?"
        header_common = [
            f"{source_path.name}  |  {dur_str}  |  {res_str}@{fps_str}fps  |  sampler={args.sampler} budget={args.budget}",
        ]
        sheet_paths = sheet.render_contact_sheets(
            cells,
            out_dir,
            header_common=header_common,
            font_path=Path(args.font) if args.font else DEFAULT_FONT,
            caption_style=args.caption_style,
            per_page=args.per_page,
            cols=args.cols,
            rows=args.rows,
        )

        timeline_path = timeline.render_timeline(
            scores, selected, out_dir / "timeline.png", font_path=Path(args.font) if args.font else DEFAULT_FONT,
            duration_s=duration_s,
        )

        tool_versions = provenance.tool_versions()
        source_dict = {
            "kind": src_info.kind,
            "name": source_path.name,
            "path": str(source_path),
            "sha256": src_info.sha256,
            "duration_s": src_info.duration_s,
            "fps": src_info.fps,
            "width": src_info.width,
            "height": src_info.height,
            "frame_count_analyzed": total_frames,
            "codec_name": src_info.codec_name,
            "is_vfr": src_info.is_vfr,
        }
        selection_dict = {
            "method": args.sampler,
            "budget": args.budget,
            "uniform_reserved": resolved_uniform_reserved,
            "selected_count": len(selected),
            "analyzed_count": total_frames,
        }
        artifacts = {
            "timeline": str(timeline_path.relative_to(out_dir)),
            "metrics": "metrics.json",
            "report": "report.md",
        }
        sheets_manifest = [{"file": str(p.relative_to(out_dir))} for p in sheet_paths]

        # Re-probe and re-hash the source now, compare to the hash taken
        # before any processing started: a real check, not an assumed
        # `True`. Cheap relative to everything else this run already did
        # (one extra read pass over the source).
        final_source_check = provenance.probe_source(source_path)
        source_unchanged = final_source_check.sha256 == src_info.sha256
        if not source_unchanged:
            warnings.append(
                "source file hash changed during this run: results may not correspond to "
                "the file currently on disk"
            )

        met = manifest_mod.build_metrics(
            per_frame=per_frame_metrics,
            zones=zones,
            thresholds={
                "global_delta_threshold": args.global_threshold,
                "local_delta_threshold": args.local_threshold,
                "min_event_spacing_s": args.min_event_spacing,
            },
            warnings=warnings,
        )
        # Written before manifest.json, not after: manifest.json's
        # postconditions.metrics_written checks the file actually exists,
        # so the file has to exist first.
        manifest_mod.write_json(out_dir / "metrics.json", met)

        man = manifest_mod.build_manifest(
            tool_versions=tool_versions,
            source=source_dict,
            selection=selection_dict,
            frames=frame_manifest_entries,
            sheets=sheets_manifest,
            artifacts=artifacts,
            warnings=warnings,
            command=" ".join(sys.argv),
            # vars(args) also contains "func" (argparse's dispatch callable,
            # set via set_defaults). json.dump(..., default=str) turns that
            # into "<function cmd_inspect at 0x...>" — a memory address that
            # differs between processes even for the exact same invocation,
            # which is a real determinism bug in a manifest whose whole
            # point is reproducibility. Drop it; it is not a parameter.
            parameters={k: v for k, v in vars(args).items() if k != "func"},
            out_dir=out_dir,
            source_unchanged=source_unchanged,
        )
        manifest_mod.write_json(out_dir / "manifest.json", man)

        quick_index = [
            f"{len(selected)} frames selected out of {total_frames} analyzed ({args.sampler}, budget {args.budget})",
            f"contact sheet: {', '.join(p.name for p in sheet_paths)}",
            f"timeline: {timeline_path.name}",
        ]
        if zones:
            quick_index.append(f"{len(zones)} local candidate zones detected")
        manifest_mod.write_report(
            out_dir / "report.md",
            quick_index=quick_index,
            observations=[f"Source: {source_path.name} ({src_info.kind})", f"Duration: {dur_str}, {res_str}@{fps_str}"],
            measurements=[
                f"{frame_manifest_entries[i]['id']}: t={s.timestamp_s:.2f}s global={s.global_change_fraction:.3f} local={s.local_energy:.3f} reasons={s.reasons}"
                for i, s in enumerate(selected)
            ],
            candidates=[f"Zone {z['id']}: {z['direction']} {z.get('dominant_hue') or '?'} cell={z['cell']}" for z in zones],
            recommendations=["Use `zoom` on frames with reason local_change to verify the event at full resolution."],
            limits=warnings or ["No warnings recorded for this run."],
        )

        print(f"video-inspect: {len(selected)}/{total_frames} frame -> {out_dir}")
        for p in sheet_paths:
            print(f"  sheet: {p}")
        print(f"  timeline: {timeline_path}")
        print(f"  manifest: {out_dir / 'manifest.json'}")
        return 0
    finally:
        shutil.rmtree(tmp_analysis, ignore_errors=True)
        if tmp_thumbs is not None:
            shutil.rmtree(tmp_thumbs, ignore_errors=True)


def _cell_to_bbox(cell_id: str) -> tuple[float, float, float, float]:
    valid_cols = describe.COL_LETTERS[: describe.GRID_COLS]
    col_letter = cell_id[:1].upper()
    row_part = cell_id[1:]
    if col_letter not in valid_cols or not row_part.isdigit():
        provenance.fail(
            f"invalid --cell '{cell_id}': expected a column in {list(valid_cols)} "
            f"followed by a row in 1..{describe.GRID_ROWS} (e.g. 'B2')"
        )
    row_num = int(row_part)
    if not (1 <= row_num <= describe.GRID_ROWS):
        provenance.fail(
            f"invalid --cell '{cell_id}': row {row_num} out of range, valid rows are 1..{describe.GRID_ROWS}"
        )
    col = describe.COL_LETTERS.index(col_letter)
    row = row_num - 1
    w = 1.0 / describe.GRID_COLS
    h = 1.0 / describe.GRID_ROWS
    return (col * w, row * h, w, h)


def _parse_bbox(raw: str) -> tuple[float, float, float, float]:
    parts = raw.split(",")
    if len(parts) != 4:
        provenance.fail(f"invalid --bbox '{raw}': expected 4 comma-separated values x,y,w,h")
    try:
        x, y, w, h = (float(v) for v in parts)
    except ValueError:
        provenance.fail(f"invalid --bbox '{raw}': all 4 values must be numbers")
    if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
        provenance.fail(f"invalid --bbox '{raw}': x and y must be normalized coordinates in [0, 1]")
    if not (w > 0.0 and h > 0.0):
        provenance.fail(f"invalid --bbox '{raw}': width and height must be positive")
    if x + w > 1.0 or y + h > 1.0:
        provenance.fail(f"invalid --bbox '{raw}': x+w and y+h must not exceed 1.0")
    return (x, y, w, h)


def cmd_zoom(args: argparse.Namespace) -> int:
    run_dir = Path(args.run).resolve()
    man_path = run_dir / "manifest.json"
    if not man_path.exists():
        provenance.fail(f"manifest.json not found in {run_dir}; pass the output dir of a previous `inspect`")
    man = json.loads(man_path.read_text())
    source_path = Path(man["source"]["path"])
    src_width = man["source"]["width"]
    src_height = man["source"]["height"]

    center_time = args.at
    if args.frame:
        match = next((f for f in man["frames"] if f["id"] == args.frame), None)
        if not match:
            provenance.fail(f"frame id not found in manifest: {args.frame}")
        center_time = match["timestamp_s"]
    if center_time is None:
        provenance.fail("zoom needs --frame F0xx or --at SECONDS")

    if args.bbox:
        bbox = _parse_bbox(args.bbox)
    elif args.cell:
        bbox = _cell_to_bbox(args.cell)
    else:
        bbox = (0.3, 0.3, 0.4, 0.4)
        print("video-inspect zoom: no --bbox/--cell given, using a default centered box", file=sys.stderr)

    out_dir = Path(args.output) if args.output else run_dir / f"zoom-{args.frame or f'{center_time:.2f}s'}"
    frames = extract.extract_crop_burst(
        source_path,
        center_time,
        bbox,
        src_width,
        src_height,
        out_dir,
        window_s=args.window,
        frame_count=args.frames,
        padding_frac=args.padding,
    )

    zoom_manifest = {
        "source_frame": args.frame,
        "center_time_s": center_time,
        "bbox_norm": list(bbox),
        "window_s": args.window,
        "frames": [{"timestamp_s": t, "file": str(p.relative_to(out_dir))} for t, p in frames],
    }
    manifest_mod.write_json(out_dir / "zoom_manifest.json", zoom_manifest)

    print(f"video-inspect zoom: {len(frames)} frame -> {out_dir}")
    for t, p in frames:
        print(f"  {t:.3f}s  {p}")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    import cv2
    import numpy as np
    from PIL import Image, ImageDraw

    before = Path(args.before).resolve()
    after = Path(args.after).resolve()
    out_dir = Path(args.output).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    info_before = provenance.probe_video(before)
    info_after = provenance.probe_video(after)
    positions = [float(p) for p in args.positions.split(",")]

    def grab_frame(path: Path, t: float, width: int, out: Path) -> Path:
        import subprocess

        cmd = [
            "ffmpeg", "-y", "-v", "error", "-ss", f"{t:.3f}", "-i", str(path),
            "-frames:v", "1", "-vf", f"scale='min({width}\\,iw)':-2:flags=area", str(out),
        ]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if proc.returncode != 0 or not out.exists():
            print(
                f"video-inspect compare: frame grab failed at t={t:.3f}s for {path.name}: "
                f"{proc.stderr.decode('utf-8', 'replace').strip().splitlines()[-1:] }",
                file=sys.stderr,
            )
        return out

    for w in env_check.optional_warnings(DEFAULT_FONT):
        print(f"video-inspect: {w}", file=sys.stderr)
    font = env_check.load_font(DEFAULT_FONT, 13)
    rows_img = []
    diff_scores = []
    tmp_dir = out_dir / "frames"
    tmp_dir.mkdir(exist_ok=True)

    for i, pct in enumerate(positions):
        # PNG, not JPEG: the mjpeg encoder can reject an exact end-of-stream
        # seek ("Non full-range YUV is non-standard") on some sources. PNG
        # has no such color-range constraint, and these are diagnostic
        # stills, not a size-sensitive output.
        # Also clamp away from the exact duration, which some containers
        # cannot seek to (no frame is presented exactly at EOF).
        eps = 0.04
        tb = min((pct / 100) * (info_before.duration_s or 0), max(0.0, (info_before.duration_s or 0) - eps))
        ta = min((pct / 100) * (info_after.duration_s or 0), max(0.0, (info_after.duration_s or 0) - eps))
        fb = grab_frame(before, tb, args.width, tmp_dir / f"before_{i:02d}.png")
        fa = grab_frame(after, ta, args.width, tmp_dir / f"after_{i:02d}.png")

        img_b = cv2.imread(str(fb))
        img_a = cv2.imread(str(fa))
        if img_b is None or img_a is None:
            diff_scores.append((pct, None))
            continue
        h = min(img_b.shape[0], img_a.shape[0])
        w = min(img_b.shape[1], img_a.shape[1])
        img_b_r, img_a_r = img_b[:h, :w], img_a[:h, :w]
        gray_b = cv2.cvtColor(img_b_r, cv2.COLOR_BGR2GRAY)
        gray_a = cv2.cvtColor(img_a_r, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray_b, gray_a)
        mean_diff = float(diff.mean())
        diff_scores.append((pct, mean_diff))
        diff_color = cv2.applyColorMap(diff, cv2.COLORMAP_INFERNO)
        diff_path = tmp_dir / f"diff_{i:02d}.jpg"
        cv2.imwrite(str(diff_path), diff_color)

        triplet = Image.new("RGB", (w * 3 + 20, h + 24), (255, 255, 255))
        triplet.paste(Image.open(fb).resize((w, h)), (0, 20))
        triplet.paste(Image.open(fa).resize((w, h)), (w + 10, 20))
        triplet.paste(Image.fromarray(cv2.cvtColor(diff_color, cv2.COLOR_BGR2RGB)).resize((w, h)), (2 * w + 20, 20))
        draw = ImageDraw.Draw(triplet)
        draw.text((0, 2), f"{pct:.0f}%  before t={tb:.2f}s | after t={ta:.2f}s | diff mean={mean_diff:.1f}", fill=(0, 0, 0), font=font)
        rows_img.append(triplet)

    if rows_img:
        total_h = sum(r.height for r in rows_img) + (len(rows_img) - 1) * 8
        max_w = max(r.width for r in rows_img)
        sheet_img = Image.new("RGB", (max_w, total_h), (245, 245, 245))
        y = 0
        for r in rows_img:
            sheet_img.paste(r, (0, y))
            y += r.height + 8
        sheet_path = out_dir / "compare-sheet.png"
        sheet_img.save(sheet_path)
    else:
        sheet_path = None

    ranked = sorted([d for d in diff_scores if d[1] is not None], key=lambda d: d[1], reverse=True)
    manifest_mod.write_json(
        out_dir / "manifest.json",
        {
            "schema_version": manifest_mod.SCHEMA_VERSION,
            "tool": {"name": "video-inspect", "version": provenance.tool_versions()["video_inspect"]},
            "before": {"path": str(before), "duration_s": info_before.duration_s},
            "after": {"path": str(after), "duration_s": info_after.duration_s},
            "positions_pct": positions,
            "diff_scores": [{"position_pct": p, "mean_abs_diff": d} for p, d in diff_scores],
            "ranked_by_difference": [{"position_pct": p, "mean_abs_diff": d} for p, d in ranked],
            "sheet": str(sheet_path.name) if sheet_path else None,
        },
    )
    print(f"video-inspect compare: {len(positions)} positions compared -> {out_dir}")
    if sheet_path:
        print(f"  sheet: {sheet_path}")
    if ranked:
        print(f"  largest difference at: {ranked[0][0]:.0f}% (mean_abs_diff={ranked[0][1]:.1f})")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="video-inspect", description="Deterministic evidence packages for agentic video understanding.")
    sub = p.add_subparsers(dest="command", required=True)

    p_inspect = sub.add_parser("inspect", help="Build a contact sheet + manifest for a video or frame directory.")
    p_inspect.add_argument("source")
    p_inspect.add_argument("--output", required=True)
    p_inspect.add_argument("--budget", type=int, default=24)
    p_inspect.add_argument(
        "--uniform-reserved", type=int, default=None,
        help="frames reserved for uniform coverage out of --budget; default scales with the "
        "budget (max(2, budget // 3)) instead of a fixed number, so low budgets keep at least "
        "some event-detection slots",
    )
    p_inspect.add_argument("--sampler", choices=["uniform", "scene", "hybrid"], default="hybrid")
    p_inspect.add_argument(
        "--caption-style", choices=["codes", "verbal", "none"], default="verbal",
        help="verbal (default): localizes the event (direction/hue/cell) without the reader "
        "having to infer position from a small pixel. codes stays available for a terser, "
        "more technical output.",
    )
    p_inspect.add_argument("--analysis-width", type=int, default=320)
    p_inspect.add_argument("--thumb-width", type=int, default=480)
    p_inspect.add_argument("--cols", type=int, default=4)
    p_inspect.add_argument("--rows", type=int, default=3)
    p_inspect.add_argument("--per-page", type=int, default=12)
    p_inspect.add_argument("--global-threshold", type=float, default=25.0)
    p_inspect.add_argument("--local-threshold", type=float, default=45.0)
    p_inspect.add_argument("--min-event-spacing", type=float, default=0.35)
    p_inspect.add_argument(
        "--min-caption-energy", type=float, default=0.1,
        help="minimum local_energy (0..1) required before a local-change frame gets a zone/color "
        "caption, instead of just its reason code; local_energy reaches 1.0 at 2x --local-threshold, "
        "so this floor is independent of and much lower than a full threshold crossing. Content-"
        "dependent: on one clean, low-noise render, measured noise stayed <=0.018 and real signal "
        ">=0.069 (a clear gap), so 0.02-0.03 lost nothing there and caught more of a fading event's "
        "tail than the 0.1 default does; on noisy real-world footage (moving grass, continuous camera "
        "motion) local_energy can sit far higher than that even without a real local event, so the "
        "default stays conservative until calibrated on more than one content type. Raise it if "
        "captions still look like noise; lower it on clean/static-background content if real events "
        "are going undescribed",
    )
    p_inspect.add_argument("--max-analysis-frames", type=int, default=None)
    p_inspect.add_argument("--assume-fps", type=float, default=None, help="for frame-directory sources only")
    p_inspect.add_argument("--font", default=None)
    p_inspect.set_defaults(func=cmd_inspect)

    p_zoom = sub.add_parser("zoom", help="Crop a burst from the ORIGINAL source around a frame/region.")
    p_zoom.add_argument("run", help="output directory of a previous `inspect` run")
    p_zoom.add_argument("--frame", default=None, help="frame id from manifest.json, e.g. F007")
    p_zoom.add_argument("--at", type=float, default=None, help="explicit timestamp in seconds")
    p_zoom.add_argument("--bbox", default=None, help="x,y,w,h normalized 0..1")
    p_zoom.add_argument("--cell", default=None, help="spatial cell id, e.g. C4")
    p_zoom.add_argument("--window", type=float, default=1.0)
    p_zoom.add_argument("--frames", type=int, default=9)
    p_zoom.add_argument("--padding", type=float, default=0.25)
    p_zoom.add_argument("--output", default=None)
    p_zoom.set_defaults(func=cmd_zoom)

    p_compare = sub.add_parser("compare", help="Relative-time comparison between two videos (before/after).")
    p_compare.add_argument("before")
    p_compare.add_argument("after")
    p_compare.add_argument("--output", required=True)
    p_compare.add_argument("--positions", default="0,10,25,50,75,90,100")
    p_compare.add_argument("--width", type=int, default=480)
    p_compare.set_defaults(func=cmd_compare)

    p_doctor = sub.add_parser(
        "doctor", help="Check required/optional dependencies (ffmpeg, numpy, opencv, Pillow, bundled font)."
    )
    p_doctor.set_defaults(func=cmd_doctor)

    return p


def cmd_doctor(args: argparse.Namespace) -> int:
    ok, report = env_check.doctor_report()
    print(report)
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command != "doctor":
        # Fail fast and clearly here, not with an ImportError three frames
        # deep once `inspect`/`zoom`/`compare` gets around to using cv2 or
        # Pillow — the whole point of a stated dependency contract.
        env_check.fail_if_missing_required()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
