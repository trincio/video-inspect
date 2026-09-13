"""Contact sheet rendering with Pillow.

A "universal" profile sized to stay well inside a typical VLM's
lower-resolution image tier (long edge under ~1568px covers Claude's
standard tier, for example), one caption line per cell in a band BELOW the
thumbnail (never overlaid on the pixels), and a header carrying everything
that would otherwise repeat per cell.

Caption style (`codes` / `verbal` / `none`) is a CLI parameter, not a fixed
choice baked into the layout.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .env_check import load_font

CANVAS_WIDTH = 1440
HEADER_HEIGHT = 64
MARGIN = 16
GAP = 12
CAPTION_HEIGHT = 30
DEFAULT_COLS = 4
DEFAULT_ROWS = 3
BG_COLOR = (245, 245, 245)
CELL_BG = (20, 20, 20)
TEXT_COLOR = (20, 20, 20)
HEADER_BG = (255, 255, 255)
CAPTION_BG = (255, 255, 255)
BORDER_COLOR = (210, 210, 210)


@dataclass
class CellData:
    frame_id: str
    image_path: Path
    timestamp_s: float
    position_pct: float
    reason_codes: list[str]
    verbal: str | None = None
    cell_addr: str | None = None


def _font(font_path: Path | None, size: int) -> ImageFont.ImageFont:
    return load_font(font_path, size)


def _fmt_time(seconds: float) -> str:
    if seconds < 0:
        return "--:--.-"
    m = int(seconds // 60)
    s = seconds - m * 60
    return f"{m:02d}:{s:05.2f}"


def caption_text(cell: CellData, style: str) -> str:
    codes = "+".join(cell.reason_codes) if cell.reason_codes else "-"
    if style == "codes":
        return f"{cell.frame_id}  {_fmt_time(cell.timestamp_s)}  {cell.position_pct:4.0f}%  {codes}"
    if style == "verbal":
        # Same fields as "codes" (id/time/position keep the frame locatable
        # even with no event); only the trailing tag is replaced by a short
        # phrase when there is something to describe, so a frame with no
        # local event still carries position_pct instead of losing it.
        tail = cell.verbal if cell.verbal else codes
        return f"{cell.frame_id}  {_fmt_time(cell.timestamp_s)}  {cell.position_pct:4.0f}%  {tail}"
    if style == "none":
        return ""
    raise ValueError(f"unknown caption style: {style}")


def paginate(cells: list[CellData], per_page: int) -> list[list[CellData]]:
    return [cells[i : i + per_page] for i in range(0, len(cells), per_page)] or [[]]


def render_page(
    cells: list[CellData],
    out_path: Path,
    *,
    header_lines: list[str],
    font_path: Path,
    caption_style: str = "codes",
    cols: int = DEFAULT_COLS,
    rows: int = DEFAULT_ROWS,
    canvas_width: int = CANVAS_WIDTH,
) -> Path:
    has_caption = caption_style != "none"
    caption_h = CAPTION_HEIGHT if has_caption else 0
    inner_w = canvas_width - 2 * MARGIN - (cols - 1) * GAP
    cell_w = inner_w // cols
    cell_h = int(cell_w * 9 / 16)
    row_h = cell_h + caption_h
    canvas_h = HEADER_HEIGHT + MARGIN + rows * row_h + (rows - 1) * GAP + MARGIN

    canvas = Image.new("RGB", (canvas_width, canvas_h), BG_COLOR)
    draw = ImageDraw.Draw(canvas)

    draw.rectangle([0, 0, canvas_width, HEADER_HEIGHT], fill=HEADER_BG)
    header_font = _font(font_path, 15)
    y = 8
    for line in header_lines:
        draw.text((MARGIN, y), line, fill=TEXT_COLOR, font=header_font)
        y += 18
    draw.line([0, HEADER_HEIGHT, canvas_width, HEADER_HEIGHT], fill=BORDER_COLOR, width=1)

    caption_font = _font(font_path, 13)

    for idx, cell in enumerate(cells):
        r, c = divmod(idx, cols)
        x0 = MARGIN + c * (cell_w + GAP)
        y0 = HEADER_HEIGHT + MARGIN + r * (row_h + GAP)

        thumb_box = (cell_w, cell_h)
        cell_canvas = Image.new("RGB", thumb_box, CELL_BG)
        try:
            img = Image.open(cell.image_path).convert("RGB")
            img.thumbnail(thumb_box, Image.LANCZOS)
            paste_x = (cell_w - img.width) // 2
            paste_y = (cell_h - img.height) // 2
            cell_canvas.paste(img, (paste_x, paste_y))
        except (FileNotFoundError, OSError):
            pass
        canvas.paste(cell_canvas, (x0, y0))
        draw.rectangle([x0, y0, x0 + cell_w - 1, y0 + cell_h - 1], outline=BORDER_COLOR, width=1)

        if has_caption:
            cap_y0 = y0 + cell_h
            draw.rectangle([x0, cap_y0, x0 + cell_w - 1, cap_y0 + caption_h - 1], fill=CAPTION_BG, outline=BORDER_COLOR)
            text = caption_text(cell, caption_style)
            draw.text((x0 + 4, cap_y0 + 7), text, fill=TEXT_COLOR, font=caption_font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, "PNG")
    return out_path


def render_contact_sheets(
    cells: list[CellData],
    out_dir: Path,
    *,
    header_common: list[str],
    font_path: Path,
    caption_style: str = "codes",
    per_page: int = 12,
    cols: int = DEFAULT_COLS,
    rows: int = DEFAULT_ROWS,
) -> list[Path]:
    pages = paginate(cells, per_page)
    paths = []
    for i, page_cells in enumerate(pages, start=1):
        header = [*header_common, f"page {i}/{len(pages)}  |  caption style: {caption_style}"]
        out_path = out_dir / f"sheet-{i:03d}.png"
        paths.append(
            render_page(
                page_cells,
                out_path,
                header_lines=header,
                font_path=font_path,
                caption_style=caption_style,
                cols=cols,
                rows=rows,
            )
        )
    return paths
