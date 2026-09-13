"""Environment checks: fail clearly on missing REQUIRED tools, warn and fall
back on OPTIONAL ones. A required component missing must stop the run with
an actionable message; an optional one missing must degrade gracefully
with a stated fallback, never crash.
"""
from __future__ import annotations

import shutil
from pathlib import Path

REQUIRED_BINARIES = ["ffmpeg", "ffprobe"]
REQUIRED_MODULES = ["numpy", "cv2", "PIL"]

INSTALL_HINTS = {
    "ffmpeg": "brew install ffmpeg   (Linux: apt/yum install ffmpeg)",
    "ffprobe": "incluso nel pacchetto ffmpeg",
    "numpy": "python3 -m pip install numpy",
    "cv2": "python3 -m pip install opencv-python",
    "PIL": "python3 -m pip install pillow",
}

MODULE_DISPLAY = {"cv2": "opencv-python (cv2)", "PIL": "Pillow (PIL)"}


def _module_version(name: str, module) -> str:
    return getattr(module, "__version__", "?")


def missing_required() -> list[tuple[str, str]]:
    """Returns a list of (kind, name) for every missing required component."""
    missing = []
    for b in REQUIRED_BINARIES:
        if shutil.which(b) is None:
            missing.append(("binario", b))
    for m in REQUIRED_MODULES:
        try:
            __import__(m)
        except ImportError:
            missing.append(("libreria Python", m))
    return missing


def fail_if_missing_required() -> None:
    missing = missing_required()
    if not missing:
        return
    lines = ["video-inspect: componenti fondamentali mancanti, impossibile procedere:"]
    for kind, name in missing:
        display = MODULE_DISPLAY.get(name, name)
        lines.append(f"  - {kind} '{display}' non trovato. Installa con: {INSTALL_HINTS.get(name, '?')}")
    raise SystemExit("\n".join(lines))


def find_bundled_font() -> Path | None:
    """Two candidate depths, to work both from the project layout
    (src/video_inspect/ -> ../../fonts) and the shared-skill layout
    (scripts/video_inspect/ -> ../fonts), without hardcoding either.
    """
    here = Path(__file__).resolve().parent
    candidates = [
        here.parent / "fonts" / "DejaVuSansMono.ttf",
        here.parent.parent / "fonts" / "DejaVuSansMono.ttf",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def optional_warnings(font_path: Path | None) -> list[str]:
    """Optional components: report what is actually being used, never crash."""
    warnings = []
    if font_path is None or not font_path.exists():
        warnings.append(
            "font incluso (DejaVu Sans Mono) non trovato: uso PIL.ImageFont.load_default() "
            "come fallback (didascalie meno leggibili, nessun accento)"
        )
    return warnings


def load_font(font_path: Path | None, size: int):
    """Truetype font if available, PIL's built-in default otherwise. Never
    raises: a missing/corrupt font degrades legibility, it must not crash
    a run that is otherwise working.
    """
    from PIL import ImageFont

    if font_path is not None:
        try:
            return ImageFont.truetype(str(font_path), size)
        except OSError:
            pass
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def doctor_report() -> tuple[bool, str]:
    lines = ["video-inspect doctor", "", "Fondamentali (obbligatori):"]
    ok = True
    for b in REQUIRED_BINARIES:
        found = shutil.which(b) is not None
        ok = ok and found
        suffix = "" if found else f"  -> installa con: {INSTALL_HINTS.get(b, '?')}"
        lines.append(f"  [{'OK' if found else 'MANCANTE'}] {b}{suffix}")
    for m in REQUIRED_MODULES:
        display = MODULE_DISPLAY.get(m, m)
        try:
            mod = __import__(m)
            found = True
            ver = _module_version(m, mod)
        except ImportError:
            found = False
            ver = None
        ok = ok and found
        suffix = f" (versione {ver})" if found else f"  -> installa con: {INSTALL_HINTS.get(m, '?')}"
        lines.append(f"  [{'OK' if found else 'MANCANTE'}] {display}{suffix}")

    lines.append("")
    lines.append("Opzionali (con fallback se mancanti):")
    font_path = find_bundled_font()
    if font_path is not None:
        lines.append(f"  [OK] font incluso: {font_path}")
    else:
        lines.append(
            "  [FALLBACK] font DejaVu Sans Mono non trovato -> uso il font di default di Pillow "
            "(didascalie meno leggibili, nessun accento)"
        )

    lines.append("")
    lines.append("ESITO: pronto all'uso." if ok else "ESITO: mancano componenti fondamentali, vedi sopra.")
    return ok, "\n".join(lines)
