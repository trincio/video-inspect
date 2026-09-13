"""manifest.json / metrics.json / report.md writers.

FACT vs INFERENCE vs RECOMMENDATION are kept in separate, clearly labelled
places, and a missing measurement is `null` + a warning, never silently
omitted.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "0.1"


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def build_manifest(
    *,
    tool_versions: dict,
    source: dict,
    selection: dict,
    frames: list[dict],
    sheets: list[dict],
    artifacts: dict,
    warnings: list[str],
    command: str,
    parameters: dict,
    out_dir: Path,
    source_unchanged: bool,
) -> dict:
    # frames[]/sheets[] store paths relative to out_dir (portable: the run
    # directory can be moved/copied without the manifest lying about where
    # things are), so postconditions below must resolve existence against
    # out_dir, not the caller's current working directory.
    #
    # Every postcondition here is a real check against the filesystem or a
    # value the caller measured, not an assumption: source_unchanged is the
    # result of re-hashing the source and comparing it to the hash taken
    # before the run started (caller's job, passed in); metrics_written
    # checks that metrics.json actually exists on disk at the moment this
    # manifest is built, which requires the caller to write metrics.json
    # BEFORE calling this function, not after.
    return {
        "schema_version": SCHEMA_VERSION,
        "tool": {"name": "video-inspect", "version": tool_versions.get("video_inspect", "0.1.0")},
        "source": source,
        "provenance": {**tool_versions, "command": command, "parameters": parameters},
        "selection": selection,
        "frames": frames,
        "sheets": sheets,
        "artifacts": artifacts,
        "warnings": warnings,
        "postconditions": {
            "source_unchanged": source_unchanged,
            "all_frame_refs_resolve": all((out_dir / f["file"]).exists() for f in frames) if frames else True,
            "all_sheet_refs_resolve": all((out_dir / s["file"]).exists() for s in sheets) if sheets else True,
            "metrics_written": (out_dir / "metrics.json").exists(),
        },
    }


def build_metrics(
    *,
    per_frame: list[dict],
    zones: list[dict],
    thresholds: dict,
    warnings: list[str],
) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "thresholds": thresholds,
        "measurements": {"per_frame": per_frame, "zones": zones},
        "warnings": warnings,
        "postconditions": {},
    }


def write_report(
    path: Path,
    *,
    quick_index: list[str],
    observations: list[str],
    measurements: list[str],
    candidates: list[str],
    recommendations: list[str],
    limits: list[str],
) -> None:
    def section(title: str, lines: list[str]) -> str:
        body = "\n".join(f"- {line}" for line in lines) if lines else "- (nessuna voce)"
        return f"# {title}\n\n{body}\n"

    text = "\n".join(
        [
            section("QUICK INDEX", quick_index),
            section("OBSERVATIONS", observations),
            section("MEASUREMENTS", measurements),
            section("CANDIDATES / INFERENCES", candidates),
            section("RECOMMENDATIONS", recommendations),
            section("LIMITS", limits),
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
