#!/usr/bin/env python3
"""event_recall@budget: does at least one selected frame fall inside each
ground-truth event window (with a small tolerance)? Use this to decide
whether the local-change channel earns its complexity versus a plain
uniform or scene-only sampler on a given corpus.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

TOLERANCE_S = 0.15


def eval_run(manifest_path: Path, ground_truth: dict) -> dict:
    man = json.loads(manifest_path.read_text())
    source_name = Path(man["source"]["name"]).name
    gt = ground_truth.get(source_name)
    if not gt:
        return {"source": source_name, "error": "no ground truth entry"}
    frame_times = [f["timestamp_s"] for f in man["frames"] if f["timestamp_s"] >= 0]
    results = []
    for ev in gt["events"]:
        lo, hi = ev["start_s"] - TOLERANCE_S, ev["end_s"] + TOLERANCE_S
        hits = [t for t in frame_times if lo <= t <= hi]
        results.append(
            {
                "event_type": ev["type"],
                "window": [ev["start_s"], ev["end_s"]],
                "hit": bool(hits),
                "hit_frames": len(hits),
                "closest_frame_offset_s": min((abs(t - (ev["start_s"] + ev["end_s"]) / 2) for t in frame_times), default=None),
            }
        )
    return {
        "source": source_name,
        "selected_count": len(man["frames"]),
        "sampler": man["selection"]["method"],
        "budget": man["selection"]["budget"],
        "events": results,
        "event_recall": sum(1 for r in results if r["hit"]) / len(results) if results else None,
    }


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: eval_recall.py <ground_truth.json> <run_dir> [run_dir...]", file=sys.stderr)
        return 1
    ground_truth = json.loads(Path(argv[0]).read_text())
    for run_dir in argv[1:]:
        man_path = Path(run_dir) / "manifest.json"
        if not man_path.exists():
            print(f"{run_dir}: no manifest.json, skipping")
            continue
        result = eval_run(man_path, ground_truth)
        print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
