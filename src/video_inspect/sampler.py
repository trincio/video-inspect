"""Deterministic frame scoring and hybrid budget selection.

No neural networks, no embeddings. Two cheap signals computed on
low-resolution analysis frames, broadly in the spirit of multi-channel
dedup approaches used by other open-source video-for-LLM tools (e.g.
claude-real-video: https://github.com/HUANGCHIHHUNGLeo/claude-real-video):

- global_change: coarse 16x16 signature, fraction of cells that moved a lot.
  Catches cuts, big camera motion, large content changes.
- local_energy: finer 32x32 grid, magnitude of the single most-changed
  region. Catches a small object moving against an otherwise static frame,
  which global_change alone would drown out. A sampler that only picks the
  most "representative" frames tends to starve exactly these rare, small,
  or brief events — which is usually what you actually want to find when
  debugging.

Both channels are FACTs (measurements); "this is an event worth a frame"
is the one INFERENCE this module makes, and it is fully driven by the
threshold/weights below, not by a model.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .extract import AnalysisFrame


@dataclass
class FrameScore:
    index: int
    timestamp_s: float
    global_change_fraction: float
    local_energy: float
    global_cell_signature: np.ndarray = field(repr=False)
    local_peak_rc: tuple[int, int] | None = None
    local_grid_shape: tuple[int, int] | None = None
    local_area_frac: float = 0.0


def _grid_signature(gray: np.ndarray, grid: int) -> np.ndarray:
    h, w = gray.shape
    cell_h, cell_w = max(1, h // grid), max(1, w // grid)
    trimmed = gray[: cell_h * grid, : cell_w * grid]
    reshaped = trimmed.reshape(grid, cell_h, grid, cell_w)
    return reshaped.mean(axis=(1, 3))  # grid x grid mean intensity


def score_frames(
    frames: list[AnalysisFrame],
    *,
    global_grid: int = 16,
    local_grid: int = 32,
    global_delta_threshold: float = 25.0,
    local_delta_threshold: float = 45.0,
) -> list[FrameScore]:
    import cv2

    scores: list[FrameScore] = []
    prev_global: np.ndarray | None = None
    prev_local: np.ndarray | None = None
    for frame in frames:
        img = cv2.imread(str(frame.path), cv2.IMREAD_COLOR)
        if img is None:
            scores.append(
                FrameScore(frame.index, frame.timestamp_s, 0.0, 0.0, np.zeros((global_grid, global_grid)))
            )
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        g_sig = _grid_signature(gray, global_grid)
        l_sig = _grid_signature(gray, local_grid)

        peak_rc = None
        local_area_frac = 0.0
        if prev_global is None:
            global_change = 0.0
            local_energy = 0.0
        else:
            g_delta = np.abs(g_sig - prev_global)
            global_change = float((g_delta >= global_delta_threshold).mean())
            l_delta = np.abs(l_sig - prev_local)
            # local_energy: the single hottest cell, normalized to [0,1] by
            # the threshold — a small object crossing one cell should read
            # as a strong local event even if the rest of the frame is dead.
            local_energy = float(min(1.0, l_delta.max() / (local_delta_threshold * 2)))
            local_area_frac = float((l_delta >= local_delta_threshold).mean())
            if l_delta.max() > 0:
                peak_rc = tuple(int(v) for v in np.unravel_index(np.argmax(l_delta), l_delta.shape))

        scores.append(
            FrameScore(
                frame.index,
                frame.timestamp_s,
                global_change,
                local_energy,
                g_sig,
                local_peak_rc=peak_rc,
                local_grid_shape=l_sig.shape,
                local_area_frac=local_area_frac,
            )
        )
        prev_global, prev_local = g_sig, l_sig
    return scores


@dataclass
class SelectedFrame:
    index: int
    timestamp_s: float
    reasons: list[str]
    global_change_fraction: float
    local_energy: float
    local_peak_rc: tuple[int, int] | None = None
    local_grid_shape: tuple[int, int] | None = None
    local_area_frac: float = 0.0


def hybrid_select(
    scores: list[FrameScore],
    *,
    budget: int = 24,
    uniform_reserved: int | None = None,
    min_event_spacing_s: float = 0.35,
    local_weight: float = 0.45,
) -> list[SelectedFrame]:
    """Uniform coverage reserve + ranked event candidates, deduplicated by a
    minimum time spacing. Always includes first and last frame (they anchor
    the uniform reserve) so the sheet never silently starts mid-action.
    """
    if not scores:
        return []
    if len(scores) <= budget:
        return [
            SelectedFrame(
                s.index,
                s.timestamp_s,
                ["all"],
                s.global_change_fraction,
                s.local_energy,
                s.local_peak_rc,
                s.local_grid_shape,
                s.local_area_frac,
            )
            for s in scores
        ]

    uniform_reserved = uniform_reserved if uniform_reserved is not None else max(2, budget // 3)
    event_slots = max(0, budget - uniform_reserved)

    duration_index = len(scores) - 1
    uniform_positions = np.linspace(0, duration_index, uniform_reserved)
    uniform_indices = sorted({int(round(p)) for p in uniform_positions})

    reasons: dict[int, list[str]] = {i: ["uniform"] for i in uniform_indices}

    # Rank remaining frames by combined score, skip the very first frame
    # (score is always 0.0 there — no predecessor to diff against).
    # local_weight=0 makes this a pure global/scene-change ranking (used by
    # --sampler scene): passing uniform_reserved=0 alone was not enough to
    # make "scene" mode scene-only, since candidates were still ranked by
    # the combined global+local score either way.
    global_weight = 1.0 - local_weight
    candidates = [s for s in scores if s.index != 0]
    candidates.sort(
        key=lambda s: global_weight * s.global_change_fraction + local_weight * s.local_energy,
        reverse=True,
    )

    # Dedup spacing applies only among EVENT picks, not against uniform
    # anchors: uniform frames exist for coverage, not to gate the event
    # budget. A long-duration event (e.g. an object crossing the frame for
    # a full second) has high scores on many consecutive analysis frames;
    # comparing candidates against uniform-anchor times too would starve
    # every event slot whenever an anchor happened to fall inside the
    # event window, discarding a genuinely distinct candidate instead of
    # keeping it.
    selected_event_times: list[float] = []
    picked = 0
    for cand in candidates:
        if picked >= event_slots:
            break
        if cand.global_change_fraction <= 0 and cand.local_energy <= 0:
            continue
        ts = cand.timestamp_s
        tag = "scene" if cand.global_change_fraction >= cand.local_energy else "local_change"
        if cand.index in reasons:
            # exact same frame as an existing pick (typically a uniform
            # anchor): merge the reason instead of skipping, but this does
            # not consume an event slot — the frame is already shown.
            if tag not in reasons[cand.index]:
                reasons[cand.index].append(tag)
            continue
        if any(abs(ts - t) < min_event_spacing_s for t in selected_event_times):
            continue
        reasons[cand.index] = [tag]
        selected_event_times.append(ts)
        picked += 1

    # Backfill unused event slots into the largest remaining temporal gap,
    # so a quiet video still gets even coverage instead of wasted budget.
    if picked < event_slots:
        chosen_indices = sorted(reasons.keys())
        while picked < event_slots:
            gaps = []
            for a, b in zip(chosen_indices, chosen_indices[1:]):
                gaps.append((scores[b].timestamp_s - scores[a].timestamp_s, a, b))
            if not gaps:
                break
            gaps.sort(reverse=True)
            _, a, b = gaps[0]
            mid = (a + b) // 2
            if mid in reasons or mid <= a or mid >= b:
                break
            reasons[mid] = ["uniform_fill"]
            chosen_indices = sorted(reasons.keys())
            picked += 1

    result = [
        SelectedFrame(
            i,
            scores[i].timestamp_s,
            reasons[i],
            scores[i].global_change_fraction,
            scores[i].local_energy,
            scores[i].local_peak_rc,
            scores[i].local_grid_shape,
            scores[i].local_area_frac,
        )
        for i in sorted(reasons.keys())
    ]
    return result
