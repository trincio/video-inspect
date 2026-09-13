#!/usr/bin/env python3
"""End-to-end and unit tests. The subprocess-level tests use ffmpeg-generated
fixtures; requires ffmpeg, NumPy, OpenCV and Pillow (same as the tool
itself). The pure-Python tests import the package directly."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).parents[1]
LAUNCHER = ROOT / "src" / "video_inspect" / "__main__.py"
sys.path.insert(0, str(ROOT / "src"))

from video_inspect import describe, extract, sampler  # noqa: E402


def make_hard_cut(path: Path, w: int = 320, h: int = 180, fps: int = 30) -> None:
    subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error",
            "-f", "lavfi", "-i", f"color=c=red:s={w}x{h}:d=1:r={fps}",
            "-f", "lavfi", "-i", f"color=c=blue:s={w}x{h}:d=1:r={fps}",
            "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0[v]",
            "-map", "[v]", "-pix_fmt", "yuv420p", str(path),
        ],
        check=True,
    )


class DirectionWordTests(unittest.TestCase):
    """direction_word() must reach every cell of its 3x4 table, including
    the rightmost column (NE/E/SE) — a prior version silently remapped
    column index 3 down to 2, making that whole column unreachable."""

    def test_all_twelve_cells_are_reachable(self) -> None:
        rows, cols = 3, 4
        seen = set()
        for r in range(rows):
            for c in range(cols):
                seen.add(describe.direction_word(r, c, rows, cols))
        self.assertEqual(seen, {"NW", "N", "NE", "W", "C", "E", "SW", "S", "SE"})

    def test_rightmost_column_is_east_leaning(self) -> None:
        self.assertEqual(describe.direction_word(0, 3, 3, 4), "NE")
        self.assertEqual(describe.direction_word(1, 3, 3, 4), "E")
        self.assertEqual(describe.direction_word(2, 3, 3, 4), "SE")


class CellBboxTests(unittest.TestCase):
    def test_valid_cell_within_grid(self) -> None:
        from video_inspect.cli import _cell_to_bbox

        x, y, w, h = _cell_to_bbox("B2")
        self.assertGreaterEqual(x, 0.0)
        self.assertGreaterEqual(y, 0.0)
        self.assertLessEqual(x + w, 1.0)
        self.assertLessEqual(y + h, 1.0)

    def test_out_of_range_row_is_rejected(self) -> None:
        # Grid is GRID_COLS x GRID_ROWS = 4 x 3 (columns A-D, rows 1-3):
        # "C4" names a row that does not exist and must be rejected, not
        # silently turned into a nonsense rectangle at the bottom edge.
        from video_inspect.cli import _cell_to_bbox

        with self.assertRaises(SystemExit):
            _cell_to_bbox("C4")


class BboxParseTests(unittest.TestCase):
    def test_valid_bbox(self) -> None:
        from video_inspect.cli import _parse_bbox

        self.assertEqual(_parse_bbox("0.1,0.2,0.3,0.4"), (0.1, 0.2, 0.3, 0.4))

    def test_bbox_extending_past_edge_is_rejected(self) -> None:
        from video_inspect.cli import _parse_bbox

        with self.assertRaises(SystemExit):
            _parse_bbox("0.8,0.8,0.5,0.5")

    def test_bbox_wrong_field_count_is_rejected(self) -> None:
        from video_inspect.cli import _parse_bbox

        with self.assertRaises(SystemExit):
            _parse_bbox("0.1,0.2,0.3")


class SamplerRankingTests(unittest.TestCase):
    """--sampler scene must rank by global change only; a prior version
    used the same combined global+local score as hybrid regardless of
    uniform_reserved, so "scene" behaved like hybrid with no uniform
    coverage instead of an actual scene/global-change-only ranking."""

    def test_local_weight_zero_ignores_local_energy_in_ranking(self) -> None:
        scores = [
            sampler.FrameScore(0, 0.0, 0.0, 0.0, __import__("numpy").zeros((2, 2))),
            sampler.FrameScore(1, 1.0, 0.1, 0.9, __import__("numpy").zeros((2, 2))),  # high local, low global
            sampler.FrameScore(2, 2.0, 0.9, 0.1, __import__("numpy").zeros((2, 2))),  # high global, low local
        ]
        # budget=1 (a single event slot) forces the ranking to actually
        # choose between the two candidates instead of fitting both.
        selected = sampler.hybrid_select(scores, budget=1, uniform_reserved=0, local_weight=0.0)
        picked_indices = {s.index for s in selected}
        # With local_weight=0, frame 2 (high global) must outrank frame 1
        # (high local) for the single available event slot.
        self.assertIn(2, picked_indices)
        self.assertNotIn(1, picked_indices)


class ShouldDescribeTests(unittest.TestCase):
    """A second dogfooding run (a real smooth Manim render) found that the
    FIRST fix for the false-color-caption bug was
    itself an overcorrection: gating on local_area_frac > 0 turned out
    mathematically equivalent to requiring local_energy >= 0.5, which
    suppressed captions on every single frame of that video — including
    ones with a genuine, correctly colored local event — because a smooth,
    anti-aliased render never made any single grid cell cross the full
    --local-threshold. The fix uses local_energy's continuous value with a
    low floor (--min-caption-energy, default 0.1) instead."""

    def _frame(self, local_energy: float, local_area_frac: float = 0.0) -> sampler.SelectedFrame:
        return sampler.SelectedFrame(
            index=1,
            timestamp_s=1.0,
            reasons=["local_change"],
            global_change_fraction=0.0,
            local_energy=local_energy,
            local_peak_rc=(1, 1),
            local_grid_shape=(4, 4),
            local_area_frac=local_area_frac,
        )

    def test_gentle_real_signal_below_hard_threshold_is_still_described(self) -> None:
        from video_inspect.cli import _should_describe

        # local_energy=0.3 with local_area_frac=0.0: no grid cell fully
        # crossed --local-threshold, but the peak delta is well above
        # noise level — exactly the reported regression case.
        sel = self._frame(local_energy=0.3, local_area_frac=0.0)
        self.assertTrue(_should_describe(sel, min_caption_energy=0.1))

    def test_near_zero_noise_is_not_described(self) -> None:
        from video_inspect.cli import _should_describe

        sel = self._frame(local_energy=0.01, local_area_frac=0.0)
        self.assertFalse(_should_describe(sel, min_caption_energy=0.1))

    def test_floor_is_configurable(self) -> None:
        from video_inspect.cli import _should_describe

        sel = self._frame(local_energy=0.3, local_area_frac=0.0)
        self.assertFalse(_should_describe(sel, min_caption_energy=0.5))


class GlobalChangeFadeLimitationTests(unittest.TestCase):
    """Documents a real, verified limitation (found via external
    dogfooding, see README "Known limitations"): global_change_fraction
    compares each frame only to the one immediately before it, so a slow,
    smooth fade never crosses the per-cell threshold at any single step
    even though the start and end frames are completely different. This
    test pins the current (expected, not desired) behavior so a future
    change to the scoring window is a deliberate decision, not a silent
    regression discovered by a user again.
    """

    def test_smooth_fade_never_registers_as_global_change(self) -> None:
        temp = tempfile.TemporaryDirectory()
        try:
            root = Path(temp.name)
            video = root / "fade.mp4"
            subprocess.run(
                [
                    "ffmpeg", "-y", "-v", "error",
                    "-f", "lavfi", "-i", "color=c=black:s=320x180:d=1:r=30",
                    "-f", "lavfi", "-i", "color=c=white:s=320x180:d=1:r=30",
                    "-filter_complex", "[0:v][1:v]xfade=transition=fade:duration=1:offset=0[v]",
                    "-map", "[v]", "-pix_fmt", "yuv420p", str(video),
                ],
                check=True,
            )
            frames = extract.extract_analysis_frames(video, root / "analysis", width=320)
            scores = sampler.score_frames(frames)
            self.assertEqual(max(s.global_change_fraction for s in scores), 0.0)
        finally:
            temp.cleanup()


class ExtractAnalysisFramesTests(unittest.TestCase):
    """A prior version appended -frames:v AFTER the output path in the
    ffmpeg command; ffmpeg binds an output option to the output URL that
    FOLLOWS it, so the cap silently had no effect."""

    def test_max_frames_actually_caps_output(self) -> None:
        temp = tempfile.TemporaryDirectory()
        try:
            root = Path(temp.name)
            video = root / "v.mp4"
            subprocess.run(
                [
                    "ffmpeg", "-y", "-v", "error", "-f", "lavfi",
                    "-i", "color=c=red:s=64x64:d=2:r=30", "-pix_fmt", "yuv420p", str(video),
                ],
                check=True,
            )
            full = extract.extract_analysis_frames(video, root / "full")
            capped = extract.extract_analysis_frames(video, root / "capped", max_frames=10)
            self.assertGreater(len(full), 10)
            self.assertEqual(len(capped), 10)
        finally:
            temp.cleanup()


class VideoInspectSubprocessTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.video = self.root / "hard_cut.mp4"
        make_hard_cut(self.video)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def invoke(self, *arguments: str) -> subprocess.CompletedProcess:
        env = {"PYTHONPATH": str(ROOT / "src")}
        import os

        return subprocess.run(
            [sys.executable, "-m", "video_inspect", *arguments],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            env={**os.environ, **env},
        )

    def test_doctor_reports_ok_when_deps_present(self) -> None:
        result = self.invoke("doctor")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_inspect_writes_required_artifacts_and_finds_the_cut(self) -> None:
        out = self.root / "run"
        result = self.invoke(
            "inspect", str(self.video), "--output", str(out), "--budget", "6", "--sampler", "hybrid"
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        for name in ("manifest.json", "metrics.json", "report.md", "sheet-001.png", "timeline.png"):
            self.assertTrue((out / name).exists(), f"missing {name}")

        manifest = json.loads((out / "manifest.json").read_text())
        self.assertTrue(manifest["postconditions"]["all_frame_refs_resolve"])
        self.assertTrue(manifest["postconditions"]["source_unchanged"])
        self.assertTrue(manifest["postconditions"]["metrics_written"])

        timestamps = [f["timestamp_s"] for f in manifest["frames"] if f["timestamp_s"] >= 0]
        self.assertTrue(any(abs(t - 1.0) < 0.2 for t in timestamps), timestamps)

    def test_manifest_parameters_do_not_leak_a_function_object(self) -> None:
        # vars(args) includes argparse's "func" dispatch callable; if it
        # leaks into the manifest, json.dump(default=str) turns it into a
        # per-process memory address ("<function cmd_inspect at 0x...>"),
        # which is non-reproducible and breaks the tool's determinism claim.
        out = self.root / "run_params"
        result = self.invoke("inspect", str(self.video), "--output", str(out), "--budget", "6")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        manifest = json.loads((out / "manifest.json").read_text())
        params = manifest["provenance"]["parameters"]
        self.assertNotIn("func", params)
        raw = json.dumps(manifest)
        self.assertNotIn("0x", raw)

    def test_inspect_on_frame_directory_without_assume_fps_does_not_crash(self) -> None:
        # All timestamps are -1 (unknown) in this case; render_timeline()
        # used to call max() on an empty filtered sequence and crash.
        frames_dir = self.root / "frames"
        frames_dir.mkdir()
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-i", str(self.video), str(frames_dir / "f_%03d.png")],
            check=True,
        )
        out = self.root / "run_frames_no_fps"
        result = self.invoke("inspect", str(frames_dir), "--output", str(out), "--budget", "6")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue((out / "timeline.png").exists())

    def test_zones_require_real_local_signal(self) -> None:
        # Found by an external dogfooding run on real footage: frames with
        # near-zero local_energy (no real local event, just noise/
        # antialiasing) were still getting a confident color caption
        # ("verde"/"ciano" that matched nothing visible in the frame),
        # because the caption generator was gated on the relative
        # "local_change" reason tag instead of an absolute signal check.
        # Invariant: a zone must never be reported for a frame whose
        # local_energy is below the configured floor.
        out = self.root / "run_zones_invariant"
        result = self.invoke(
            "inspect", str(self.video), "--output", str(out), "--budget", "10", "--sampler", "hybrid"
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        metrics = json.loads((out / "metrics.json").read_text())
        zoned_frame_ids = {z["frame_id"] for z in metrics["measurements"]["zones"]}
        below_floor_frame_ids = {
            pf["frame_id"] for pf in metrics["measurements"]["per_frame"] if pf["local_energy"] < 0.1
        }
        self.assertEqual(zoned_frame_ids & below_floor_frame_ids, set())

    def test_scene_sampler_does_not_reserve_uniform_coverage(self) -> None:
        out = self.root / "run_scene"
        result = self.invoke(
            "inspect", str(self.video), "--output", str(out), "--budget", "4", "--sampler", "scene"
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        manifest = json.loads((out / "manifest.json").read_text())
        self.assertEqual(manifest["selection"]["uniform_reserved"], 0)

    def test_invalid_budget_fails_with_clear_message(self) -> None:
        out = self.root / "run_bad_budget"
        result = self.invoke("inspect", str(self.video), "--output", str(out), "--budget", "0")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--budget", result.stdout + result.stderr)

    def test_uniform_reserved_exceeding_budget_fails_with_clear_message(self) -> None:
        out = self.root / "run_bad_reserved"
        result = self.invoke(
            "inspect", str(self.video), "--output", str(out),
            "--budget", "4", "--uniform-reserved", "10",
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("uniform-reserved", result.stdout + result.stderr)

    def test_per_page_exceeding_grid_fails_with_clear_message(self) -> None:
        out = self.root / "run_bad_perpage"
        result = self.invoke(
            "inspect", str(self.video), "--output", str(out),
            "--budget", "6", "--rows", "1", "--cols", "1", "--per-page", "4",
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--per-page", result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
