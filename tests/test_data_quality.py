"""The quality report must find exactly the faults injected into invented tracking."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from scripts import data_quality as dq


N_FRAMES = 3000
BASES = [(0.2, 0.2), (0.2, 0.4), (0.2, 0.6), (0.2, 0.8), (0.8, 0.2), (0.8, 0.4), (0.8, 0.6), (0.8, 0.8)]
XML = ('<Root><Team id="h"/><Team id="a"/>'
       + "".join(f'<Player teamId="{"h" if slot < 4 else "a"}"/>' for slot in range(8)) + "</Root>")


def fixture():
    """Still players, ball next to slot 0, and a known number of each fault."""
    players = np.tile(np.asarray(BASES, dtype=float), (N_FRAMES, 1, 1))
    ball = players[:, 0, :] + (0.005, 0.002)
    ball[100:125] = np.nan                # 1 s gap
    ball[500:750] = np.nan                # 10 s gap
    ball[1000] = (0.5, 0.5)               # false detection: 2 jumps
    ball[1200:1210] = (1.2, 0.5)          # outside the pitch: 2 more jumps
    players[1500:, 1, 0] += 0.1           # identity switch: 1 jump
    players[2000:2050, 3] = players[2000:2050, 2]  # 2 s duplicate: 2 jumps
    players[2200:2210, 3] = players[2200:2210, 2]  # too short to count as duplicate: 2 jumps
    players[2500:2600, 7] = np.nan        # one fragmented track
    return players, ball


def write_atd(folder, players, ball, line_filter=lambda line: line):
    xml, raw = Path(folder) / "m.xml", Path(folder) / "m.txt"
    xml.write_text(XML, encoding="utf-8")
    with raw.open("w", encoding="utf-8", newline="\n") as stream:
        for frame in range(len(players)):
            groups = ";".join(f"{x},{y},0" for x, y in players[frame])
            stream.write(line_filter(f"{frame + 1}:{groups}:{ball[frame, 0]},{ball[frame, 1]},0") + "\n")
    return xml, raw


class FaultDetectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        cls.xml, cls.raw = write_atd(cls.temp.name, *fixture())
        cls.report = dq.analyze(cls.xml, cls.raw, "test", ((0, 1499),), (105.0, 68.0), 25)

    def test_ball_faults_are_counted_exactly(self):
        ball = self.report["all"]["ball"]
        self.assertEqual(ball["present_frames"], N_FRAMES - 275)
        self.assertEqual(ball["gap_count"], 2)
        self.assertEqual(ball["gap_max_s"], 10.0)
        self.assertEqual(ball["jump_count"], 4)
        self.assertEqual(ball["out_of_pitch_pct"], round(100 * 10 / (N_FRAMES - 275), 2))
        self.assertEqual(ball["missing_share"]["from_5_to_30s"], round(100 * 250 / 275, 2))
        self.assertEqual(self.report["all"]["nearest"]["within_control_pct"], round(100 * (N_FRAMES - 275 - 11) / (N_FRAMES - 275), 2))

    def test_player_faults_are_counted_exactly(self):
        scope = self.report["all"]
        self.assertEqual(scope["jumps"]["count"], 5)
        self.assertEqual(scope["duplicates"]["events"], 1)
        self.assertEqual(scope["duplicates"]["frames_pct"], round(100 * 50 / N_FRAMES, 2))
        self.assertEqual(scope["segments"]["count"], 9)
        self.assertEqual(scope["segments"]["slots_used"], 8)
        self.assertEqual(scope["tracks"]["none_pct"], 0.0)

    def test_scope_only_counts_its_own_frames(self):
        graded = self.report["graded"]
        self.assertEqual(graded["frames"], 1500)
        self.assertEqual(graded["ball"]["present_frames"], 1225)
        self.assertEqual(graded["jumps"]["count"], 0)
        self.assertEqual(graded["duplicates"]["events"], 0)

    def test_report_holds_aggregates_only(self):
        with tempfile.TemporaryDirectory() as out:
            dq.write_report(self.report, out)
            text = (Path(out) / "quality_report.md").read_text(encoding="utf-8")
            stored = json.loads((Path(out) / "quality_report.json").read_text(encoding="utf-8"))
        self.assertIn("Piłka widoczna", text)
        self.assertEqual(stored, self.report)
        for private in (str(self.raw), self.raw.name, "0.205"):
            self.assertNotIn(private, text + json.dumps(stored))

    def test_published_count_mismatch_is_reported(self):
        report = dq.analyze(self.xml, self.raw, "test", None, (105.0, 68.0), 25,
                            expected=(("load.records", N_FRAMES), ("all.ball.present_frames", 1)))
        self.assertEqual([check["ok"] for check in report["reference_checks"]], [True, False])
        with tempfile.TemporaryDirectory() as out:
            dq.write_report(report, out)
            self.assertIn("NIEZGODNE", (Path(out) / "quality_report.md").read_text(encoding="utf-8"))


class CommandTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_malformed_lines_are_counted_and_two_value_groups_are_read(self):
        players, ball = fixture()
        marker = "SECRETVALUE"
        xml, raw = write_atd(self.root, players[:100], ball[:100],
                             lambda line: marker if line.startswith("50:") else line.replace(",0;", ";"))
        report = dq.analyze(xml, raw, "test", None, (105.0, 68.0), 25)
        self.assertEqual(report["load"]["malformed_lines"], 1)
        self.assertEqual(report["all"]["ball"]["present_frames"], 99)
        self.assertNotIn(marker, json.dumps(report))

    def test_errors_do_not_repeat_file_content(self):
        xml = self.root / "m.xml"
        xml.write_text("<Root>SECRETVALUE", encoding="utf-8")
        script = Path(dq.__file__).resolve()
        result = subprocess.run(
            [sys.executable, "-I", str(script), "analyze", "--atd-xml", str(xml),
             "--atd-raw", str(xml), "--out", str(self.root / "out")],
            cwd=self.root, capture_output=True, text=True)
        self.assertEqual(result.returncode, 1)
        self.assertNotIn("SECRETVALUE", result.stdout + result.stderr)
        self.assertFalse((self.root / "out").exists())

    def test_copied_script_runs_alone_and_compares_reports(self):
        players, ball = fixture()
        xml, raw = write_atd(self.root, players, ball)
        copy = self.root / "data_quality.py"
        copy.write_bytes(Path(dq.__file__).read_bytes())
        for name in ("a", "b"):
            subprocess.run([sys.executable, "-I", str(copy), "analyze", "--atd-xml", str(xml), "--atd-raw", str(raw),
                            "--graded", "0-1499", "--name", name, "--out", str(self.root / name)],
                           cwd=self.root, check=True, capture_output=True)
        subprocess.run([sys.executable, "-I", str(copy), "compare", str(self.root / "a/quality_report.json"),
                        str(self.root / "b/quality_report.json"), "--out", str(self.root / "c")],
                       cwd=self.root, check=True, capture_output=True)
        text = (self.root / "c/quality_comparison.md").read_text(encoding="utf-8")
        self.assertIn("| Miara | a | b |", text)
        self.assertIn("a: oceniane klatki", text)


if __name__ == "__main__":
    unittest.main()
