"""IDSSE degradation on a small invented DFL-format match; the real dataset is not needed."""

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

from scripts import degrade_idsse
from solution.tracking import load_tracking

ROOT = Path(__file__).resolve().parent.parent
MATCH = "TST001"
HALF = 400


def write_match(root):
    """Two halves of HALF frames, 4 players per team, one stoppage per half."""
    rng = np.random.default_rng(1)
    root.mkdir(parents=True)
    (root / f"DFL_02_01_matchinformation_DFL-COM-000001_DFL-MAT-{MATCH}.xml").write_text(
        '<?xml version="1.0"?><PutDataRequest><Teams>'
        '<Team TeamId="GUEST" TeamName="Guest FC" Role="guest"/>'
        '<Team TeamId="HOME" TeamName="Home FC" Role="home"/></Teams></PutDataRequest>', encoding="utf-8")
    body = ['<MetaData Type="pitch-size"><PitchSize X="105.00" Y="68.00"/></MetaData>']
    for section, first in (("firstHalf", 10000), ("secondHalf", 100000)):
        frames = range(first, first + HALF)
        t = np.arange(HALF)
        ball = np.c_[20 * np.sin(t / 40), 10 * np.cos(t / 50)]
        status = np.where((t >= 150) & (t < 220), 0, 1)
        possession = np.where(t < 100, 1, np.where(t < 300, 2, 1))
        body.append(f'<FrameSet GameSection="{section}" TeamId="referee" PersonId="REF">'
                    + "".join(f'<Frame N="{n}" X="0" Y="0"/>' for n in frames) + "</FrameSet>")
        for team, sign in (("HOME", -1), ("GUEST", 1)):
            for k in range(4):
                base = np.array([sign * (5 + 8 * k), -20 + 12 * k])
                path = base + ball * 0.5 + rng.normal(0, 0.05, (HALF, 2)).cumsum(axis=0)
                body.append(f'<FrameSet GameSection="{section}" TeamId="{team}" PersonId="DFL-OBJ-{team}{k}">'
                            + "".join(f'<Frame N="{n}" X="{x:.2f}" Y="{y:.2f}" S="1" M="1"/>'
                                      for n, (x, y) in zip(frames, path)) + "</FrameSet>")
        body.append(f'<FrameSet GameSection="{section}" TeamId="BALL" PersonId="DFL-OBJ-BALL">'
                    + "".join(f'<Frame N="{n}" X="{x:.2f}" Y="{y:.2f}" Z="{1.5 if i % 90 < 10 else 0}" '
                              f'BallPossession="{p}" BallStatus="{s}"/>'
                              for i, (n, (x, y), p, s) in enumerate(zip(frames, ball, possession, status)))
                    + "</FrameSet>")
    (root / f"DFL_04_03_positions_raw_observed_DFL-COM-000001_DFL-MAT-{MATCH}.xml").write_text(
        '<?xml version="1.0"?><PutDataRequest><Positions>' + "".join(body) + "</Positions></PutDataRequest>",
        encoding="utf-8")


CONFIG = degrade_idsse.Config(seed=3, total_frames=1000, frames_before=50, frames_halftime=100,
                              mean_objects_graded=7.0)


class DegradeIdsseTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        root = Path(cls.temp.name)
        write_match(root / "data")
        cls.manifest, cls.files = degrade_idsse.generate(MATCH, root / "out", CONFIG, data_dir=root / "data")

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_competition_layout_and_project_reader(self):
        players, ball, teams = load_tracking(self.files["atd_meta"], self.files["atd_raw"], CONFIG.total_frames)
        self.assertEqual(players.shape, (1000, 46, 2))
        self.assertEqual(list(np.bincount(teams)), [23, 23])
        self.assertTrue(np.isnan(players[:50]).all() and np.isnan(ball[:50]).all())
        seen = ball[~np.isnan(ball[:, 0])]
        self.assertTrue(((seen > -0.2) & (seen < 1.2)).all(), "coordinates must be normalised")

    def test_official_baseline_reads_it(self):
        sys.path.insert(0, str(ROOT / "official" / "starter"))
        sys.path.insert(0, str(ROOT / "official"))
        import baseline
        slots = baseline.team_of_each_track(str(self.files["atd_meta"]))
        codes = baseline.nearest_player_labels(str(self.files["atd_raw"]), slots, 1000, (105.0, 68.0))
        self.assertEqual(len(slots), 46)
        self.assertTrue((codes[:50] == baseline.NOBODY).all())
        self.assertTrue((codes != baseline.NOBODY).any())

    def test_labels_only_in_ground_truth(self):
        lines = self.files["gt"].read_text(encoding="utf-8").splitlines()
        self.assertEqual(lines[0], "frame,label")
        labels = [line.split(",")[1] for line in lines[1:]]
        self.assertEqual(len(labels), 1000)
        first, second = self.manifest["graded_blocks"]
        self.assertEqual(first, [50, 50 + HALF - 1])
        self.assertEqual(second[0], 50 + HALF + 100)
        self.assertTrue(all(label == "X" for label in labels[:50] + labels[first[1] + 1:second[0]]))
        self.assertEqual(labels[50:50 + 100], ["H"] * 100)
        self.assertEqual(set(labels[50 + 150:50 + 220]), {"D"})
        raw = self.files["atd_raw"].read_text(encoding="ascii")
        meta = self.files["atd_meta"].read_text(encoding="utf-8")
        for leak in ("DFL-OBJ", "HOME0", "BallStatus", "BallPossession"):
            self.assertNotIn(leak, raw + meta)

    def test_frozen_dead_ball_removed_except_restart(self):
        _, ball, _ = load_tracking(self.files["atd_meta"], self.files["atd_raw"], CONFIG.total_frames)
        dead = ball[50 + 150:50 + 220, 0]
        shown = np.flatnonzero(~np.isnan(dead))
        if len(shown):
            self.assertEqual(shown[-1], len(dead) - 1, "only the end of a stoppage may show the ball")
            self.assertTrue((np.diff(shown) == 1).all())

    def test_calibration_and_determinism(self):
        checks = self.manifest["checks"]
        self.assertAlmostEqual(checks["ball_missing_graded"], CONFIG.ball_missing_graded, delta=0.05)
        self.assertAlmostEqual(checks["mean_objects_graded"], CONFIG.mean_objects_graded, delta=0.5)
        with tempfile.TemporaryDirectory() as other:
            again, _ = degrade_idsse.generate(MATCH, Path(other), CONFIG, data_dir=Path(self.temp.name) / "data")
        self.assertEqual(again["sha256"], self.manifest["sha256"])


if __name__ == "__main__":
    unittest.main()
