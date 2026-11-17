"""Behavioral edge cases for the first algorithm; all observations are invented."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import phase1
from scripts.make_step1_synthetic import generate_step1, verify_step1
from scripts.make_synthetic import METADATA, TRACKING
from solution.possession import Config, predict
from solution.tracking import load_tracking


PITCH = np.array([105.3, 68.0])
TEAMS = np.array([0, 0, 1, 1])


def scene(frames=350, fps=25, moving=True):
    players = np.tile(np.array([[30., 30.], [40., 45.], [70., 30.], [60., 15.]]), (frames, 1, 1))
    if moving:
        players[:, :, 0] += np.arange(frames)[:, None] / fps
    ball = players[:, 0, :] + [0.5, 0.1]
    return players / PITCH, ball / PITCH


class PossessionTests(unittest.TestCase):
    def test_short_occlusion_keeps_team(self):
        players, ball = scene()
        ball[100:125] = np.nan
        labels, _ = predict(players, ball, TEAMS)
        self.assertTrue(np.all(labels[100:125] == "H"))

    def test_long_occlusion_expires_to_unknown_not_dead(self):
        players, ball = scene()
        ball[100:250] = np.nan
        labels, _ = predict(players, ball, TEAMS)
        self.assertTrue(np.all(labels[160:250] == "X"))
        self.assertNotIn("D", labels)
        self.assertTrue(np.all(labels[255:] == "H"))

    def test_no_initial_observation_does_not_invent_owner(self):
        players, ball = scene()
        ball[:100] = np.nan
        labels, _ = predict(players, ball, TEAMS)
        self.assertTrue(np.all(labels[:100] == "X"))

    def test_sustained_missing_ball_and_still_players_then_restart(self):
        players, ball = scene()
        players[100:225] = players[99]
        ball[100:225] = np.nan
        ball[225:] = players[225:, 2] + [0.5 / PITCH[0], 0.1 / PITCH[1]]
        labels, _ = predict(players, ball, TEAMS)
        self.assertTrue(np.all(labels[100:120] == "H"))
        self.assertTrue(np.all(labels[150:225] == "D"))
        self.assertTrue(np.all(labels[230:] == "A"))

    def test_visible_stationary_distant_ball_can_support_dead(self):
        players, ball = scene(moving=False)
        ball[:] = [0.95, 0.95]
        labels, _ = predict(players, ball, TEAMS)
        self.assertTrue(np.all(labels[50:] == "D"))

    def test_stationary_live_ball_near_player_is_not_dead(self):
        players, ball = scene(moving=False)
        labels, _ = predict(players, ball, TEAMS)
        self.assertTrue(np.all(labels[10:] == "H"))
        self.assertNotIn("D", labels)

    def test_missing_players_are_not_evidence_of_stillness(self):
        players, ball = scene(moving=False)
        players[:, 1:] = np.nan
        ball[:] = np.nan
        labels, _ = predict(players, ball, TEAMS)
        self.assertTrue(np.all(labels == "X"))

    def test_full_tracking_outage_does_not_preserve_stale_dead_state(self):
        players, ball = scene(frames=500, moving=False)
        ball[:100] = [0.95, 0.95]
        players[100:300] = np.nan
        ball[100:300] = np.nan
        labels, _ = predict(players, ball, TEAMS)
        self.assertEqual(labels[99], "D")
        self.assertTrue(np.all(labels[100:304] == "X"))
        self.assertTrue(np.all(labels[310:] == "H"))

    def test_short_opponent_touch_does_not_switch_team(self):
        players, ball = scene()
        ball[100:103] = players[100:103, 2] + [0.5 / PITCH[0], 0.1 / PITCH[1]]
        labels, _ = predict(players, ball, TEAMS)
        self.assertTrue(np.all(labels[90:120] == "H"))

    def test_time_thresholds_scale_with_fps(self):
        for fps in (25, 50):
            with self.subTest(fps=fps):
                players, ball = scene(frames=7 * fps, fps=fps)
                ball[2 * fps:6 * fps] = np.nan
                labels, _ = predict(players, ball, TEAMS, fps=fps)
                self.assertEqual(labels[3 * fps], "H")
                self.assertEqual(labels[5 * fps], "X")

    def test_inference_is_causal_and_does_not_read_files(self):
        players, ball = scene()
        ball[100:200] = np.nan
        with patch("builtins.open", side_effect=AssertionError("Inference may not read files")):
            complete, _ = predict(players, ball, TEAMS)
            prefix, _ = predict(players[:175], ball[:175], TEAMS)
        np.testing.assert_array_equal(complete[:175], prefix)

    def test_invalid_parameters_are_rejected(self):
        players, ball = scene()
        with self.assertRaises(ValueError):
            Config(hold_seconds=-1)
        with self.assertRaises(ValueError):
            predict(players, ball, TEAMS, fps=0)


class Step1IntegrationTests(unittest.TestCase):
    def test_new_suite_is_deterministic_and_contains_counterexamples(self):
        with tempfile.TemporaryDirectory() as folder:
            one, two = Path(folder) / "one", Path(folder) / "two"
            first = generate_step1(one, 3000)
            second = generate_step1(two, 3000)
            self.assertEqual(first, second)
            self.assertEqual(first, verify_step1(one))
            players, ball, teams = load_tracking(one / METADATA, one / TRACKING, 3000)
            labels, _ = predict(players, ball, teams)
            self.assertEqual(len(labels), 3000)
            self.assertEqual(set(labels), set("HADX"))
            import json
            cases = json.loads((one / "synthetic_cases.json").read_text())
            names = {case["name"] for case in cases}
            self.assertTrue({"ambiguous_live_missing", "unobserved_change", "dead_visible_far"} <= names)

    def test_environment_rejects_unapproved_data_location(self):
        with self.assertRaisesRegex(ValueError, "synthetic"):
            phase1.synthetic_environment(phase1.ROOT / "data/do-not-open")

    def test_loader_rejects_misaligned_frames(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            generate_step1(root, 3000)
            path = root / TRACKING
            data = path.read_text()
            path.write_text("2:" + data.split(":", 1)[1])
            with self.assertRaisesRegex(ValueError, "frames"):
                load_tracking(root / METADATA, path, 3000)


if __name__ == "__main__":
    unittest.main()
