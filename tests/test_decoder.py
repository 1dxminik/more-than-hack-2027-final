"""Behaviour of the offline decoder and X filling; every observation here is invented."""

import unittest
from unittest.mock import patch

import numpy as np

import phase1
from solution.decoder import DecoderConfig, decode
from solution.possession import fill_abstentions


PITCH = np.array([105.3, 68.0])
TEAMS = np.array([0, 0, 1, 1])
NEAR = np.array([0.5, 0.1]) / PITCH


def scene(frames=1000, fps=25, moving=True):
    """Home owns the ball throughout; players jog, so activity says the ball is alive."""
    time = np.arange(frames)[:, None] / fps
    base = np.array([[30., 30.], [40., 45.], [70., 30.], [60., 15.]])
    players = np.tile(base, (frames, 1, 1))
    if moving:
        players[:, :, 0] += 12 * np.sin(0.25 * time)
        players[:, :, 1] += 12 * np.cos(0.25 * time)
    players = players / PITCH
    return players, players[:, 0, :] + NEAR


def spells(labels):
    changes = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    bounds = np.r_[0, changes, len(labels)]
    return [(labels[a], b - a) for a, b in zip(bounds, bounds[1:])]


class DecoderTests(unittest.TestCase):
    def test_no_abstention_inside_a_block_and_x_outside(self):
        players, ball = scene()
        ball[200:600] = np.nan
        labels = decode(players, ball, TEAMS, blocks=((100, 899),))
        self.assertNotIn("X", labels[100:900])
        self.assertTrue(np.all(labels[:100] == "X"))
        self.assertTrue(np.all(labels[900:] == "X"))

    def test_long_gap_in_live_play_keeps_the_team(self):
        players, ball = scene()
        ball[200:700] = np.nan  # 20 s, far beyond the two seconds of the causal memory
        labels = decode(players, ball, TEAMS)
        self.assertTrue(np.all(labels == "H"))

    def test_future_frames_label_the_start_of_a_block(self):
        players, ball = scene()
        ball[:400] = np.nan
        labels = decode(players, ball, TEAMS)
        self.assertTrue(np.all(labels == "H"))

    def test_short_opponent_contact_creates_no_possession(self):
        players, ball = scene()
        ball[500:520] = players[500:520, 2] + NEAR  # 0.8 s
        labels = decode(players, ball, TEAMS)
        self.assertTrue(np.all(labels == "H"))

    def test_sustained_change_is_located_close_to_the_truth(self):
        players, ball = scene()
        ball[500:] = players[500:, 2] + NEAR
        labels = decode(players, ball, TEAMS)
        self.assertEqual([letter for letter, _ in spells(labels)], ["H", "A"])
        self.assertLessEqual(abs(spells(labels)[0][1] - 500), 2)

    def test_change_hidden_by_a_gap_lands_in_the_middle_of_the_gap(self):
        players, ball = scene()
        ball[400:600] = np.nan
        ball[600:] = players[600:, 2] + NEAR
        pattern = spells(decode(players, ball, TEAMS))
        self.assertEqual([letter for letter, _ in pattern], ["H", "A"])
        self.assertEqual(pattern[0][1], 500)

    def test_still_players_and_distant_still_ball_are_dead_then_restart(self):
        players, ball = scene()
        players[400:700] = players[399]
        ball[400:700] = [0.95, 0.95]
        ball[700:] = players[700:, 2] + NEAR
        labels = decode(players, ball, TEAMS)
        self.assertEqual([letter for letter, _ in spells(labels)], ["H", "D", "A"])
        self.assertTrue(np.all(labels[450:680] == "D"))

    def test_still_players_with_ball_at_feet_are_not_dead(self):
        players, ball = scene(moving=False)
        ball = ball + 1e-3 * np.sin(np.arange(len(ball)))[:, None]  # small touches: a moving ball
        labels = decode(players, ball, TEAMS)
        self.assertNotIn("D", labels)

    def test_minimum_lengths_hold_inside_the_block(self):
        rng = np.random.default_rng(0)
        players, ball = scene(frames=2000)
        for start in rng.integers(0, 1900, 60):
            ball[start:start + 40] = players[start:start + 40, 2] + NEAR
        config = DecoderConfig(switch_cost_seconds=0.1)
        for letter, length in spells(decode(players, ball, TEAMS, config=config))[1:-1]:
            self.assertGreaterEqual(length, 50 if letter in "HA" else 25)

    def test_blocks_do_not_share_information(self):
        players, ball = scene()
        ball[500:] = players[500:, 2] + NEAR
        together = decode(players, ball, TEAMS, blocks=((0, 499), (500, 999)))
        np.testing.assert_array_equal(together[:500], decode(players[:500], ball[:500], TEAMS))
        np.testing.assert_array_equal(together[500:], decode(players[500:], ball[500:], TEAMS))

    def test_track_order_does_not_change_the_answer(self):
        players, ball = scene()
        ball[500:] = players[500:, 2] + NEAR
        order = np.array([3, 1, 0, 2])
        np.testing.assert_array_equal(decode(players, ball, TEAMS),
                                      decode(players[:, order], ball, TEAMS[order]))

    def test_time_settings_scale_with_fps(self):
        for fps in (25, 50):
            with self.subTest(fps=fps):
                players, ball = scene(frames=40 * fps, fps=fps)
                ball[20 * fps:] = players[20 * fps:, 2] + NEAR
                ball[10 * fps:10 * fps + fps // 2] = players[10 * fps:10 * fps + fps // 2, 2] + NEAR
                pattern = spells(decode(players, ball, TEAMS, fps=fps))
                self.assertEqual([letter for letter, _ in pattern], ["H", "A"])

    def test_tracking_outage_never_becomes_stillness_or_a_crash(self):
        players, ball = scene()
        players[300:500] = np.nan
        ball[300:500] = np.nan
        labels = decode(players, ball, TEAMS)
        self.assertTrue(np.all(labels == "H"))

    def test_is_deterministic_and_reads_no_files(self):
        players, ball = scene()
        ball[300:600] = np.nan
        with patch("builtins.open", side_effect=AssertionError("Inference may not read files")):
            first = decode(players, ball, TEAMS)
        np.testing.assert_array_equal(first, decode(players, ball, TEAMS))

    def test_invalid_settings_are_rejected(self):
        players, ball = scene(frames=10)
        with self.assertRaises(ValueError):
            DecoderConfig(switch_cost_seconds=0)
        with self.assertRaises(ValueError):
            DecoderConfig(quiet_speed_mps=3, busy_speed_mps=2)
        with self.assertRaises(ValueError):
            decode(players, ball, TEAMS, fps=0)
        with self.assertRaises(ValueError):
            decode(players, ball, TEAMS, blocks=((0, 10),))


class FillTests(unittest.TestCase):
    def test_matches_the_official_fill_and_keeps_spell_score(self):
        pm = phase1.metrics()
        gt = pm.label_codes(np.array(list("XHHHHHDDDDAAAAAAXXHHHHDDAAAX")))
        hidden = np.array(list("XXHXHXDXXDXAXAXAXXXHXHDXAXAX"))
        runs = pm.valid_runs(gt)
        blocks = [(first, stop - 1) for first, stop in runs]
        filled = fill_abstentions(hidden, blocks)
        np.testing.assert_array_equal(pm.label_codes(filled)[gt != pm.X],
                                      pm.fill_abstentions(pm.label_codes(hidden), runs)[gt != pm.X])
        before = pm.evaluate(gt, pm.label_codes(hidden))
        after = pm.evaluate(gt, pm.label_codes(filled))
        self.assertEqual(before["spell_f1_mean"], after["spell_f1_mean"])
        self.assertGreater(after["macro_f1"], before["macro_f1"])
        self.assertEqual(after["n_abstain_on_graded"], 0)

    def test_block_without_labels_and_frames_outside_blocks_stay_x(self):
        labels = np.array(list("XXXHXX"))
        np.testing.assert_array_equal(fill_abstentions(labels, ((0, 1), (3, 4))), np.array(list("XXXHHX")))


if __name__ == "__main__":
    unittest.main()
