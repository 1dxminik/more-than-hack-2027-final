"""TCN features, flips, binning and Viterbi on tiny invented scenes; numpy only, no dataset."""

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

from solution import tcn_features as tf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "official" / "starter"))
import baseline  # noqa: E402  (official, unchanged)


def scene(n=60, seed=0):
    rng = np.random.default_rng(seed)
    players = 0.5 + 0.2 * rng.standard_normal((n, 6, 2)).cumsum(axis=0) * 0.01
    ball = players[:, 0].copy() + 0.002
    ball[20:23] = np.nan
    players[30:35, 4] = np.nan
    teams = np.array([0, 0, 0, 1, 1, 1])
    return players, ball, teams


class Viterbi(unittest.TestCase):
    def test_zero_penalty_is_argmax(self):
        lp = np.log(np.random.default_rng(1).dirichlet(np.ones(3), size=200))
        np.testing.assert_array_equal(tf.viterbi(lp, 0.0), lp.argmax(axis=1))

    def test_short_excursion_removed_long_kept(self):
        p = np.full((100, 3), 0.1)
        p[:, 0] = 0.8
        p[40:42] = [0.3, 0.6, 0.1]          # 2 weak steps of A inside H
        p[70:] = [0.05, 0.9, 0.05]          # a long, confident A
        path = tf.viterbi(np.log(p), 4.0)
        self.assertTrue((path[:70] == 0).all())
        self.assertTrue((path[70:] == 1).all())

    def test_many_matches_single(self):
        lp = np.log(np.random.default_rng(2).dirichlet(np.ones(3), size=300))
        many = tf.viterbi_many(lp, [0, 1.5, 6])
        for row, pen in zip(many, [0, 1.5, 6]):
            np.testing.assert_array_equal(row, tf.viterbi(lp, pen))


class Features(unittest.TestCase):
    def test_finite_and_shape(self):
        x = tf.frame_features(*scene())
        self.assertEqual(x.shape, (60, len(tf.NAMES)))
        self.assertTrue(np.isfinite(x).all())

    def test_slot_order_does_not_matter(self):
        players, ball, teams = scene()
        order = np.array([2, 0, 1, 5, 3, 4])
        np.testing.assert_allclose(tf.frame_features(players, ball, teams),
                                   tf.frame_features(players[:, order], ball, teams[order]), atol=1e-6)

    def test_team_swap_matches_swapped_input(self):
        players, ball, teams = scene()
        x = tf.frame_features(players, ball, teams)
        swapped, y = tf.flip(x, np.array([0, 1, 2, -1] * 15), swap_teams=True)
        np.testing.assert_allclose(swapped, tf.frame_features(players, ball, 1 - teams), atol=1e-6)
        np.testing.assert_array_equal(y[:4], [1, 0, 2, -1])

    def test_mirror_matches_mirrored_input(self):
        players, ball, teams = scene()
        mirrored = players.copy()
        mirrored[..., 0] = 1 - mirrored[..., 0]
        mball = ball.copy()
        mball[:, 0] = 1 - mball[:, 0]
        x, _ = tf.flip(tf.frame_features(players, ball, teams), np.zeros(60), mirror_x=True)
        np.testing.assert_allclose(x, tf.frame_features(mirrored, mball, teams), atol=1e-5)

    def test_short_ball_gap_interpolated(self):
        ball = np.array([[0.0, 0.0], [np.nan, np.nan], [np.nan, np.nan], [0.3, 0.3]] + [[np.nan, np.nan]] * 60
                        + [[0.5, 0.5]])
        filled, seen, interp, _, _ = tf.interpolate_ball(ball, max_gap=50)
        np.testing.assert_allclose(filled[1:3], [[0.1, 0.1], [0.2, 0.2]])
        self.assertTrue(interp[1:3].all() and not interp[4:64].any())
        self.assertTrue(np.isnan(filled[4:64]).all())


class Binning(unittest.TestCase):
    def test_majority_and_ignore(self):
        codes = np.array([0, 0, 1, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2])
        np.testing.assert_array_equal(tf.bin_labels(codes), [0, tf.IGNORE, 2])
        self.assertEqual(len(tf.upsample(np.array([0, 1, 2]), 13)), 13)


class Baseline(unittest.TestCase):
    def test_nearest_player_matches_official(self):
        players, ball, teams = scene()
        with tempfile.TemporaryDirectory() as folder:
            raw = Path(folder) / "raw.txt"

            def group(xy):
                return "NaN,NaN,NaN" if not np.isfinite(xy).all() else f"{float(xy[0])!r},{float(xy[1])!r},0"

            raw.write_text("".join(f"{i + 1}:" + ";".join(group(p) for p in players[i]) + ":" + group(ball[i]) + "\n"
                                   for i in range(len(ball))))
            official = baseline.nearest_player_labels(str(raw), teams, len(ball), (105.0, 68.0))
        np.testing.assert_array_equal(tf.nearest_player_codes(players, ball, teams), official)


if __name__ == "__main__":
    unittest.main()
