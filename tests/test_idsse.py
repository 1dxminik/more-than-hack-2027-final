"""IDSSE reader on a tiny invented file in the DFL format; the real dataset is not needed."""

from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts import idsse


MATCH = "ABC123"
INFORMATION = """<?xml version="1.0"?><PutDataRequest><Teams>
<Team TeamId="GUEST" Role="guest"/><Team TeamId="HOME" Role="home"/></Teams></PutDataRequest>"""


def frame_set(section, team, person, rows):
    return (f'<FrameSet GameSection="{section}" MatchId="M" TeamId="{team}" PersonId="{person}">'
            + "".join(rows) + "</FrameSet>")


def player_rows(frames, x):
    return [f'<Frame N="{n}" T="2023-01-01T00:00:00.000+00:00" X="{x}" Y="0" M="1"/>' for n in frames]


def ball_rows(frames, possession, status):
    return [f'<Frame N="{n}" T="2023-01-01T00:00:{i:02d}.000+00:00" X="0" Y="0" Z="0" M="1" '
            f'BallPossession="{p}" BallStatus="{s}"/>'
            for i, (n, p, s) in enumerate(zip(frames, possession, status))]


def write_fixture(root, second_half_ball=True):
    body = ['<MetaData Type="pitch-size"><PitchSize X="100.00" Y="50.00"/></MetaData>',
            frame_set("firstHalf", "referee", "R", player_rows(range(10, 13), 5)),
            frame_set("firstHalf", "HOME", "P1", player_rows(range(10, 13), -25)),
            frame_set("firstHalf", "GUEST", "P2", player_rows(range(11, 13), 25)),
            frame_set("firstHalf", "BALL", "B", ball_rows(range(10, 13), (1, 2, 2), (1, 1, 0)))]
    if second_half_ball:
        body += [frame_set("secondHalf", "HOME", "P1", player_rows(range(50, 52), -25)),
                 frame_set("secondHalf", "BALL", "B", ball_rows(range(50, 52), (2, 2), (0, 1)))]
    (root / f"DFL_02_01_matchinformation_DFL-COM-000001_DFL-MAT-{MATCH}.xml").write_text(INFORMATION)
    (root / f"DFL_04_03_positions_raw_observed_DFL-COM-000001_DFL-MAT-{MATCH}.xml").write_text(
        "<PutDataRequest><Positions>" + "".join(body) + "</Positions></PutDataRequest>")


class ReaderTests(unittest.TestCase):
    def test_labels_positions_and_blocks(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write_fixture(root)
            self.assertEqual(idsse.match_ids(root), [MATCH])
            match = idsse.load_match(MATCH, root)
            np.testing.assert_array_equal(match.labels, [idsse.H, idsse.A, idsse.D, idsse.D, idsse.A])
            self.assertEqual(match.blocks, [(0, 2), (3, 4)])
            self.assertEqual(match.pitch, (100.0, 50.0))
            # Sorted by (team, person): the guest slot first. The referee is never a player.
            self.assertEqual(match.players.shape, (5, 2, 2))
            np.testing.assert_array_equal(match.teams, [1, 0])
            np.testing.assert_allclose(match.players[1, :, 0], [0.75, 0.25])
            self.assertTrue(np.isnan(match.players[0, 0, 0]))
            np.testing.assert_allclose(match.ball[0], [0.5, 0.5])
            cached = idsse.load_match(MATCH, root)
            np.testing.assert_array_equal(cached.labels, match.labels)

    def test_missing_half_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            write_fixture(Path(folder), second_half_ball=False)
            with self.assertRaisesRegex(ValueError, "two halves"):
                idsse.load_match(MATCH, Path(folder))


class MergeTests(unittest.TestCase):
    def test_brief_spell_returns_to_the_previous_team(self):
        H, A, D = idsse.H, idsse.A, idsse.D
        labels = np.array([H] * 50 + [A] * 10 + [H] * 50, dtype=np.int8)
        merged = idsse.merge_brief_spells(labels, [(0, 109)], seconds=1)
        np.testing.assert_array_equal(merged, np.full(110, H))
        kept = idsse.merge_brief_spells(labels, [(0, 109)], seconds=0.2)
        np.testing.assert_array_equal(kept, labels)
        # A restart has a clear possessor: a brief spell after D stays, and D never changes.
        labels = np.array([D] * 30 + [A] * 10 + [H] * 50, dtype=np.int8)
        np.testing.assert_array_equal(idsse.merge_brief_spells(labels, [(0, 89)], seconds=1), labels)

    def test_blocks_are_independent(self):
        labels = np.array([idsse.H] * 40 + [idsse.A] * 10, dtype=np.int8)
        merged = idsse.merge_brief_spells(labels, [(0, 39), (40, 49)], seconds=1)
        np.testing.assert_array_equal(merged, labels)


if __name__ == "__main__":
    unittest.main()
