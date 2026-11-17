"""Preserved C5 plateau and blending counterexamples, independent of any match."""
import unittest
import numpy as np
from solution import c5_structural as c5, c3_structural as c3


class RuleTests(unittest.TestCase):
    def fixture(self):
        n = 160
        trace = dict(labels=np.full(n, "H"), state=np.full(n, ord("H"), np.uint8),
                     age=np.zeros(n, int), hold=np.array(75), owner_supported=np.ones(n, bool),
                     weak=np.zeros(n, bool), reasons=np.full(n, ""), coherent=np.zeros(n, bool),
                     decision_reason=np.full(n, ""), restart_side=np.full(n, -1),
                     d_entry=np.zeros(n, bool), inside=np.ones(n, bool))
        probability = np.tile([.04, .01, .95], (n, 1))
        x = np.zeros((n, 37))
        x[:, 0] = -.499
        x[:, 2] = 1
        x[:, [14, 26]] = .01
        x[:, [17, 29]] = 1
        x[:, [22, 34]] = .5
        return trace, probability, x, np.full((n, 22, 2), .5)

    def test_confirmed_contact_at_line_blocks_plateau(self):
        trace, p, x, players = self.fixture()
        self.assertTrue(c3.context(trace, p, x)["plateau"].all())
        self.assertFalse(c5.context(trace, p, x, players, (105.3, 68.))["plateau"].any())

    def test_raw_X_or_D_preserves_plateau(self):
        for label in ("X", "D"):
            trace, p, x, players = self.fixture()
            trace["labels"][80] = label
            self.assertTrue(c5.context(trace, p, x, players, (105.3, 68.))["plateau"].all())

    def test_zero_alpha_does_not_use_meta_probabilities(self):
        trace, p, x, players = self.fixture()
        data = c5.context(trace, p, x, players, (105.3, 68.))
        cfg = dict(alpha=0., tier=False, min_D_len=40)
        a, _ = c5.decode(data, p, cfg)
        b, _ = c5.decode(data, np.tile([.99, .005, .005], (len(p), 1)), cfg)
        np.testing.assert_array_equal(a, b)
