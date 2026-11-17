"""MS-TCN shapes, ignored labels and a few optimisation steps on invented data; skipped without torch."""

import importlib.util
import unittest

import numpy as np

HAS_TORCH = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(HAS_TORCH, "torch not installed (requirements-nn.txt)")
class Model(unittest.TestCase):
    def test_shapes_and_learning(self):
        import torch
        from solution import tcn_model
        torch.manual_seed(0)
        model = tcn_model.MSTCN(5, channels=8, layers=3, stages=2, dropout=0.0)
        x = torch.randn(2, 5, 64)
        y = (x[:, 0] > 0).long()               # label 1 where the first feature is positive
        y[:, :4] = -1                          # ignored
        outputs = model(x)
        self.assertEqual([tuple(o.shape) for o in outputs], [(2, 3, 64)] * 2)
        weight = torch.ones(3)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        first = float(tcn_model.loss(model(x), y, weight))
        for _ in range(30):
            optimizer.zero_grad()
            loss = tcn_model.loss(model(x), y, weight)
            loss.backward()
            optimizer.step()
        self.assertLess(float(loss), first)
        logp = tcn_model.predict_log_probs(model, x[0].numpy().T, np.zeros(5), np.ones(5))
        self.assertEqual(logp.shape, (64, 3))
        np.testing.assert_allclose(np.exp(logp).sum(axis=1), 1, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
