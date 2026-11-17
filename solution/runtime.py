"""C5-alpha-0.1 inference with ordinary imports and no development cache."""
import json
import numpy as np
import torch

from . import assets, c4_structural as c4, c5_structural as c5, dense_had as net
from .n14_trace import trace

LABELS = np.array(list("HADX"))


class Runtime:
    def __init__(self, weights_dir=assets.DEFAULT_WEIGHTS):
        self.config = assets.configuration()
        if torch.__version__ != self.config["torch_version"]:
            raise ValueError("C5 requires the pinned PyTorch CPU build: " + self.config["torch_version"])
        torch.set_num_threads(self.config["threads"])
        torch.use_deterministic_algorithms(True)
        paths = assets.verify(weights_dir)
        self.weights = [net.load(paths[row["file"]]) for row in assets.manifest()["files"] if row["role"] == "tcn"]
        self.linear = json.loads(paths["linear.json"].read_text(encoding="utf-8-sig"))
        with np.load(paths["meta.npz"], allow_pickle=False) as data:
            self.meta = {k: data[k] for k in data.files}
        if set(self.meta) != {"mean", "std", "weight", "bias"}:
            raise ValueError("Unexpected meta-model schema")
        if any(not np.isfinite(v).all() for v in self.meta.values()) or (self.meta["std"] <= 0).any():
            raise ValueError("Invalid meta-model parameters")

    def predict_block(self, players, ball, teams, *, diagnostics=False):
        players, ball, teams = np.asarray(players), np.asarray(ball), np.asarray(teams)
        n = len(ball)
        if n < 2 or players.shape != (n, len(teams), 2) or ball.shape != (n, 2):
            raise ValueError("A tracking block needs at least two frames and matching x/y arrays")
        if not np.isin(teams, [0, 1]).all() or set(teams.tolist()) != {0, 1}:
            raise ValueError("Both teams must be present in the slot mapping")
        if np.isinf(players).any() or np.isinf(ball).any():
            raise ValueError("Use NaN for missing observations, never infinity")
        pitch = tuple(self.config["pitch_m"])
        provenance = trace(players, ball, teams, self.linear, self.config["n14_flags"], pitch)
        x = net.features(players, ball, teams, pitch)
        probability = np.mean(np.stack([net.probabilities(x, *w) for w in self.weights]), axis=0)
        meta_context = c4.context(provenance, probability, x, players, pitch)
        features = c4.meta_features(meta_context, provenance)
        m = self.meta
        with torch.inference_mode():
            normalized = torch.from_numpy(np.clip((features - m["mean"]) / m["std"], -8, 8))
            meta_probability = torch.nn.functional.linear(normalized, torch.from_numpy(m["weight"]), torch.from_numpy(m["bias"])).softmax(1).numpy()
        context = c5.context(provenance, probability, x, players, pitch)
        prediction, details = c5.decode(context, meta_probability, self.config["variant"])
        labels = LABELS[prediction]
        if diagnostics:
            return labels, dict(n14=provenance["labels"], probability=probability, meta_probability=meta_probability,
                                features=x, trace=provenance, mechanisms=details)
        return labels

    def predict_match(self, players, ball, teams, periods=None):
        periods = self.config["periods"] if periods is None else periods
        n = len(ball)
        previous = 0
        for start, stop in periods:
            if (type(start) is not int or type(stop) is not int or start < previous
                    or start < 0 or stop > n or stop - start < 2):
                raise ValueError("Periods must be ordered, disjoint, nonempty intervals inside the input")
            previous = stop
        if not periods:
            raise ValueError("At least one processing period is required")
        labels = np.full(n, "X")
        for start, stop in periods:
            labels[start:stop] = self.predict_block(players[start:stop], ball[start:stop], teams)
        return labels
