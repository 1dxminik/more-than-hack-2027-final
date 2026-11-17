"""Bidirectional multi-stage temporal convolutional network (MS-TCN style) for H/A/D per 5 fps step.

Farha & Gall, MS-TCN (CVPR 2019): dilated non-causal 1-D convolutions, a refinement stage that
reads the previous stage's probabilities, and a truncated-MSE smoothing loss against
over-segmentation. Offline inference is allowed in the challenge, so every layer sees past and
future context. Requires PyTorch (requirements-nn.txt).
"""

import torch
from torch import nn
import torch.nn.functional as F

N_CLASSES = 3


class DilatedResidual(nn.Module):
    def __init__(self, dilation, channels, dropout):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.out = nn.Conv1d(channels, channels, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return x + self.drop(self.out(F.relu(self.conv(x))))


class Stage(nn.Module):
    def __init__(self, inputs, channels, layers, dropout):
        super().__init__()
        self.inp = nn.Conv1d(inputs, channels, 1)
        self.layers = nn.ModuleList(DilatedResidual(2 ** i, channels, dropout) for i in range(layers))
        self.out = nn.Conv1d(channels, N_CLASSES, 1)

    def forward(self, x):
        x = self.inp(x)
        for layer in self.layers:
            x = layer(x)
        return self.out(x)


class MSTCN(nn.Module):
    """Input (B, F, T) standardised features; output list of (B, 3, T) logits, one per stage."""

    def __init__(self, features, channels=64, layers=10, stages=2, dropout=0.3):
        super().__init__()
        self.first = Stage(features, channels, layers, dropout)
        self.refine = nn.ModuleList(Stage(N_CLASSES, channels, layers, dropout) for _ in range(stages - 1))

    def forward(self, x):
        outputs = [self.first(x)]
        for stage in self.refine:
            outputs.append(stage(F.softmax(outputs[-1], dim=1)))
        return outputs


def loss(outputs, labels, class_weight, smooth=0.15, tau=4.0):
    """Sum over stages of weighted CE (labels -1 ignored) + truncated-MSE smoothing on log-probs."""
    total = 0.0
    valid = labels >= 0
    pair = (valid[:, 1:] & valid[:, :-1]).unsqueeze(1).float()
    for logits in outputs:
        total = total + F.cross_entropy(logits, labels.long(), weight=class_weight, ignore_index=-1)
        logp = F.log_softmax(logits, dim=1)
        diff = (logp[:, :, 1:] - logp.detach()[:, :, :-1]) ** 2
        tmse = (diff.clamp(max=tau * tau) * pair).sum() / (pair.sum() * N_CLASSES).clamp(min=1)
        total = total + smooth * tmse
    return total


@torch.no_grad()
def predict_log_probs(model, x, mean, std):
    """(T, F) numpy features of one match -> (T, 3) numpy log-probabilities of the last stage."""
    model.eval()
    tensor = torch.from_numpy(((x - mean) / std).T.copy()).float().unsqueeze(0)
    return F.log_softmax(model(tensor)[-1], dim=1)[0].T.numpy()
