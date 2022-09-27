import numpy as np
import torch
from torch import nn

class XSigmoidLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_prime_t, y_t):
        ey_t = y_t - y_prime_t
        return torch.mean(ey_t * (2 * nn.Sigmoid()(ey_t) - 1))

class WaveBenderLoss(nn.Module):
    def __init__(self):
        super(WaveBenderLoss, self).__init__()

    def forward(self, prediction, targets):
        mel_target = targets
        mel_target.requires_grad = False

        mel_out = prediction
        # mel_loss = nn.MSELoss(reduction='mean')(mel_out, mel_target)
        mel_loss = XSigmoidLoss()(mel_out, mel_target)

        return mel_loss