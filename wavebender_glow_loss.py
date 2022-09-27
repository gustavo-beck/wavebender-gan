import numpy as np
import torch
from torch import nn

class XSigmoidLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_prime_t, y_t):
        ey_t = y_t - y_prime_t
        return torch.mean(ey_t * (2 * nn.Sigmoid()(ey_t) - 1))

class WaveBenderGlowLoss(nn.Module):
    def __init__(self):
        super(WaveBenderGlowLoss, self).__init__()

    def forward(self, glow_output, prediction, targets):
        mel_target = targets
        mel_target.requires_grad = False

        mel_out = prediction
        mel_loss = XSigmoidLoss()(mel_out, mel_target)

        z, log_s_list, log_det_W_list = glow_output
        for i, log_s in enumerate(log_s_list):
            if i == 0:
                log_s_total = torch.sum(log_s)
                log_det_W_total = log_det_W_list[i]
            else:
                log_s_total = log_s_total + torch.sum(log_s)
                log_det_W_total += log_det_W_list[i]

        glow_loss = torch.sum(z*z)/2 - log_s_total - log_det_W_total
        glow_loss /= (z.size(0)*z.size(1)*z.size(2))

        loss = mel_loss + torch.exp(glow_loss / 10) / (1 + torch.exp(glow_loss / 10))

        return loss