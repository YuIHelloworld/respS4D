"""RespS4D training loss (pure-motion configuration).

    L = alpha * NegPearson(r_hat, r) + beta(e) * SpectralMSE(r_hat, r)
        + lam_amp * Huber(std(r_hat), std(r))

with beta(e) = beta_init * eta ** ((e - 1) / total_epochs) an epoch-dependent
spectral curriculum weight. This is the reduced clean-motion loss (no cardiac /
uncertainty terms), matching the paper's Eq. (13).
"""

import torch
from torch import nn
from torch.nn import functional as F


class NegPearsonLoss(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        pred_c = pred - pred.mean(dim=-1, keepdim=True)
        target_c = target - target.mean(dim=-1, keepdim=True)
        num = (pred_c * target_c).sum(dim=-1)
        den = (torch.sqrt((pred_c ** 2).sum(dim=-1) + self.eps) *
               torch.sqrt((target_c ** 2).sum(dim=-1) + self.eps))
        return (1 - num / den).mean()


class SpectralLoss(nn.Module):
    """MSE between normalized FFT magnitude spectra (FP32 to avoid ComplexHalf)."""

    def forward(self, pred, target):
        with torch.amp.autocast("cuda", enabled=False):
            norm = pred.shape[-1] ** 0.5
            pred_mag = torch.abs(torch.fft.rfft(pred.float(), dim=-1)) / norm
            target_mag = torch.abs(torch.fft.rfft(target.float(), dim=-1)) / norm
            return F.mse_loss(pred_mag, target_mag)


class RespS4DLoss(nn.Module):
    def __init__(
        self,
        alpha: float = 1.0,
        beta_init: float = 0.1,
        lam_amp: float = 0.2,
        eta: float = 5.0,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta_init = beta_init
        self.lam_amp = lam_amp
        self.eta = eta
        self.pearson = NegPearsonLoss()
        self.spectral = SpectralLoss()

    def forward(self, r_hat, resp_gt, epoch: int = 1, total_epochs: int = 50):
        beta = self.beta_init * (self.eta ** ((epoch - 1) / total_epochs))
        loss_pearson = self.pearson(r_hat, resp_gt)
        loss_spectral = self.spectral(r_hat, resp_gt)
        std_pred = r_hat.float().std(-1) + 1e-8
        std_gt = resp_gt.float().std(-1) + 1e-8
        loss_amp = F.huber_loss(std_pred, std_gt.detach(), delta=0.5)
        total = self.alpha * loss_pearson + beta * loss_spectral + self.lam_amp * loss_amp
        return total, {
            "total": total.item(),
            "pearson": loss_pearson.item(),
            "spectral": loss_spectral.item(),
            "amplitude": loss_amp.item(),
            "beta": beta,
        }
