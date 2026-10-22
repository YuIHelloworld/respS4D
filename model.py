"""RespS4D — pure-motion respiratory waveform model.

Three parameterized submodules, matching the released checkpoints exactly:

    encoder      SharedEncoder  (motion stem + temporal pyramid + S4D blocks)
    resp_branch  RespiratoryBranch  (spatial pool + temporal self-attn + 1D head)
    unc_phys     UncHead  (small waveform-quality head; not used at inference)

Forward returns the predicted respiratory waveform r_phys [B, T].
"""

from typing import Optional

import torch
from torch import nn

from encoder import SharedEncoder, RespiratoryBranch


class UncHead(nn.Module):
    """Scalar waveform-quality head: waveform [B, T] -> sigma2 [B, 1].

    Four spectral/statistical summary features -> 2-layer MLP with Softplus.
    Kept so the released state_dict loads with strict=True; its output is not
    used by the pure-motion training objective.
    """

    def __init__(self, hidden: int = 32, fs: float = 20.0):
        super().__init__()
        self.fs = fs
        self.mlp = nn.Sequential(
            nn.Linear(4, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
            nn.Softplus(),
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        with torch.amp.autocast("cuda", enabled=False):
            w = waveform.float()
            T = w.shape[-1]
            freqs = torch.fft.rfftfreq(T, d=1.0 / self.fs).to(w.device)
            mag = torch.abs(torch.fft.rfft(w, dim=-1)) / (T ** 0.5)
            resp_mask = (freqs >= 0.1) & (freqs <= 0.5)
            cardiac_mask = (freqs >= 0.7) & (freqs <= 3.0)
            resp_energy = mag[:, resp_mask].pow(2).mean(-1)
            cardiac_energy = mag[:, cardiac_mask].pow(2).mean(-1)
            wstd = w.std(-1)
            skew = ((w - w.mean(-1, keepdim=True)).pow(3).mean(-1) / (wstd + 1e-8).pow(3))
            feats = torch.stack([resp_energy, cardiac_energy, wstd, skew], dim=-1)
        return self.mlp(feats.to(waveform.dtype))


class RespS4D(nn.Module):
    """RespS4D pure-motion model.

    Defaults reproduce the paper's appearance-weighted motion stem + vertical
    optical-flow + respiration-adapted bidirectional S4D configuration.
    """

    def __init__(
        self,
        dim: int = 64,
        frame: int = 160,
        encoder_layers: int = 4,
        temporal_ds_layers: int = 3,
        resp_self_attn_layers: int = 2,
        num_heads: int = 4,
        ff_dim: int = 96,
        theta: float = 0.6,
        dropout: float = 0.1,
        fs: float = 20.0,
        motion_mode: str = "farneback_y",
        diff_mode: str = "appear_gate",
        use_token_ln: bool = True,
        tdt_attn_mode: str = "serial",
        tdt_use_spatial_attn: bool = False,
        tdt_use_temporal_attn: bool = True,
        tdt_temporal_op: str = "s4d",
        s4d_state: int = 64,
        s4d_dt_min: float = 0.05,
        s4d_dt_max: float = 0.4,
        s4d_bidirectional: bool = True,
        s4d_init_mode: str = "resp_log",
        s4d_freq_min: float = 0.1,
        s4d_freq_max: float = 0.5,
        s4d_d_init: Optional[float] = 0.1,
        s4d_output_scale: float = 0.5,
    ):
        super().__init__()
        self.fs = fs
        s4d_fs_eff = fs / (2 ** temporal_ds_layers)  # bottleneck sampling rate

        self.encoder = SharedEncoder(
            dim=dim, num_layers=encoder_layers, num_heads=num_heads,
            ff_dim=ff_dim, dropout=dropout, theta=theta,
            temporal_ds_layers=temporal_ds_layers,
            motion_mode=motion_mode,
            disable_frame_diff=False,
            diff_mode=diff_mode,
            use_token_ln=use_token_ln,
            tdt_attn_mode=tdt_attn_mode,
            tdt_use_spatial_attn=tdt_use_spatial_attn,
            tdt_use_temporal_attn=tdt_use_temporal_attn,
            tdt_temporal_op=tdt_temporal_op,
            s4d_state=s4d_state,
            s4d_dt_min=s4d_dt_min,
            s4d_dt_max=s4d_dt_max,
            s4d_bidirectional=s4d_bidirectional,
            s4d_init_mode=s4d_init_mode,
            s4d_freq_min=s4d_freq_min,
            s4d_freq_max=s4d_freq_max,
            s4d_fs_eff=s4d_fs_eff,
            s4d_d_init=s4d_d_init,
            s4d_output_scale=s4d_output_scale,
        )
        self.resp_branch = RespiratoryBranch(
            dim=dim, num_heads=num_heads,
            num_self_attn_layers=resp_self_attn_layers,
            dropout=dropout,
        )
        self.unc_phys = UncHead(fs=fs)

        self.apply(self._init_weights)

    @torch.no_grad()
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x: torch.Tensor):
        """x: [B, T, C, H, W]  ->  r_phys [B, T], sigma2 [B, 1]."""
        shared = self.encoder(x)              # [B, dim, T, H', W']
        r_phys = self.resp_branch(shared)     # [B, T]
        sigma2 = self.unc_phys(r_phys)        # [B, 1]  (unused by the loss)
        return r_phys, sigma2
