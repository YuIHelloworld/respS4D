"""S4D layer: diagonal structured state-space temporal convolution.

Reference:
  Gu et al., "On the Parameterization and Initialization of Diagonal State
  Space Models", NeurIPS 2022.

This is a small dependency-free S4D-Lin style implementation used as a
drop-in temporal operator inside the TDT backbone.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class S4DKernel(nn.Module):
    """Compute one causal S4D convolution kernel per channel."""

    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        dt_min: float = 0.1,
        dt_max: float = 1.0,
        init_mode: str = "band_uniform",
        freq_min: float = 0.1,
        freq_max: float = 0.5,
        fs_eff: float = 2.5,
    ):
        super().__init__()
        if d_state % 2 != 0:
            raise ValueError("d_state must be even")
        if dt_min <= 0 or dt_max <= 0 or dt_min >= dt_max:
            raise ValueError("Require 0 < dt_min < dt_max")
        if init_mode not in {"s4d_lin", "band_uniform", "resp_log"}:
            raise ValueError("init_mode must be 's4d_lin', 'band_uniform' or 'resp_log'")
        if freq_min <= 0 or freq_max <= freq_min:
            raise ValueError("Require 0 < freq_min < freq_max for respiratory init")
        if fs_eff <= 0:
            raise ValueError("fs_eff must be positive")
        if init_mode == "resp_log" and freq_max >= 0.5 * fs_eff:
            raise ValueError("resp_log freq_max must stay below the effective Nyquist rate")

        channels = d_model
        states = d_state // 2

        log_dt = torch.rand(channels) * (math.log(dt_max) - math.log(dt_min))
        log_dt = log_dt + math.log(dt_min)
        self.log_dt = nn.Parameter(log_dt)

        self.log_A_real = nn.Parameter(torch.log(0.5 * torch.ones(channels, states)))
        if init_mode == "s4d_lin":
            # Literal S4D-Lin default (Gu et al. 2022): A_imag[k] = pi*k rad/sec
            # (~k/2 Hz). Kept as the Scheme-A ablation baseline: because our
            # temporal_op runs after temporal_down x3 (fs_eff=2.5 Hz, Nyquist
            # 1.25 Hz), this ladder places ~91% of states above the effective
            # Nyquist, where they alias back into the respiratory band. This is
            # the init-domain mismatch the band-aware/resp_log modes correct.
            A_imag = (math.pi * torch.arange(states).float()).repeat(channels, 1)
        elif init_mode == "band_uniform":
            # Band-aware A_imag init. Default S4D-Lin uses A_imag[k]=pi*k rad/sec
            # (k/2 Hz), which assumes Nyquist >> N/2 Hz. Our temporal_op runs after
            # temporal_down x3, so fs_eff=2.5 Hz and Nyquist=1.25 Hz: with N=32
            # states, the S4D-Lin ladder puts 29/32 states (91%) above Nyquist
            # where they alias instead of contributing useful frequency content.
            # Spreading A_imag uniformly over [0, pi) rad/sec keeps the states in
            # the respiratory-scale range while preserving the S4D pair form.
            A_imag = (math.pi * torch.arange(states).float()
                      / max(states, 1)).repeat(channels, 1)
        else:
            # Respiratory-log init places the *discrete* oscillator frequencies
            # log-uniformly in [freq_min, freq_max] Hz after the current dt and
            # temporal downsampling are applied:
            #   f = A_imag * dt * fs_eff / (2*pi)
            # This avoids wasting low-dt channels on sub-respiratory frequencies.
            target_freq = torch.logspace(
                math.log10(freq_min),
                math.log10(freq_max),
                states,
            )
            dt = torch.exp(log_dt).unsqueeze(-1)
            A_imag = (2.0 * math.pi * target_freq.unsqueeze(0)) / (fs_eff * dt)
        self.A_imag = nn.Parameter(A_imag)

        C = torch.randn(channels, states, 2) / math.sqrt(states)
        self.C = nn.Parameter(C)

    def forward(self, length: int) -> Tensor:
        """Return a real kernel with shape [d_model, length]."""
        dt = torch.exp(self.log_dt)
        A = -torch.exp(self.log_A_real) + 1j * self.A_imag
        C = torch.view_as_complex(self.C)

        dtA = A * dt.unsqueeze(-1)
        steps = torch.arange(length, device=A.device, dtype=dtA.real.dtype)
        zoh = (torch.exp(dtA) - 1.0) / A
        C_eff = C * zoh
        K = C_eff.unsqueeze(-1) * torch.exp(dtA.unsqueeze(-1) * steps)
        return 2.0 * K.real.sum(dim=1)


class S4D(nn.Module):
    """S4D block: kernel convolution, skip connection, and GLU output.

    Input/output shape: [B, d_model, L].
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        dropout: float = 0.0,
        dt_min: float = 0.1,
        dt_max: float = 1.0,
        bidirectional: bool = False,
        init_mode: str = "band_uniform",
        freq_min: float = 0.1,
        freq_max: float = 0.5,
        fs_eff: float = 2.5,
        d_init: float | None = None,
        output_scale: float = 1.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.bidirectional = bidirectional
        self.output_scale = float(output_scale)
        if self.output_scale <= 0:
            raise ValueError("output_scale must be positive")
        kernel_kwargs = dict(
            init_mode=init_mode,
            freq_min=freq_min,
            freq_max=freq_max,
            fs_eff=fs_eff,
        )
        self.kernel = S4DKernel(d_model, d_state, dt_min, dt_max, **kernel_kwargs)
        self.kernel_bwd = (
            S4DKernel(d_model, d_state, dt_min, dt_max, **kernel_kwargs)
            if bidirectional else None
        )
        if d_init is None:
            self.D = nn.Parameter(torch.randn(d_model))
        else:
            self.D = nn.Parameter(torch.full((d_model,), float(d_init)))
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
        self.out_proj = nn.Sequential(
            nn.Conv1d(d_model, 2 * d_model, kernel_size=1),
            nn.GLU(dim=-2),
        )

    @staticmethod
    def _causal_conv(u: Tensor, kernel: S4DKernel, length: int) -> Tensor:
        """Apply one causal S4D convolution and keep the first L samples."""
        K = kernel(length).float()
        K_f = torch.fft.rfft(K, n=2 * length)
        u_f = torch.fft.rfft(u.float(), n=2 * length)
        return torch.fft.irfft(u_f * K_f, n=2 * length)[..., :length]

    def forward(self, u: Tensor) -> Tensor:
        length = u.size(-1)
        device_type = u.device.type if u.device.type in {"cuda", "cpu"} else "cpu"
        with torch.amp.autocast(device_type=device_type, enabled=False):
            y = self._causal_conv(u, self.kernel, length)
            if self.kernel_bwd is not None:
                y_bwd = self._causal_conv(u.flip(-1), self.kernel_bwd, length).flip(-1)
                y = y + y_bwd

        y = y.to(dtype=u.dtype)
        y = y + u * self.D.to(dtype=u.dtype).unsqueeze(-1)
        y = self.activation(y)
        return self.out_proj(self.dropout(y)) * self.output_scale
