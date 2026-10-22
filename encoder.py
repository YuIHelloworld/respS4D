"""Encoder and respiratory readout blocks for RespDisentangleNet."""

import math
from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from s4d_layer import S4D


class TokenLayerNorm(nn.Module):
    """Per-token LayerNorm for 5-D feature maps [B, C, T, H, W].

    Standard pre-norm in Transformers uses LayerNorm, not BatchNorm.
    BatchNorm3d normalises over the batch dimension which is unreliable at
    batch_size=1 and introduces train/eval discrepancy via running statistics.
    TokenLayerNorm permutes to [..., C] and applies LayerNorm per spatial token,
    matching the semantics of ViT pre-norm exactly.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.ln = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T, H, W]
        x = x.permute(0, 2, 3, 4, 1)   # [B, T, H, W, C]
        x = self.ln(x)
        return x.permute(0, 4, 1, 2, 3) # [B, C, T, H, W]

class SinusoidalPositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding for sequence input [B, T, dim].

    Added before temporal self-attention in RespiratoryBranch so the model
    can distinguish the position of each breathing cycle within the window.
    The buffer is precomputed up to max_len (default 512, covers 8s windows);
    longer sequences (e.g. 30s = 600 frames) get a dynamically computed encoding
    so the buffer size — and therefore checkpoint compatibility — is unchanged.
    """
    def __init__(self, dim: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("pe", self._build_pe(max_len, dim).unsqueeze(0))  # [1, max_len, dim]

    @staticmethod
    def _build_pe(length: int, dim: int) -> torch.Tensor:
        pe = torch.zeros(length, dim)
        position = torch.arange(0, length).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, dim]
        T = x.size(1)
        if T <= self.pe.size(1):
            pe = self.pe[:, :T]
        else:
            # Sequence longer than the precomputed buffer (e.g. 30s windows).
            pe = self._build_pe(T, self.dim).to(device=x.device, dtype=x.dtype).unsqueeze(0)
        x = x + pe
        return self.dropout(x)


# ==============================================================================
# Building Blocks (unchanged from v1)
# ==============================================================================

class CDC_T(nn.Module):
    """Temporal Center-Difference Convolution (from PhysFormer).

    Combines vanilla 3D convolution with a temporal difference term to capture
    subtle frame-to-frame skin color changes. theta controls the balance.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.6):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation,
                              groups=groups, bias=bias)
        self.theta = theta

    def forward(self, x):
        out_normal = self.conv(x)
        if math.fabs(self.theta - 0.0) < 1e-8:
            return out_normal
        if self.conv.weight.shape[2] > 1:
            kernel_diff = (self.conv.weight[:, :, 0, :, :].sum(2).sum(2) +
                          self.conv.weight[:, :, 2, :, :].sum(2).sum(2))
            kernel_diff = kernel_diff[:, :, None, None, None]
            out_diff = F.conv3d(input=x, weight=kernel_diff, bias=self.conv.bias,
                               stride=self.conv.stride, padding=0,
                               dilation=self.conv.dilation, groups=self.conv.groups)
            return out_normal - self.theta * out_diff
        return out_normal


# ==============================================================================
# Shared Encoder
# ==============================================================================

class FusionStem(nn.Module):
    """Fusion Stem: appearance + multi-order frame differences (from RhythmFormer).

    Processes both raw appearance and temporal differences to capture static
    skin properties and dynamic physiological changes simultaneously.

    Input:  [B, T, C, H, W]
    Output: [B*T, 64, H/4, W/4]
    """
    MOTION_CHANNELS = {
        "none": 0,
        "farneback": 2,
        "farneback_x": 1,
        "farneback_y": 1,
        "farneback_mag": 1,
        "farneback_env": 3,  # flow_x, flow_y, |flow_y - spatial_median| (polarity-invariant)
        "farneback_y_env": 2,  # flow_y plus sign-free local vertical-motion envelope
        "farneback_y_bodyonly": 1,  # flow_y with offline upper-body silhouette masking in the loader
    }

    DIFF_MODES = ("raw_4order", "shao_norm_1order", "raw_1order", "multiscale",
                  "learnable_bandpass", "learnable_bandpass_m6",
                  "free_temporal", "gabor_init_free_m6",
                  "lowrank_coherent", "basis_mask_gate",
                  "appear_gate", "coherence_gate")
    # Centered temporal-difference lags (frames @20fps) for diff_mode="multiscale".
    # The shortest lag retains the cardiac / RIIV coupling cue; the longer lags
    # pass the 0.1-0.5 Hz respiratory band that a single lag-1 difference (the
    # cardiac-matched raw_4order) suppresses ~4x relative to the 1 Hz cardiac band.
    MULTISCALE_LAGS = (1, 5, 10, 16)

    # ---- diff_mode="learnable_bandpass" (HybridRespBandStem) ----
    # 2 fixed fast centred-difference taps (motion-edge velocity + cardiac/RIIV
    # cue) + len(LB_GABOR_MU0) learnable constant-Q Gabor band-pass taps tuned to
    # the 0.1-0.5 Hz respiratory band.  Every kernel is zero-mean (no static
    # appearance leak) and L2-normalised; a per-tap learnable gain keeps the fast
    # and band-pass taps RMS-balanced.  Total taps x 3 RGB = 12 channels, aligned
    # with raw_4order for a fair head-to-head.  See _build_lb_bank / forward.
    #   diff_mode="free_temporal" is the matched capacity control: same kernel
    #   length and tap count, but unconstrained (zero-mean only) temporal kernels.
    #   diff_mode="gabor_init_free_m6" uses the M6 fast/Gabor bank only as the
    #   initial value, then trains fully free zero-mean/L2 temporal kernels.
    LB_FS = 20.0                 # frame rate (Hz)
    LB_K = 121                   # temporal kernel length (frames, ~6.05 s)
    LB_FAST_LAGS = (1, 2)        # fixed centred raw-diff lags: X(t+L) - X(t-L)
    # Gabor centre-freq init (Hz).  K=121 keeps >=1 cycle in-window down to ~0.2 Hz
    # so the zero-mean DC removal does not push the low tap up (realised peaks
    # ~0.21/0.44 Hz).  The slow end (0.1-0.2 Hz) is left to the band-init S4D; the
    # stem covers the resolvable mid-upper respiratory band.  mu is learnable, so
    # report the realised |H(f)| peak (not the raw mu) for interpretability.
    LB_GABOR_MU0 = (0.20, 0.42)  # low-/high-resp band, realisable at K=121
    LB_GABOR_Q = 2.0             # constant-Q init: bandwidth_hz = mu / Q

    # ---- diff_mode="learnable_bandpass_m6" (wider respiratory-band coverage) ----
    # Same HybridRespBandStem machinery, but 4 Gabor taps span the resting
    # respiratory band instead of 2: 2 fast + 4 Gabor = 6 taps x 3 RGB = 18ch.
    # The 0.14 Hz init sits below one in-window cycle at K=121 (period 7.14 s >
    # 6.05 s), so its realised |H(f)| peak is pushed up by the zero-mean DC
    # removal — report the realised peak, not the raw mu, for interpretability.
    M6_FAST_LAGS = (1, 2)
    M6_GABOR_MU0 = (0.14, 0.22, 0.34, 0.48)  # ~8.4 / 13.2 / 20.4 / 28.8 bpm
    M6_K = 121
    M6_Q = 2.0

    # ---- diff_mode="lowrank_coherent" (Plan A.3: spatial-coherent low-rank stem) ----
    # Keep the broadband raw_4order difference (TIME axis untouched — no respiratory
    # band prior, which §FusionStem-log proved net-negative), then project each diff
    # channel's spatial map onto LR_NUM_BASIS learnable smooth spatial fields = a
    # low-rank "thoracic-expansion" subspace.  Respiration is a spatially-coherent,
    # low-rank expansion/contraction pattern; texture flicker / sensor noise / micro-
    # expressions are spatially high-rank/incoherent and lie outside span(basis), so
    # the projection removes them.  Basis is parameterised as a low-res control grid
    # (LR_GRID) bilinearly upsampled to the diff resolution (built-in smoothness) and
    # vertical-anatomy initialised (global volume + vertical ramp + vertical Gaussian
    # bands), matching the empirical flow_y >> flow_x finding.  no-OF differentiable
    # surrogate for RespVAD / flow-subspace / multi-region-OF grids.
    LR_NUM_BASIS = 6
    LR_GRID = 16

    # ---- diff_mode="basis_mask_gate" (Codex P1: large-kernel spatial-coherence gate) ----
    # Cheaper rung of the same spatial axis: soft-reweight the broadband diff by a
    # spatially low-pass motion-energy gate, diff' = diff * sigmoid(s*lowpass(E)+b).
    # A large depthwise kernel learns the spatial-coherence scale; spatially-sustained
    # (coherent) motion is kept, isolated high-spatial-frequency flicker is attenuated.
    # Soft gate (re-weighting) vs lowrank_coherent's hard low-rank bottleneck.
    BG_KERNEL = 11

    # ---- diff_mode="appear_gate" (G1: appearance->motion soft-attention gate) ----
    # DeepPhys/MTTS-CAN SAM paradigm: derive the spatial gate from the static
    # APPEARANCE frame (skin / body silhouette), NOT the diff's own energy, so the
    # mask can localise the quiet thoracic region and suppress loud head motion --
    # the discrimination an energy self-gate (basis_mask_gate) cannot make.  Small
    # appearance conv -> 11x11 coherence blur + low-res learnable static prior ->
    # sigmoid, applied to the broadband 12ch raw_4order diff (TIME axis untouched).
    AG_HIDDEN = 8
    AG_PRIOR_GRID = 8

    # ---- diff_mode="coherence_gate" (G2: true spatial-coherence gate) ----
    # Same soft-gate form as basis_mask_gate, but the gate INPUT is the local
    # spatial-coherence ratio C = lowpass(diff)^2 / lowpass(diff^2) in [0,1]
    # (magnitude-invariant), not raw motion energy: coherent thoracic expansion ->
    # C~1 (kept), incoherent specular/quantisation flicker -> C~0 (suppressed).
    # Directly tests the basis_mask_gate "30s gain = incoherent-flicker suppression"
    # mechanism hypothesis by gating on coherence rather than magnitude.

    def __init__(self, alpha=0.5, beta=0.5, motion_mode: str = "none",
                 disable_frame_diff: bool = False,
                 diff_mode: str = "raw_4order"):
        """
        Args:
            disable_frame_diff: ablation switch.  When True, the multi-order
                frame-difference path is removed and the stem reduces to the
                appearance Conv2d branch only (motion channels, if any, are
                concatenated with the raw RGB so OF still reaches the network).
                Used to isolate the contribution of the 4-order frame-diff
                path inherited from RhythmFormer Fusion_Stem.
            diff_mode: choose temporal-difference operator.
                "raw_4order" (default) — RhythmFormer-style 4-order raw frame
                difference stacked into 12 channels.
                "shao_norm_1order" — Shao 2025 Eq.8 normalized 1-order
                difference V_D(t)=(V(t+1)-V(t))/(V(t+1)+V(t)+eps), 3 channels.
                Used for stem-design head-to-head against Shao 2025.
                "raw_1order" — plain 1-order raw difference V(t+1)-V(t), 3 ch.
                "multiscale" — multi-scale zero-phase centered differences
                I(t+L)-I(t-L) stacked over MULTISCALE_LAGS (4 lags x 3 RGB =
                12 channels, drop-in vs raw_4order).  Matches the stem's
                temporal receptive field to the slow respiratory band: a single
                lag-1 difference is high-pass with gain ~4x weaker at 0.25 Hz
                than at the 1 Hz cardiac band, so it suppresses exactly the
                respiratory band we want to read out.
        """
        super().__init__()
        if motion_mode not in self.MOTION_CHANNELS:
            raise ValueError(f"Unsupported motion_mode={motion_mode!r}")
        if diff_mode not in self.DIFF_MODES:
            raise ValueError(
                f"Unsupported diff_mode={diff_mode!r}; expected one of {self.DIFF_MODES}"
            )
        self.motion_mode = motion_mode
        self.motion_channels = self.MOTION_CHANNELS[motion_mode]
        self.disable_frame_diff = disable_frame_diff
        self.diff_mode = diff_mode
        # raw_4order: 4 orders × 3 RGB = 12 channels (RhythmFormer-style)
        # raw_1order: 1 order × 3 RGB = 3 channels (single-tap V(t+1)-V(t))
        # shao_norm_1order: 1 order × 3 RGB = 3 channels (Shao Eq.8, normalized)
        # multiscale: len(MULTISCALE_LAGS) centered diffs × 3 RGB (12 by default)
        # learnable_bandpass / free_temporal: (NUM_FAST + NUM_GABOR) taps × 3 RGB
        self.multiscale_lags = self.MULTISCALE_LAGS
        # Resolve the learnable-temporal-stem spec (fast lags / Gabor centres /
        # kernel length / Q) per diff_mode so learnable_bandpass(_m6) and the
        # free_temporal control share one machinery with different tap budgets.
        (self._lb_fast_lags, self._lb_gabor_mu0,
         self._lb_K, self._lb_Q) = self._resolve_lb_spec(diff_mode)
        self._lb_num_fast = len(self._lb_fast_lags)
        self._lb_num_gabor = len(self._lb_gabor_mu0)
        self._lb_num_taps = self._lb_num_fast + self._lb_num_gabor
        if diff_mode == "multiscale":
            diff_raw_channels = 3 * len(self.multiscale_lags)
        elif diff_mode in ("raw_4order", "lowrank_coherent", "basis_mask_gate",
                           "appear_gate", "coherence_gate"):
            # The spatial-coherence modes operate ON the raw_4order diff (12ch),
            # preserving it; they only re-shape it spatially, so stem_diff input
            # channels are identical to raw_4order for a clean drop-in A/B.
            diff_raw_channels = 12
        elif diff_mode in ("learnable_bandpass", "learnable_bandpass_m6",
                           "free_temporal", "gabor_init_free_m6"):
            diff_raw_channels = 3 * self._lb_num_taps
        else:
            diff_raw_channels = 3
        self.diff_in_channels = diff_raw_channels + self.motion_channels
        self._init_temporal_bank(diff_mode)
        self._init_spatial_modules(diff_mode)

        appear_in_channels = 3 + (self.motion_channels if disable_frame_diff else 0)
        self.stem_appear = nn.Sequential(
            nn.Conv2d(appear_in_channels, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        if not disable_frame_diff:
            self.stem_diff = nn.Sequential(
                nn.Conv2d(self.diff_in_channels, 64, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            )
            self.fusion1 = nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            )
            self.fusion2 = nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            )
        else:
            # Single-path appearance-only stem; one extra 3x3 conv to keep
            # parameter count and receptive field roughly comparable.
            self.fusion1 = nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            )
        self.alpha = alpha
        self.beta = beta
        # Interpretability hooks: set export_maps=True (e.g. in the basis-map
        # export script) to stash the last projection coefficients / gate map.
        self.export_maps = False
        self._last_coeff: Optional[Tensor] = None
        self._last_gate: Optional[Tensor] = None

    # ------------------------------------------------------------------
    # Learnable temporal-filter stem (diff_mode in {learnable_bandpass,
    # free_temporal}).  Shared depthwise-over-RGB temporal-conv machinery.
    # ------------------------------------------------------------------
    def _resolve_lb_spec(self, diff_mode: str):
        """Return (fast_lags, gabor_mu0, K, Q) for the learnable-temporal modes.

        learnable_bandpass and the free_temporal capacity control share the M4
        spec (2 fast + 2 Gabor, 12ch); learnable_bandpass_m6 widens to 2 fast +
        4 Gabor (18ch).  Non-bank modes get the M4 defaults harmlessly (unused).
        """
        if diff_mode in ("learnable_bandpass_m6", "gabor_init_free_m6"):
            return self.M6_FAST_LAGS, self.M6_GABOR_MU0, self.M6_K, self.M6_Q
        return self.LB_FAST_LAGS, self.LB_GABOR_MU0, self.LB_K, self.LB_GABOR_Q

    def _init_temporal_bank(self, diff_mode: str) -> None:
        """Create learnable parameters for the temporal-filter diff modes."""
        gamma0 = math.log(math.e - 1.0)  # softplus(gamma0) == 1 -> RMS-matched
        if diff_mode in ("learnable_bandpass", "learnable_bandpass_m6"):
            mu0 = torch.tensor(self._lb_gabor_mu0, dtype=torch.float32)
            self.lb_mu = nn.Parameter(mu0.clone())                 # centre (Hz)
            self.lb_bw = nn.Parameter(mu0.clone() / self._lb_Q)    # bandwidth (Hz)
            self.lb_gamma_raw = nn.Parameter(torch.full((self._lb_num_taps,), gamma0))
            t = torch.arange(self._lb_K, dtype=torch.float32) - (self._lb_K - 1) / 2.0
            self.register_buffer("lb_t", t, persistent=False)
        elif diff_mode == "free_temporal":
            self.ft_kernel = nn.Parameter(torch.randn(self._lb_num_taps, self._lb_K) * 0.02)
            self.ft_gamma_raw = nn.Parameter(torch.full((self._lb_num_taps,), gamma0))
        elif diff_mode == "gabor_init_free_m6":
            bank = self._build_gabor_seed_bank(
                self._lb_fast_lags, self._lb_gabor_mu0, self._lb_K, self._lb_Q
            )
            self.ft_kernel = nn.Parameter(bank.clone())
            self.ft_gamma_raw = nn.Parameter(torch.full((self._lb_num_taps,), gamma0))

    # ------------------------------------------------------------------
    # Spatial-coherence stem (diff_mode in {lowrank_coherent, basis_mask_gate}).
    # Operate on the *spatial* axis only; the temporal-difference axis is the
    # broadband raw_4order diff and is left untouched (band priors proved harmful).
    # ------------------------------------------------------------------
    def _init_spatial_modules(self, diff_mode: str) -> None:
        """Create learnable params for the spatial-coherence diff modes (Plan A / P1)."""
        if diff_mode == "lowrank_coherent":
            # Low-res learnable control grid -> bilinear-upsampled smooth basis.
            self.lr_basis = nn.Parameter(
                self._make_vertical_basis_init(self.LR_NUM_BASIS, self.LR_GRID, self.LR_GRID)
            )
        elif diff_mode == "basis_mask_gate":
            k = self.BG_KERNEL
            self.bg_blur = nn.Conv2d(1, 1, kernel_size=k, padding=k // 2, bias=False)
            with torch.no_grad():
                self.bg_blur.weight.copy_(self._gaussian_kernel(k))
            # gate = sigmoid(scale * lowpass(energy) + bias); init near pass-through.
            self.bg_scale = nn.Parameter(torch.tensor(1.0))
            self.bg_bias = nn.Parameter(torch.tensor(1.5))
        elif diff_mode == "appear_gate":
            # G1: appearance->motion soft-attention gate (DeepPhys/MTTS-CAN SAM).
            # Small conv on the static appearance frame -> spatial mask; 11x11
            # Gaussian blur imposes the coherence length; a low-res learnable prior
            # (G3) lets it learn WHERE respiration lives, decoupled from diff energy.
            self.ag_conv = nn.Sequential(
                nn.Conv2d(3, self.AG_HIDDEN, kernel_size=3, padding=1),
                nn.GroupNorm(1, self.AG_HIDDEN),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.AG_HIDDEN, 1, kernel_size=1),
            )
            # Zero-init the last conv weight + bias=1.5 -> gate starts ~sigmoid(1.5)
            # uniform (near pass-through); the appearance signal enters as it trains.
            nn.init.zeros_(self.ag_conv[-1].weight)
            nn.init.constant_(self.ag_conv[-1].bias, 1.5)
            k = self.BG_KERNEL
            self.ag_blur = nn.Conv2d(1, 1, kernel_size=k, padding=k // 2, bias=False)
            with torch.no_grad():
                self.ag_blur.weight.copy_(self._gaussian_kernel(k))
            g = self.AG_PRIOR_GRID
            self.ag_prior = nn.Parameter(torch.zeros(1, 1, g, g))
        elif diff_mode == "coherence_gate":
            # G2: true spatial-coherence gate; same form as basis_mask_gate but the
            # gate input is the coherence ratio C in [0,1], not raw motion energy.
            k = self.BG_KERNEL
            self.cg_blur = nn.Conv2d(1, 1, kernel_size=k, padding=k // 2, bias=False)
            with torch.no_grad():
                self.cg_blur.weight.copy_(self._gaussian_kernel(k))
            self.cg_scale = nn.Parameter(torch.tensor(1.0))
            self.cg_bias = nn.Parameter(torch.tensor(1.5))

    @staticmethod
    def _make_vertical_basis_init(k: int, h0: int, w0: int) -> Tensor:
        """Vertical/anatomical init for the low-rank basis.

        field 0   : global constant (overall thoracic volume / whole-frame expansion)
        field 1   : centred vertical ramp (global vertical excursion)
        fields 2+ : evenly-spaced vertical Gaussian bands (chest / shoulder / face
                    heights), constant across width -> encodes the flow_y prior.
        A small random perturbation breaks symmetry so the fields can specialise.
        """
        y = torch.linspace(0.0, 1.0, h0).view(h0, 1)
        fields = [torch.ones(h0, w0), (y - 0.5).expand(h0, w0).clone()]
        n_band = max(k - len(fields), 0)
        for i in range(n_band):
            c = (i + 0.5) / max(n_band, 1)
            sigma = 0.5 / max(n_band, 1)
            band = torch.exp(-((y - c) ** 2) / (2.0 * sigma ** 2)).expand(h0, w0).clone()
            fields.append(band)
        basis = torch.stack(fields[:k], dim=0)
        return basis + 0.01 * torch.randn_like(basis)

    @staticmethod
    def _gaussian_kernel(k: int) -> Tensor:
        """Normalised 2D Gaussian depthwise kernel [1,1,k,k] (sum=1) for the gate low-pass."""
        t = torch.arange(k, dtype=torch.float32) - (k - 1) / 2.0
        g = torch.exp(-(t ** 2) / (2.0 * (k / 4.0) ** 2))
        ker = torch.outer(g, g)
        ker = ker / ker.sum()
        return ker.view(1, 1, k, k)

    def _build_basis_fields(self, H: int, W: int, device) -> Tensor:
        """Upsample the low-res control grid to [K,H,W] and L2-normalise each field (fp32).

        Not zero-meaned: the constant field captures the spatial mean of the (already
        time-DC-removed) diff = the global frame-to-frame expansion, a real resp cue.
        """
        b = self.lr_basis.float().unsqueeze(0)                       # [1,K,h0,w0]
        b = F.interpolate(b, size=(H, W), mode="bilinear", align_corners=False)[0]
        b = b / (b.flatten(1).norm(dim=1).view(-1, 1, 1) + 1e-6)     # unit-L2 per field
        return b.to(device)

    def _apply_lowrank_projection(self, diff: Tensor) -> Tensor:
        """Project each diff channel's spatial map onto span(basis): Σ_k <diff,B_k> B_k.

        diff [N,C,H,W] -> low-rank reconstruction [N,C,H,W] in span(basis); the
        component orthogonal to the coherent-expansion subspace (incoherent noise)
        is dropped.  Basis built in fp32; matmuls run in diff.dtype (tensor cores
        accumulate in fp32, so AMP-safe).
        """
        N, C, H, W = diff.shape
        Bn = self._build_basis_fields(H, W, diff.device)             # [K,H,W] fp32 unit-norm
        Bf = Bn.reshape(Bn.shape[0], H * W).to(diff.dtype)           # [K,HW]
        df = diff.reshape(N * C, H * W)
        coeff = df @ Bf.t()                                          # [N*C,K]
        if self.export_maps:
            self._last_coeff = coeff.detach().reshape(N, C, -1)
        rec = coeff @ Bf                                             # [N*C,HW]
        return rec.reshape(N, C, H, W)

    def _apply_basis_gate(self, diff: Tensor) -> Tensor:
        """Soft spatial-coherence gate: diff * sigmoid(scale * lowpass(energy) + bias)."""
        energy = diff.pow(2).mean(dim=1, keepdim=True)               # [N,1,H,W]
        energy = energy / (energy.mean(dim=(2, 3), keepdim=True) + 1e-6)
        blurred = self.bg_blur(energy)
        gate = torch.sigmoid(self.bg_scale * blurred + self.bg_bias)
        if self.export_maps:
            self._last_gate = gate.detach()
        return diff * gate

    def _apply_appear_gate(self, diff: Tensor, appear: Tensor) -> Tensor:
        """G1 appearance->motion gate: diff * sigmoid(blur(conv(appearance)) + prior).

        The mask is derived from the static appearance frame (skin / body
        silhouette), so it can up-weight the quiet thoracic region and suppress
        loud head motion -- a discrimination the energy self-gate cannot make.

        Memory-lean form (single static appearance + half-res compute): the gate
        is computed ONCE per window from the time-averaged appearance frame and
        broadcast over time + the diff channels, and ag_conv runs at half spatial
        resolution with the mask bilinear-upsampled back.  The appearance is
        ~static within a short window and the 11x11 Gaussian blur keeps the gate
        smooth, so this is near-equivalent to the per-frame full-res gate while
        cutting the gate's peak activation by ~(T x 4) -- which keeps the run off
        the VRAM ceiling (and out of the cuDNN slow-kernel regime).

        diff   : [B, T, C, H, W]
        appear : [B, 3, H, W]   (one static appearance frame per window)
        """
        _, _, _, H, W = diff.shape
        appear_lr = F.interpolate(appear, size=(H // 2, W // 2), mode="bilinear",
                                  align_corners=False)              # [B,3,H/2,W/2]
        logit_lr = self.ag_blur(self.ag_conv(appear_lr))            # [B,1,H/2,W/2]
        logit = F.interpolate(logit_lr, size=(H, W), mode="bilinear",
                              align_corners=False)                  # [B,1,H,W]
        prior = F.interpolate(self.ag_prior, size=(H, W), mode="bilinear",
                              align_corners=False)
        gate = torch.sigmoid(logit + prior)                        # [B,1,H,W]
        if self.export_maps:
            self._last_gate = gate.detach()
        return diff * gate.unsqueeze(1)                            # [B,1,1,H,W] broadcast

    def _apply_coherence_gate(self, diff: Tensor) -> Tensor:
        """G2 true spatial-coherence gate: diff * sigmoid(scale * C + bias).

        C = lowpass(diff)^2 / lowpass(diff^2) in [0,1] is the magnitude-invariant
        coherent fraction of local motion energy (per channel, then averaged):
        coherent thoracic expansion -> C~1 (kept), incoherent flicker -> C~0.
        """
        N, C, H, W = diff.shape
        d = diff.reshape(N * C, 1, H, W)
        mu = self.cg_blur(d)                                         # local mean
        sec = self.cg_blur(d * d)                                    # local 2nd moment
        coh = (mu * mu) / (sec + 1e-6)                               # [N*C,1,H,W] in [0,1]
        coh = coh.reshape(N, C, H, W).mean(dim=1, keepdim=True)      # [N,1,H,W]
        gate = torch.sigmoid(self.cg_scale * coh + self.cg_bias)
        if self.export_maps:
            self._last_gate = gate.detach()
        return diff * gate

    def _raw_4order_diff(self, x_rgb: Tensor) -> Tensor:
        """RhythmFormer-style 4 stacked 1st-order frame differences -> [B,T,12,H,W].

        Shared by raw_4order and the spatial-coherence modes so they provably use
        the same broadband temporal-difference front-end.
        """
        T = x_rgb.shape[1]
        x1 = torch.cat([x_rgb[:, :1], x_rgb[:, :1], x_rgb[:, :T - 2]], dim=1)
        x2 = torch.cat([x_rgb[:, :1], x_rgb[:, :T - 1]], dim=1)
        x3 = x_rgb
        x4 = torch.cat([x_rgb[:, 1:], x_rgb[:, T - 1:]], dim=1)
        x5 = torch.cat([x_rgb[:, 2:], x_rgb[:, T - 1:], x_rgb[:, T - 1:]], dim=1)
        return torch.cat([x2 - x1, x3 - x2, x4 - x3, x5 - x4], dim=2)

    def _build_gabor_seed_bank(self, fast_lags, gabor_mu0, K: int, q: float) -> Tensor:
        """Fixed fast + Gabor seed bank for free temporal kernels.

        The returned rows are only an initialization for gabor_init_free_m6; after
        construction, ft_kernel is trained as an unconstrained temporal bank.
        """
        c = (K - 1) // 2
        rows = []
        for lag in fast_lags:
            row = torch.zeros(K, dtype=torch.float32)
            row[c + lag] = 1.0
            row[c - lag] = -1.0
            rows.append(row)
        t = torch.arange(K, dtype=torch.float32) - (K - 1) / 2.0
        for mu in gabor_mu0:
            bw = max(float(mu) / float(q), 0.02)
            sigma_t = self.LB_FS / (2.0 * math.pi * bw)
            env = torch.exp(-(t ** 2) / (2.0 * sigma_t ** 2))
            carrier = torch.cos(2.0 * math.pi * float(mu) / self.LB_FS * t)
            rows.append(carrier * env)
        bank = torch.stack(rows, dim=0)
        bank = bank - bank.mean(dim=1, keepdim=True)
        return bank / (bank.norm(dim=1, keepdim=True) + 1e-6)

    def _build_lb_bank(self) -> Tensor:
        """[num_taps, K] bank: fixed fast centred diffs + constant-Q Gabor taps.

        Every kernel is zero-mean (DC removed so static appearance cannot leak)
        and L2-normalised so all taps start at a common scale.
        """
        K = self._lb_K
        c = (K - 1) // 2
        device = self.lb_mu.device
        rows = []
        for lag in self._lb_fast_lags:                     # fixed motion-edge taps
            row = torch.zeros(K, device=device)
            row[c + lag] = 1.0
            row[c - lag] = -1.0
            rows.append(row)
        t = self.lb_t.to(device)                           # learnable Gabor taps
        mu = self.lb_mu.clamp(0.02, self.LB_FS / 2 - 0.02)
        bw = self.lb_bw.clamp(0.02, self.LB_FS / 2)
        for m in range(self._lb_num_gabor):
            sigma_t = self.LB_FS / (2.0 * math.pi * bw[m])  # constant-Q: ~1/mu init
            env = torch.exp(-(t ** 2) / (2.0 * sigma_t ** 2))
            carrier = torch.cos(2.0 * math.pi * mu[m] / self.LB_FS * t)
            rows.append(carrier * env)
        bank = torch.stack(rows, dim=0)
        bank = bank - bank.mean(dim=1, keepdim=True)
        bank = bank / (bank.norm(dim=1, keepdim=True) + 1e-6)
        return bank

    def _build_ft_bank(self) -> Tensor:
        """[num_taps, K] unconstrained kernels, zero-mean + L2 (capacity control)."""
        bank = self.ft_kernel - self.ft_kernel.mean(dim=1, keepdim=True)
        return bank / (bank.norm(dim=1, keepdim=True) + 1e-6)

    def _apply_temporal_bank(self, x_rgb: Tensor, bank: Tensor, gamma: Tensor) -> Tensor:
        """Apply a [num_taps, K] temporal kernel bank to each RGB channel.

        x_rgb [B,T,3,H,W] -> [B,T,3*num_taps,H,W] (group-major over RGB).  Uses
        cross-correlation (PyTorch conv) so a [+1@c+L, -1@c-L] row reproduces the
        centred difference X(t+L)-X(t-L); symmetric Gabor rows are zero-phase.
        """
        n, K = bank.shape
        xp = x_rgb.permute(0, 2, 1, 3, 4)                   # [B,3,T,H,W]
        xp = F.pad(xp, (0, 0, 0, 0, K // 2, K // 2), mode="replicate")
        weight = bank.repeat(3, 1).view(3 * n, 1, K, 1, 1)  # groups=3, shared over RGB
        out = F.conv3d(xp, weight, groups=3)                # [B,3n,T,H,W]
        out = out * gamma.repeat(3).view(1, 3 * n, 1, 1, 1)
        return out.permute(0, 2, 1, 3, 4).contiguous()      # [B,T,3n,H,W]

    def forward(self, x):
        B, T, C, H, W = x.shape
        if self.motion_mode != "none":
            expected_c = 3 + self.motion_channels
            if C != expected_c:
                raise ValueError(
                    f"motion_mode={self.motion_mode!r} expects C={expected_c}, got C={C}"
                )
            x_rgb = x[:, :, :3]
            motion = x[:, :, 3:]
        else:
            if C < 3:
                raise ValueError(f"FusionStem expects at least RGB input with C>=3, got C={C}")
            x_rgb = x[:, :, :3]
            motion = None

        if self.disable_frame_diff:
            # Appearance-only: still admit OF channels (if any) by concatenating
            # them with RGB so the motion signal still reaches the encoder; the
            # ablation removes only the multi-order frame-difference path.
            if motion is not None:
                appear_input = torch.cat([x_rgb, motion], dim=2)
                appear_input = appear_input.contiguous().view(
                    B * T, 3 + self.motion_channels, H, W)
            else:
                appear_input = x_rgb.contiguous().view(B * T, 3, H, W)
            feat = self.stem_appear(appear_input)
            return self.fusion1(feat)

        if self.diff_mode == "raw_4order":
            # Multi-order temporal differences (4 orders → 12 channels)
            diff = self._raw_4order_diff(x_rgb)
            appear_frame = x_rgb
        elif self.diff_mode == "lowrank_coherent":
            # Plan A.3: broadband raw_4order diff, then spatial low-rank projection
            # onto the learnable coherent-expansion subspace (time axis untouched).
            diff = self._raw_4order_diff(x_rgb)
            B_, T_, Cd, Hh, Ww = diff.shape
            diff = self._apply_lowrank_projection(
                diff.reshape(B_ * T_, Cd, Hh, Ww)
            ).reshape(B_, T_, Cd, Hh, Ww)
            appear_frame = x_rgb
        elif self.diff_mode == "basis_mask_gate":
            # Codex P1: broadband raw_4order diff, then soft large-kernel
            # spatial-coherence gate (re-weighting, not projection).
            diff = self._raw_4order_diff(x_rgb)
            B_, T_, Cd, Hh, Ww = diff.shape
            diff = self._apply_basis_gate(
                diff.reshape(B_ * T_, Cd, Hh, Ww)
            ).reshape(B_, T_, Cd, Hh, Ww)
            appear_frame = x_rgb
        elif self.diff_mode == "appear_gate":
            # G1: broadband raw_4order diff, soft-gated by an appearance-derived
            # spatial-attention mask (single static appearance, half-res compute;
            # DeepPhys/MTTS-CAN SAM paradigm).
            diff = self._raw_4order_diff(x_rgb)                     # [B,T,12,H,W]
            appe = x_rgb.mean(dim=1)                                # [B,3,H,W] static appearance
            diff = self._apply_appear_gate(diff, appe)             # gate broadcast over T,C
            appear_frame = x_rgb
        elif self.diff_mode == "coherence_gate":
            # G2: broadband raw_4order diff, soft-gated by the spatial-coherence
            # ratio C = lowpass(diff)^2 / lowpass(diff^2) (magnitude-invariant).
            diff = self._raw_4order_diff(x_rgb)
            B_, T_, Cd, Hh, Ww = diff.shape
            diff = self._apply_coherence_gate(
                diff.reshape(B_ * T_, Cd, Hh, Ww)
            ).reshape(B_, T_, Cd, Hh, Ww)
            appear_frame = x_rgb
        elif self.diff_mode == "multiscale":
            # Multi-scale zero-phase centered differences d_L(t)=I(t+L)-I(t-L).
            # |H_L(f)|=2|sin(2*pi*f*L/fs)|: the short lag keeps the cardiac/RIIV
            # cue, the longer lags lift the 0.1-0.5 Hz respiratory band that a
            # single lag-1 difference (raw_4order) attenuates ~4x.  Edges are
            # held (replicate first/last frame); L is clamped so short windows
            # never index out of range.  Channel count stays 3*len(lags)=12,
            # so stem_diff is unchanged and this is a drop-in vs raw_4order.
            diffs = []
            for lag in self.multiscale_lags:
                L = max(1, min(int(lag), T - 1))
                fwd = torch.cat(
                    [x_rgb[:, L:], x_rgb[:, -1:].expand(-1, L, -1, -1, -1)], dim=1)
                back = torch.cat(
                    [x_rgb[:, :1].expand(-1, L, -1, -1, -1), x_rgb[:, :T - L]], dim=1)
                diffs.append(fwd - back)
            diff = torch.cat(diffs, dim=2)
            appear_frame = x_rgb
        elif self.diff_mode == "raw_1order":
            # Plain 1-order frame difference V(t+1)-V(t) (3 channels, no normalization)
            x_next = torch.cat([x_rgb[:, 1:], x_rgb[:, T-1:]], dim=1)
            diff = x_next - x_rgb
            appear_frame = x_rgb
        elif self.diff_mode in ("learnable_bandpass", "learnable_bandpass_m6"):
            # HybridRespBandStem: 2 fixed fast centred-diff taps (motion-edge
            # velocity + cardiac/RIIV cue) + constant-Q Gabor band-pass taps tuned
            # to the 0.1-0.5 Hz respiratory band.  Zero-mean + L2 kernels, per-tap
            # learnable gain.  M4=12ch (aligned with raw_4order) / M6=18ch.
            bank = self._build_lb_bank().to(x_rgb.dtype)
            diff = self._apply_temporal_bank(x_rgb, bank, F.softplus(self.lb_gamma_raw))
            appear_frame = x_rgb
        elif self.diff_mode in ("free_temporal", "gabor_init_free_m6"):
            # Capacity / receptive-field control for learnable_bandpass: same K and
            # tap count, unconstrained (zero-mean only) temporal kernels.  The
            # gabor_init_free_m6 variant starts from the M6 fast/Gabor bank, then
            # lets the kernels move freely; FFT reveals the learned passbands.
            bank = self._build_ft_bank().to(x_rgb.dtype)
            diff = self._apply_temporal_bank(x_rgb, bank, F.softplus(self.ft_gamma_raw))
            appear_frame = x_rgb
        else:
            # Shao 2025 Eq.8 normalized 1-order frame difference (3 channels)
            x_next = torch.cat([x_rgb[:, 1:], x_rgb[:, T-1:]], dim=1)
            diff = (x_next - x_rgb) / (x_next + x_rgb + 1e-6)
            appear_frame = x_rgb

        if self.motion_mode != "none":
            diff = torch.cat([diff, motion], dim=2)
        diff = diff.view(B * T, self.diff_in_channels, H, W)
        appear = appear_frame.contiguous().view(B * T, 3, H, W)

        feat_appear = self.stem_appear(appear)
        feat_diff = self.stem_diff(diff)

        path1 = self.alpha * feat_appear + self.beta * feat_diff
        path1 = self.fusion1(path1)
        path2 = self.fusion2(feat_diff)
        return self.alpha * path1 + self.beta * path2

class TDSpatialAttention(nn.Module):
    """Spatial-only attention with TDC projection.

    Attends across H'×W' spatial tokens WITHIN each frame independently.
    Complexity: O(T × (H'W')²) — with H'W'=64 this is very cheap.
    """
    def __init__(self, dim, num_heads=4, dropout=0.1, theta=0.6, tau=2.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.tau = tau

        self.proj_q = nn.Sequential(
            CDC_T(dim, dim, 3, stride=1, padding=1, groups=1, bias=False, theta=theta),
            nn.BatchNorm3d(dim),
        )
        self.proj_k = nn.Sequential(
            CDC_T(dim, dim, 3, stride=1, padding=1, groups=1, bias=False, theta=theta),
            nn.BatchNorm3d(dim),
        )
        self.proj_v = nn.Conv3d(dim, dim, 1, stride=1, padding=0, bias=False)
        self.proj_out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, C, T, H, W = x.shape
        q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x)
        S = H * W
        q = q.permute(0, 2, 1, 3, 4).reshape(B * T, C, H * W).transpose(1, 2)
        k = k.permute(0, 2, 1, 3, 4).reshape(B * T, C, H * W).transpose(1, 2)
        v = v.permute(0, 2, 1, 3, 4).reshape(B * T, C, H * W).transpose(1, 2)

        q = q.view(B * T, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B * T, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B * T, S, self.num_heads, self.head_dim).transpose(1, 2)

        attn = self.dropout(F.softmax((q @ k.transpose(-2, -1)) / self.tau, dim=-1))
        out = (attn @ v).transpose(1, 2).contiguous().view(B * T, S, C)
        out = self.proj_out(out)
        return out.transpose(1, 2).view(B, T, C, H, W).permute(0, 2, 1, 3, 4)

class TDTemporalAttention(nn.Module):
    """Temporal-only attention with TDC projection.

    Attends across T' temporal tokens per spatial location independently.
    Complexity: O(H'W' × T'²) — after temporal downsampling T' is small.
    """
    def __init__(self, dim, num_heads=4, dropout=0.1, theta=0.6, tau=2.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.tau = tau

        self.proj_q = nn.Sequential(
            CDC_T(dim, dim, 3, stride=1, padding=1, groups=1, bias=False, theta=theta),
            nn.BatchNorm3d(dim),
        )
        self.proj_k = nn.Sequential(
            CDC_T(dim, dim, 3, stride=1, padding=1, groups=1, bias=False, theta=theta),
            nn.BatchNorm3d(dim),
        )
        self.proj_v = nn.Conv3d(dim, dim, 1, stride=1, padding=0, bias=False)
        self.proj_out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, C, T, H, W = x.shape
        q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x)

        q = q.permute(0, 3, 4, 1, 2).reshape(B * H * W, C, T).transpose(1, 2)
        k = k.permute(0, 3, 4, 1, 2).reshape(B * H * W, C, T).transpose(1, 2)
        v = v.permute(0, 3, 4, 1, 2).reshape(B * H * W, C, T).transpose(1, 2)

        q = q.view(B * H * W, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B * H * W, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B * H * W, T, self.num_heads, self.head_dim).transpose(1, 2)

        attn = self.dropout(F.softmax((q @ k.transpose(-2, -1)) / self.tau, dim=-1))
        out = (attn @ v).transpose(1, 2).contiguous().view(B * H * W, T, C)
        out = self.proj_out(out)
        return out.transpose(1, 2).view(B, H, W, C, T).permute(0, 3, 4, 1, 2)

class TDS4DTemporal(nn.Module):
    """Drop-in S4D replacement for TDTemporalAttention.

    Keeps the CDC_T projection so the temporal-difference inductive bias remains
    unchanged, then applies one diagonal SSM per channel at each spatial token.

    Input/output shape: [B, C, T, H, W].
    """

    def __init__(
        self,
        dim: int,
        dropout: float = 0.1,
        theta: float = 0.6,
        d_state: int = 64,
        dt_min: float = 0.1,
        dt_max: float = 1.0,
        bidirectional: bool = False,
        init_mode: str = "band_uniform",
        freq_min: float = 0.1,
        freq_max: float = 0.5,
        fs_eff: float = 2.5,
        d_init: Optional[float] = None,
        output_scale: float = 1.0,
    ):
        super().__init__()
        self.proj_in = nn.Sequential(
            CDC_T(dim, dim, 3, stride=1, padding=1, groups=1, bias=False, theta=theta),
            nn.BatchNorm3d(dim),
        )
        self.s4d = S4D(
            d_model=dim,
            d_state=d_state,
            dropout=dropout,
            dt_min=dt_min,
            dt_max=dt_max,
            bidirectional=bidirectional,
            init_mode=init_mode,
            freq_min=freq_min,
            freq_max=freq_max,
            fs_eff=fs_eff,
            d_init=d_init,
            output_scale=output_scale,
        )

    def forward(self, x: Tensor) -> Tensor:
        B, C, T, H, W = x.shape
        x = self.proj_in(x)
        x = x.permute(0, 3, 4, 1, 2).contiguous().view(B * H * W, C, T)
        x = self.s4d(x)
        return x.view(B, H, W, C, T).permute(0, 3, 4, 1, 2).contiguous()

class SpatioTemporalFFN(nn.Module):
    """ST Feed-Forward: 1x1x1 → depthwise 3x3x3 → 1x1x1 (from PhysFormer)."""
    def __init__(self, dim, ff_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(dim, ff_dim, 1, bias=False), nn.BatchNorm3d(ff_dim), nn.GELU(),
            nn.Conv3d(ff_dim, ff_dim, 3, stride=1, padding=1, groups=ff_dim, bias=False),
            nn.BatchNorm3d(ff_dim), nn.GELU(),
            nn.Conv3d(ff_dim, dim, 1, bias=False), nn.BatchNorm3d(dim),
        )

    def forward(self, x):
        return self.net(x)

class FactorizedTDTransformerBlock(nn.Module):
    """Factorized TD-Transformer: Spatial Attn → Temporal Attn → ST-FFN.

    Total: O(T×S² + S×T²) where S=H'W', much cheaper than O((T×S)²).

    Pre-norm uses TokenLayerNorm (per-token LayerNorm over channel dim) instead
    of BatchNorm3d.  BN3d was unreliable at batch_size=1 and introduced a
    train/eval gap via running-statistics bias.  LN is batch-size-independent.

    Args:
        use_token_ln: when False, all three pre-norms are replaced by plain
            BatchNorm3d.  Used to ablate the TokenLN→BN3d swap and reproduce
            the original PhysFormer/RhythmFormer normalization choice.
    """
    def __init__(self, dim, num_heads=4, ff_dim=96, dropout=0.1, theta=0.6,
                 use_token_ln: bool = True, attn_mode: str = "serial",
                 use_spatial_attn: bool = True,
                 use_temporal_attn: bool = True,
                 temporal_op: str = "attn",
                 s4d_state: int = 64,
                 s4d_dt_min: float = 0.1,
                 s4d_dt_max: float = 1.0,
                 s4d_bidirectional: bool = False,
                 s4d_init_mode: str = "band_uniform",
                 s4d_freq_min: float = 0.1,
                 s4d_freq_max: float = 0.5,
                 s4d_fs_eff: float = 2.5,
                 s4d_d_init: Optional[float] = None,
                 s4d_output_scale: float = 1.0):
        super().__init__()
        if attn_mode not in {"serial", "parallel"}:
            raise ValueError("attn_mode must be 'serial' or 'parallel'")
        if temporal_op not in {"attn", "s4d"}:
            raise ValueError("temporal_op must be 'attn' or 's4d'")
        if not use_spatial_attn and not use_temporal_attn:
            raise ValueError("At least one of spatial/temporal attention must remain enabled")
        norm_factory = (lambda d: TokenLayerNorm(d)) if use_token_ln else (lambda d: nn.BatchNorm3d(d))
        self.use_token_ln = use_token_ln
        self.attn_mode = attn_mode
        # E10/E11 attention-branch ablation switches. Defaults keep the original
        # serial spatial->temporal TDTransformer path unchanged.
        self.use_spatial_attn = use_spatial_attn
        self.use_temporal_attn = use_temporal_attn
        self.temporal_op = temporal_op
        self.norm_s  = norm_factory(dim)
        self.spatial_attn = TDSpatialAttention(dim, num_heads, dropout, theta)
        self.norm_t  = norm_factory(dim)
        if temporal_op == "attn":
            self.temporal_attn = TDTemporalAttention(dim, num_heads, dropout, theta)
        else:
            self.temporal_attn = TDS4DTemporal(
                dim,
                dropout=dropout,
                theta=theta,
                d_state=s4d_state,
                dt_min=s4d_dt_min,
                dt_max=s4d_dt_max,
                bidirectional=s4d_bidirectional,
                init_mode=s4d_init_mode,
                freq_min=s4d_freq_min,
                freq_max=s4d_freq_max,
                fs_eff=s4d_fs_eff,
                d_init=s4d_d_init,
                output_scale=s4d_output_scale,
            )
        self.norm_ff = norm_factory(dim)
        self.ffn = SpatioTemporalFFN(dim, ff_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        if self.attn_mode == "parallel":
            # Original E12 parallel path:
            # spatial = self.dropout(self.spatial_attn(self.norm_s(x)))
            # temporal = self.dropout(self.temporal_attn(self.norm_t(x)))
            # x = x + spatial + temporal
            # E10/E11 modification: keep the disabled branch as identity by
            # contributing zero residual instead of removing module definitions.
            spatial = (
                self.dropout(self.spatial_attn(self.norm_s(x)))
                if self.use_spatial_attn else torch.zeros_like(x)
            )
            temporal = (
                self.dropout(self.temporal_attn(self.norm_t(x)))
                if self.use_temporal_attn else torch.zeros_like(x)
            )
            x = x + spatial + temporal
        else:
            # Original serial path:
            # x = x + self.dropout(self.spatial_attn(self.norm_s(x)))
            # x = x + self.dropout(self.temporal_attn(self.norm_t(x)))
            # E10/E11 modification: skip only the selected attention residual.
            if self.use_spatial_attn:
                x = x + self.dropout(self.spatial_attn(self.norm_s(x)))
            if self.use_temporal_attn:
                x = x + self.dropout(self.temporal_attn(self.norm_t(x)))
        x = x + self.dropout(self.ffn(self.norm_ff(x)))
        return x

class SharedEncoder(nn.Module):
    """Shared encoder with temporal downsampling + factorized attention.

    Pipeline:
      FusionStem → PatchEmbed → TemporalDownsample → FactorizedTDTransformer × N → TemporalUpsample

    Input:  [B, T, C, H, W]
    Output: [B, dim, T, H', W']   (same temporal resolution as input)
    """
    def __init__(self, dim=64, num_layers=4, num_heads=4, ff_dim=96,
                 dropout=0.1, theta=0.6, temporal_ds_layers=3,
                 motion_mode: str = "none",
                 disable_frame_diff: bool = False,
                 diff_mode: str = "raw_4order",
                 use_token_ln: bool = True,
                 tdt_attn_mode: str = "serial",
                 tdt_use_spatial_attn: bool = True,
                 tdt_use_temporal_attn: bool = True,
                 tdt_temporal_op: str = "attn",
                 s4d_state: int = 64,
                 s4d_dt_min: float = 0.1,
                 s4d_dt_max: float = 1.0,
                 s4d_bidirectional: bool = False,
                 s4d_init_mode: str = "band_uniform",
                 s4d_freq_min: float = 0.1,
                 s4d_freq_max: float = 0.5,
                 s4d_fs_eff: float = 2.5,
                 s4d_d_init: Optional[float] = None,
                 s4d_output_scale: float = 1.0):
        super().__init__()
        self.dim = dim
        self.temporal_ds_layers = temporal_ds_layers
        self.tdt_attn_mode = tdt_attn_mode
        self.tdt_use_spatial_attn = tdt_use_spatial_attn
        self.tdt_use_temporal_attn = tdt_use_temporal_attn
        self.tdt_temporal_op = tdt_temporal_op
        self.fusion_stem = FusionStem(
            motion_mode=motion_mode,
            disable_frame_diff=disable_frame_diff,
            diff_mode=diff_mode,
        )
        self.patch_embedding = nn.Conv3d(64, dim, kernel_size=(1, 4, 4), stride=(1, 4, 4))

        # Temporal downsampling: stride-2 conv along time
        self.temporal_down = nn.ModuleList()
        for _ in range(temporal_ds_layers):
            self.temporal_down.append(nn.Sequential(
                nn.BatchNorm3d(dim),
                nn.Conv3d(dim, dim, kernel_size=(2, 1, 1), stride=(2, 1, 1)),
                nn.GELU(),
            ))

        # Factorized transformer blocks (operate at reduced temporal resolution)
        self.transformer_blocks = nn.ModuleList([
            FactorizedTDTransformerBlock(dim, num_heads, ff_dim, dropout, theta,
                                         use_token_ln=use_token_ln,
                                         attn_mode=tdt_attn_mode,
                                         use_spatial_attn=tdt_use_spatial_attn,
                                         use_temporal_attn=tdt_use_temporal_attn,
                                          temporal_op=tdt_temporal_op,
                                          s4d_state=s4d_state,
                                          s4d_dt_min=s4d_dt_min,
                                          s4d_dt_max=s4d_dt_max,
                                          s4d_bidirectional=s4d_bidirectional,
                                          s4d_init_mode=s4d_init_mode,
                                          s4d_freq_min=s4d_freq_min,
                                          s4d_freq_max=s4d_freq_max,
                                          s4d_fs_eff=s4d_fs_eff,
                                          s4d_d_init=s4d_d_init,
                                          s4d_output_scale=s4d_output_scale)
            for _ in range(num_layers)
        ])

        # Temporal upsampling: restore original frame rate
        self.temporal_up = nn.ModuleList()
        for _ in range(temporal_ds_layers):
            self.temporal_up.append(nn.Sequential(
                nn.Upsample(scale_factor=(2, 1, 1)),
                nn.Conv3d(dim, dim, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0)),
                nn.BatchNorm3d(dim),
                nn.GELU(),
            ))

    def forward(self, x):
        B, T, C, H, W = x.shape
        fused = self.fusion_stem(x)  # [B*T, 64, H/4, W/4]
        _, C_out, Hf, Wf = fused.shape
        features = fused.view(B, T, C_out, Hf, Wf).permute(0, 2, 1, 3, 4)
        features = self.patch_embedding(features)  # [B, dim, T, H/16, W/16]

        for down in self.temporal_down:
            features = down(features)

        for block in self.transformer_blocks:
            features = block(features)

        for up in self.temporal_up:
            features = up(features)

        # Handle rounding: trim or pad to match original T
        T_out = features.shape[2]
        if T_out > T:
            features = features[:, :, :T, :, :]
        elif T_out < T:
            pad = T - T_out
            features = F.pad(features, (0, 0, 0, 0, 0, pad))

        return features


# ==============================================================================
# Auxiliary Cardiac Branch (simplified — no RIIV head)
# [ISOLATED LEGACY STRATUM: envelope-RIV generation — see full banner above
#  EnvelopeExtractor. Load-bearing for v4 ckpts, ≈0 Pearson, do not extend.]
# ==============================================================================

class RespiratoryBranch(nn.Module):
    """Primary respiratory waveform decoder from shared physical features.

    Pipeline:
      1. Branch-specific 3D conv + spatial pool → [B, T, dim]
      2. Sinusoidal PE + temporal self-attention (respiratory long-range patterns)
      3. 1D conv head → respiratory waveform [B, T]

    Input:  [B, dim, T, H', W'] shared features
    Output: resp waveform [B, T]

    Note: the legacy v3 cardiac-conditioning path (CardiacFeatureDecomposer +
    SubtractionGatedCrossAttention) was removed 2026-05-18. `env_window` and
    `num_cross_attn_layers` are retained as inert kwargs for caller-signature
    compatibility; `cardiac_feat` in forward is accepted but ignored.
    """
    def __init__(self, dim=64, num_heads=4, num_self_attn_layers=2,
                 num_cross_attn_layers=2, dropout=0.1, env_window=10):
        super().__init__()

        # Branch-specific spatial feature extraction
        self.spatial_conv = nn.Sequential(
            nn.Conv3d(dim, dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(dim), nn.GELU(),
            nn.Conv3d(dim, dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(dim), nn.GELU(),
        )

        self.pos_enc = SinusoidalPositionalEncoding(dim, dropout=dropout)

        # Temporal self-attention for long-range respiratory patterns
        self.temporal_self_attn = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=dim, nhead=num_heads, dim_feedforward=dim * 2,
                dropout=dropout, batch_first=True, norm_first=True,
            )
            for _ in range(num_self_attn_layers)
        ])

        # Final signal projection
        self.head = nn.Sequential(
            nn.Conv1d(dim, dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(dim // 2), nn.GELU(),
            nn.Conv1d(dim // 2, 1, kernel_size=1),
        )

    def forward(self, shared_features, cardiac_feat=None, return_feat: bool = False):
        """
        Args:
            shared_features: [B, dim, T, H', W']
            cardiac_feat:    accepted but ignored (legacy v3 signature)
            return_feat:     if True, also return the pre-head feature [B, dim, T]
        Returns:
            resp: [B, T] predicted respiratory waveform
        """
        x = self.spatial_conv(shared_features)           # [B, dim, T, H', W']
        feat_seq = x.mean(dim=-1).mean(dim=-1)           # [B, dim, T]
        feat_seq = feat_seq.transpose(1, 2)              # [B, T, dim]

        feat_seq = self.pos_enc(feat_seq)

        for self_attn_layer in self.temporal_self_attn:
            feat_seq = self_attn_layer(feat_seq)

        feat_out = feat_seq.transpose(1, 2)              # [B, dim, T]
        resp = self.head(feat_out).squeeze(1)             # [B, T]
        if return_feat:
            return resp, feat_out
        return resp


# ==============================================================================
# Full Model
# ==============================================================================

