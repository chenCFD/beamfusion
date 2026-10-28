"""
model.py — BeamFusion architecture (for study / Python->C porting groundwork)

Reference:
  Y. Qian, K. Zhao, J. Jin, X. Luo, G. Huang, J. Chen, J. Benesty,
  "Online Neural Fusion of Distortionless Differential Beamformers for
  Robust Speech Enhancement," arXiv:2510.24497, 2025.

WHAT THIS FILE IS
------------------
A from-scratch re-implementation of the paper's data flow:

    mixture y(t) --STFT--> Y(f,l)
        --[P fixed beamformers, e.g. MWNG + 4x first-order DMA]--> Z_p(f,l)
        --[stack Re/Im/|.| of all P beams]--> Z_in in R^(3P x F x T)
        --[ERB compression F -> F']--------------------------> (3P, F', T)
        --[Encoder: strided conv, downsamples frequency]-----> (C, F'', T)
        --[Grouped Dual-Path RNN: intra-freq + inter-time]---> (C, F'', T)
        --[Decoder: transposed conv, upsamples frequency]----> (P, F', T)
        --[ERB expansion F' -> F]-----------------------------> (P, F, T)
        --[softmax over P]------------------------------------> W_p(f,l), sum_p W_p = 1
        --[Sum_p W_p * Z_p]------------------------------------> fused Z(f,l)
        --ISTFT--> enhanced waveform

The paper says the backbone is adapted from GTCRN (Rong et al., ICASSP 2024)
with encoder/decoder/GDPRNN hidden dim = 32, ~85K params, ~142 MMac/s.

IMPORTANT — WHAT IS EXACT VS. APPROXIMATED
-------------------------------------------
The paper does not release code, and several implementation details are not
fully specified in the text. To let you inspect the *end-to-end data flow*
today, this file makes the following choices, each flagged inline as APPROX:

  * Fixed beamformers (MWNG, DMA-I..IV): MWNG is implemented exactly as the
    max-white-noise-gain / delay-and-sum solution (this is closed-form and
    unambiguous). DMA-I..IV are implemented as single-null LCMV beamformers
    (distortionless toward the target + one null toward the interference
    angle), which is the standard way to realize a first-order differential
    beamformer with a null — but it is NOT necessarily byte-identical to the
    specific closed-form DMA-I..IV weights in Benesty & Chen's book [32]
    that the paper cites. Swap in the exact formulas from that reference
    before trusting absolute numbers.
  * ERB compression: band count / edges are not stated in the paper. Uses a
    standard low-band-linear + high-band-ERB-triangular scheme.
  * Encoder/decoder conv depth, kernel sizes, and whether GTCRN's grouped
    dilated conv + SE + TRA blocks are used: not reproduced in full detail.
    A simplified causal conv encoder/decoder stands in for it. For exact
    GTCRN blocks, see: https://github.com/Xiaobin-Rong/gtcrn (Rong et al.)
    and swap into EncoderBlock / DecoderBlock below.
  * Param count: with the real GTCRN blocks missing, hidden_dim=32 alone
    only reaches ~6.3K params (~13x under the paper's ~85K). Config widens
    hidden_dim to 128, adds a 3rd encoder/decoder stage, and deepens the
    G-DPRNN stack to 4 blocks so the TOTAL PARAMETER COUNT lands near 85K
    (see the long comment on Config.enc_channels) -- this matches capacity,
    not the paper's actual architecture or its 142 MMac/s runtime cost.

None of this changes the *shapes* flowing through the network, which is the
point of this script (you said you want to understand the architecture
before committing to a training run).

Run this file directly to see the tensor shape after every stage:
    python model.py
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 0. Config
# ---------------------------------------------------------------------------
class Config:
    # --- array geometry (Sec. 4.1 of the paper) ---
    n_mics = 8
    mic_spacing = 0.01                     # 1 cm ULA
    sound_speed = 343.0
    look_angle_deg = 0.0                   # target steering direction
    null_angles_deg = [90.0, 120.0, 150.0, 180.0]  # DMA-I..IV null directions

    # --- STFT (Sec. 4.3) ---
    sample_rate = 16000
    n_fft = 512
    win_length = 512
    overlap = 128
    hop_length = 128                      # see feature.py: hop 384 has no COLA window

    # --- ERB compression (APPROX: band count not stated in paper) ---
    n_freqs = n_fft // 2 + 1               # 257
    erb_subband_1 = 65                     # low bins kept linear
    erb_subband_2 = 64                     # remaining bins -> this many ERB bands
    # => F' = erb_subband_1 + erb_subband_2 = 129

    # --- BeamFusion network ---
    # Sec. 4.3 states encoder/decoder/GDPRNN dim = 32 and ~85K params, ~142 MMac/s.
    # Those two numbers are INCONSISTENT with the simplified 2-stage conv encoder +
    # 2-block G-DPRNN this file originally used: that combination only reaches ~6.3K
    # params even at hidden_dim=32 (the real GTCRN backbone hits 85K by other means --
    # grouped dilated conv + SE + TRA blocks per encoder/decoder stage -- which are
    # NOT reproduced here, see module docstring). Rather than silently keep dim=32
    # and under-shoot params by ~13x, this file instead widens hidden_dim, adds a
    # 3rd encoder/decoder stage, and deepens the G-DPRNN stack until the *parameter
    # count* lands near the paper's ~85K, so at least capacity is comparable (APPROX:
    # this is a capacity-matching substitute for the missing GTCRN blocks, not a
    # reproduction of them -- runtime MACs will differ from the paper's 142 MMac/s).
    n_beams = 1 + len(null_angles_deg)     # P = MWNG + 4 DMAs = 5
    enc_channels = (16, 32, 128)           # 3 encoder stages; last entry = hidden_dim (APPROX, was 2 stages @ 32)
    hidden_dim = enc_channels[-1]          # bottleneck / G-DPRNN channel width
    n_groups = 4                           # G-DPRNN grouping factor -> group_ch = hidden_dim / n_groups = 32
    n_dprnn_blocks = 4                     # depth of the dual-path stack (APPROX, was 2)
    lcmv_reg = 1e-8                        # diagonal loading for LCMV solve


# ---------------------------------------------------------------------------
# 1. Fixed beamformers: MWNG + single-null LCMV DMAs
#    (pure signal processing, no learnable parameters)
# ---------------------------------------------------------------------------
class FixedBeamformers(nn.Module):
    """
    Computes P fixed, frequency-domain beamforming weight vectors h_p(f) of
    shape [n_mics] once at construction time (closed-form, numpy), then
    applies them to the multichannel STFT at inference time via einsum.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        freqs = np.fft.rfftfreq(cfg.n_fft, d=1.0 / cfg.sample_rate)  # [F]
        tau0 = cfg.mic_spacing / cfg.sound_speed

        def steering_vector(angle_deg):
            theta = np.deg2rad(angle_deg)
            m = np.arange(cfg.n_mics)[:, None]           # [M,1]
            f = freqs[None, :]                            # [1,F]
            phase = +1j * 2 * np.pi * f * m * tau0 * np.cos(theta)  # +j: verified in feature.py --selftest
            return np.exp(phase)                          # [M,F] complex

        d_look = steering_vector(cfg.look_angle_deg)

        # --- MWNG: max white-noise-gain, closed form h = d / (d^H d) ---
        dHd = np.sum(np.conj(d_look) * d_look, axis=0)     # [F], real ~ M
        h_mwng = d_look / dHd[None, :]                     # [M,F]

        # --- DMA-I..IV: single-null LCMV (distortionless + one null) ---
        weights = [h_mwng]
        I2 = np.eye(2, dtype=complex)
        g = np.array([1.0, 0.0], dtype=complex)
        for null_deg in cfg.null_angles_deg:
            d_null = steering_vector(null_deg)
            h = np.zeros((cfg.n_mics, len(freqs)), dtype=complex)
            for fbin in range(len(freqs)):
                C = np.stack([d_look[:, fbin], d_null[:, fbin]], axis=1)  # [M,2]
                CHC = C.conj().T @ C + cfg.lcmv_reg * I2
                w = C @ np.linalg.solve(CHC, g)
                h[:, fbin] = w
            weights.append(h)

        H = np.stack(weights, axis=0)  # [P, M, F] complex
        assert H.shape == (cfg.n_beams, cfg.n_mics, len(freqs))

        # store as a complex buffer (moves with .to(device), not trainable)
        self.register_buffer("H", torch.from_numpy(H).to(torch.complex64))

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """
        y: [B, M, F, T] complex multichannel STFT
        returns Z: [B, P, F, T] complex fixed-beamformer outputs
        """
        # Z[b,p,f,t] = sum_m conj(H[p,m,f]) * y[b,m,f,t]   (i.e. h^H y)
        return torch.einsum("pmf,bmft->bpft", self.H.conj(), y)


# ---------------------------------------------------------------------------
# 2. ERB compression / expansion (frequency-axis only)
# ---------------------------------------------------------------------------
class ERBFilterBank(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        C = self._build_compress_matrix(
            cfg.n_freqs, cfg.sample_rate, cfg.erb_subband_1, cfg.erb_subband_2
        )
        E = C.T.copy()
        col_sums = E.sum(axis=1, keepdims=True)
        col_sums[col_sums == 0] = 1.0
        E = E / col_sums

        self.register_buffer("compress_mat", torch.from_numpy(C).float())  # [F', F]
        self.register_buffer("expand_mat", torch.from_numpy(E).float())    # [F, F']

    @staticmethod
    def _hz2erb(f):
        return 21.4 * np.log10(4.37 * f / 1000.0 + 1.0)

    @staticmethod
    def _erb2hz(e):
        return (10.0 ** (e / 21.4) - 1.0) * 1000.0 / 4.37

    def _build_compress_matrix(self, n_freqs, sr, sub1, sub2):
        freqs_hz = np.linspace(0, sr / 2, n_freqs)
        n_out = sub1 + sub2
        Cmat = np.zeros((n_out, n_freqs), dtype=np.float32)

        for i in range(sub1):
            Cmat[i, i] = 1.0

        f_lo, f_hi = freqs_hz[sub1], freqs_hz[-1]
        erb_lo, erb_hi = self._hz2erb(f_lo), self._hz2erb(f_hi)
        erb_pts = np.linspace(erb_lo, erb_hi, sub2 + 2)
        hz_pts = self._erb2hz(erb_pts)

        for b in range(sub2):
            f_left, f_center, f_right = hz_pts[b], hz_pts[b + 1], hz_pts[b + 2]
            for k in range(sub1, n_freqs):
                f = freqs_hz[k]
                if f_center > f_left and f_left <= f <= f_center:
                    w = (f - f_left) / (f_center - f_left)
                elif f_right > f_center and f_center < f <= f_right:
                    w = (f_right - f) / (f_right - f_center)
                else:
                    w = 0.0
                Cmat[sub1 + b, k] = w

        row_sums = Cmat.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return Cmat / row_sums

    def compress(self, x: torch.Tensor) -> torch.Tensor:
        """x: [..., F, T] -> [..., F', T]"""
        return torch.einsum("of,...ft->...ot", self.compress_mat, x)

    def expand(self, x: torch.Tensor) -> torch.Tensor:
        """x: [..., F', T] -> [..., F, T]"""
        return torch.einsum("fo,...ot->...ft", self.expand_mat, x)


# ---------------------------------------------------------------------------
# 3. Causal conv building blocks (simplified GTCRN-style encoder/decoder)
# ---------------------------------------------------------------------------
class CausalConv2d(nn.Module):
    """Conv2d that only ever looks at past time frames (pads time on the
    left only); frequency axis uses ordinary symmetric padding."""

    def __init__(self, in_ch, out_ch, kernel_size=(3, 3), stride=(1, 1)):
        super().__init__()
        kf, kt = kernel_size
        self.pad_time = kt - 1
        self.conv = nn.Conv2d(
            in_ch, out_ch, kernel_size, stride=stride, padding=(kf // 2, 0)
        )

    def forward(self, x):
        x = F.pad(x, (self.pad_time, 0, 0, 0))  # (left_T, right_T, left_F, right_F)
        return self.conv(x)


class EncoderBlock(nn.Module):
    """
    Kernel is (3,1): 3 taps across frequency, 1 tap across time. Downsampling
    happens only on the frequency axis (stride on freq, stride=1 on time), so
    the block has no temporal receptive field of its own — all temporal
    modeling is left to the G-DPRNN below. This sidesteps the fiddly
    arithmetic of causal *transposed* convolutions in the decoder (see
    DecoderBlock) while keeping the encoder itself trivially causal.
    """

    def __init__(self, in_ch, out_ch, freq_stride=2):
        super().__init__()
        self.conv = CausalConv2d(in_ch, out_ch, kernel_size=(3, 1), stride=(freq_stride, 1))
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.PReLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DecoderBlock(nn.Module):
    """Mirrors EncoderBlock: kernel (3,1), upsamples only the frequency axis
    via ConvTranspose2d with stride=1 on time — so, unlike a general
    transposed conv, it needs no time-axis cropping to stay causal."""

    def __init__(self, in_ch, out_ch, freq_stride=2, last=False):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(
            in_ch, out_ch, kernel_size=(3, 1), stride=(freq_stride, 1),
            padding=(1, 0), output_padding=(freq_stride - 1, 0),
        )
        self.last = last
        if not last:
            self.bn = nn.BatchNorm2d(out_ch)
            self.act = nn.PReLU()

    def forward(self, x):
        x = self.deconv(x)
        if not self.last:
            x = self.act(self.bn(x))
        return x


# ---------------------------------------------------------------------------
# 4. Grouped Dual-Path RNN (intra-frame over frequency, inter-frame over time)
# ---------------------------------------------------------------------------
def _run_rnn_chunked(rnn, x, max_batch=32768):
    """Run an nn.GRU over x [N, L, C] in chunks of at most `max_batch` sequences.
    Sequences are independent, so this is numerically identical to one call; it only
    avoids cuDNN failures on very large RNN batches (the intra-frame GRU sees
    B*G*T sequences, e.g. 30*4*1250 = 150k for a full 10 s batch)."""
    if x.shape[0] <= max_batch:
        return rnn(x)[0]
    return torch.cat([rnn(c)[0] for c in x.split(max_batch, dim=0)], dim=0)


class GroupedDualPathRNNBlock(nn.Module):
    def __init__(self, channels, n_groups):
        super().__init__()
        assert channels % n_groups == 0
        self.n_groups = n_groups
        self.group_ch = channels // n_groups

        # unidirectional (causal) GRUs, as stated in the paper for G-DPRNN
        self.intra_rnn = nn.GRU(self.group_ch, self.group_ch, batch_first=True)
        self.inter_rnn = nn.GRU(self.group_ch, self.group_ch, batch_first=True)
        self.intra_proj = nn.Linear(self.group_ch, self.group_ch)
        self.inter_proj = nn.Linear(self.group_ch, self.group_ch)
        self.intra_norm = nn.LayerNorm(self.group_ch)
        self.inter_norm = nn.LayerNorm(self.group_ch)

    def forward(self, x):
        """x: [B, C, F, T] -> [B, C, F, T]"""
        B, C, Fr, T = x.shape
        G, Cg = self.n_groups, self.group_ch

        # --- intra-frame RNN: sequence axis = frequency, for each time step ---
        # [B,C,F,T] -> [B,G,Cg,F,T] -> [B*G*T, F, Cg]
        h = x.reshape(B, G, Cg, Fr, T)
        h_intra = h.permute(0, 1, 4, 3, 2).reshape(B * G * T, Fr, Cg)
        out = _run_rnn_chunked(self.intra_rnn, h_intra)
        out = self.intra_norm(self.intra_proj(out))
        out = out.reshape(B, G, T, Fr, Cg).permute(0, 1, 4, 3, 2)  # [B,G,Cg,F,T]
        h = h + out
        h = h.reshape(B, C, Fr, T)

        # --- inter-frame RNN: sequence axis = time, for each frequency bin ---
        h2 = h.reshape(B, G, Cg, Fr, T)
        h_inter = h2.permute(0, 1, 3, 4, 2).reshape(B * G * Fr, T, Cg)
        out2 = _run_rnn_chunked(self.inter_rnn, h_inter)
        out2 = self.inter_norm(self.inter_proj(out2))
        out2 = out2.reshape(B, G, Fr, T, Cg).permute(0, 1, 4, 2, 3)  # [B,G,Cg,F,T]
        h2 = h2 + out2
        h2 = h2.reshape(B, C, Fr, T)

        return h2


# ---------------------------------------------------------------------------
# 5. Full BeamFusion network
# ---------------------------------------------------------------------------
class BeamFusionNet(nn.Module):
    """Encoder: in_ch -> enc_channels[0] -> enc_channels[1] -> ... -> enc_channels[-1] (=hidden_dim),
    each stage stride-2 on frequency. Decoder mirrors the channel list back down to P, last stage
    un-activated (raw logits). No skip connections between matching encoder/decoder stages, so the
    stage count on each side doesn't need to divide the frequency axis evenly -- the final
    ERB-grid-size correction in BeamFusionPipeline/BeamFusionFromFeat.forward() absorbs any
    rounding from the stride-2 convs along the way."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        P = cfg.n_beams
        in_ch = 3 * P  # Re, Im, |.| per beam

        enc_ch = [in_ch, *cfg.enc_channels]                     # e.g. [15, 16, 32, 128]
        self.encoders = nn.ModuleList(
            [EncoderBlock(enc_ch[i], enc_ch[i + 1], freq_stride=2) for i in range(len(enc_ch) - 1)]
        )

        self.dprnn_blocks = nn.ModuleList(
            [GroupedDualPathRNNBlock(cfg.hidden_dim, cfg.n_groups) for _ in range(cfg.n_dprnn_blocks)]
        )

        dec_ch = [*cfg.enc_channels[::-1], P]                   # e.g. [128, 32, 16, 5]
        n_dec = len(dec_ch) - 1
        self.decoders = nn.ModuleList(
            [DecoderBlock(dec_ch[i], dec_ch[i + 1], freq_stride=2, last=(i == n_dec - 1))
             for i in range(n_dec)]
        )

    def forward(self, feat_compressed, verbose=False):
        """feat_compressed: [B, 3P, F', T] real -> weight logits [B, P, ~F', T]"""
        x = feat_compressed
        for i, enc in enumerate(self.encoders):
            x = enc(x)
            if verbose:
                print(f"  after enc{i + 1}" + (" (encoder out)" if i == len(self.encoders) - 1 else "")
                      + f"{'':8s}: {tuple(x.shape)}")

        for i, block in enumerate(self.dprnn_blocks):
            x = block(x)
            if verbose:
                print(f"  after G-DPRNN block {i}    : {tuple(x.shape)}")

        for i, dec in enumerate(self.decoders):
            x = dec(x)
            if verbose:
                print(f"  after dec{i + 1}" + (" (decoder out)" if i == len(self.decoders) - 1 else "")
                      + f"{'':8s}: {tuple(x.shape)}")
        return x


# ---------------------------------------------------------------------------
# 6. End-to-end wrapper: STFT features -> fused complex spectrum
# ---------------------------------------------------------------------------
class BeamFusionPipeline(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.fixed_bf = FixedBeamformers(cfg)
        self.erb = ERBFilterBank(cfg)
        self.net = BeamFusionNet(cfg)

    @staticmethod
    def to_input_features(z_beams: torch.Tensor) -> torch.Tensor:
        """[B,P,F,T] complex -> [B,3P,F,T] real (Re, Im, |.| stacked per beam)"""
        B, P, Fr, T = z_beams.shape
        feat = torch.stack([z_beams.real, z_beams.imag, z_beams.abs()], dim=2)  # [B,P,3,F,T]
        return feat.reshape(B, P * 3, Fr, T)

    def forward(self, y_stft: torch.Tensor, verbose=False):
        """y_stft: [B, M, F, T] complex multichannel STFT of the mixture."""
        z_beams = self.fixed_bf(y_stft)                       # [B,P,F,T] complex
        if verbose:
            print(f"  fixed-beamformer outputs Z : {tuple(z_beams.shape)}  ({z_beams.dtype})")

        feat = self.to_input_features(z_beams)                 # [B,3P,F,T] real
        if verbose:
            print(f"  stacked Re/Im/|.| features : {tuple(feat.shape)}")

        feat_c = self.erb.compress(feat)                       # [B,3P,F',T]
        if verbose:
            print(f"  after ERB compression      : {tuple(feat_c.shape)}")

        logits_c = self.net(feat_c, verbose=verbose)            # [B,P,~F',T]

        # The strided encoder (floor division) and its mirrored transposed-
        # conv decoder don't exactly invert each other when F' isn't a power
        # of two (a very common encoder/decoder-audio-net issue). Rather than
        # hand-tune per-layer output_padding, snap the frequency axis back to
        # the exact ERB band count F' before expanding — cheap and exact.
        target_Fp = self.cfg.erb_subband_1 + self.cfg.erb_subband_2
        if logits_c.shape[-2] != target_Fp:
            logits_c = F.interpolate(
                logits_c, size=(target_Fp, logits_c.shape[-1]), mode="nearest"
            )
            if verbose:
                print(f"  (resized decoder freq axis -> F'={target_Fp} to match ERB grid)")

        logits = self.erb.expand(logits_c)                      # [B,P,F,T]
        if verbose:
            print(f"  after ERB expansion        : {tuple(logits.shape)}")

        weights = F.softmax(logits, dim=1)                      # [B,P,F,T], sum_p = 1
        if verbose:
            print(f"  fusion weights W_p(f,t)    : {tuple(weights.shape)}"
                  f"   (sum over P, mean={weights.sum(1).mean().item():.4f}, should be 1.0)")

        fused = (weights.to(z_beams.dtype) * z_beams).sum(dim=1)  # [B,F,T] complex
        if verbose:
            print(f"  fused spectrum Z_hat(f,t)  : {tuple(fused.shape)}  ({fused.dtype})")

        return fused, weights, z_beams


class BeamFusionFromFeat(nn.Module):
    """Same network as BeamFusionPipeline, but starting from the features that
    feature.py already computed (fixed beamformers + Re/Im/|.| stacking done offline).

        feat [B, 3P, F, T]  = [Re_0..Re_{P-1}, Im_0..Im_{P-1}, Mag_0..Mag_{P-1}]
        -> ERB compress -> BeamFusionNet -> ERB expand -> softmax over P -> W_p(f,t)
        -> fused = sum_p W_p * Z_p        (Z_p = Re_p + j Im_p, taken from `feat`)

    Returns fused [B, 2, F, T] (Re, Im — same layout as the target) and weights [B, P, F, T].
    No complex tensors are used, so it exports / ports to C more easily."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.erb = ERBFilterBank(cfg)
        self.net = BeamFusionNet(cfg)

    def forward(self, feat: torch.Tensor):
        P = self.cfg.n_beams
        feat_c = self.erb.compress(feat)                        # [B,3P,F',T]
        logits_c = self.net(feat_c)                             # [B,P,~F',T]
        target_Fp = self.cfg.erb_subband_1 + self.cfg.erb_subband_2
        if logits_c.shape[-2] != target_Fp:
            logits_c = F.interpolate(logits_c, size=(target_Fp, logits_c.shape[-1]), mode="nearest")
        logits = self.erb.expand(logits_c)                      # [B,P,F,T]
        weights = F.softmax(logits, dim=1)                      # sum_p W_p = 1 (distortionless)
        z_re, z_im = feat[:, :P], feat[:, P:2 * P]
        fused = torch.stack([(weights * z_re).sum(1), (weights * z_im).sum(1)], dim=1)  # [B,2,F,T]
        return fused, weights


# ---------------------------------------------------------------------------
# 7. Smoke test: one dummy multichannel input, print every stage's shape
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    cfg = Config()

    B = 1
    M = cfg.n_mics
    duration_sec = 1.0
    N = int(cfg.sample_rate * duration_sec)

    x = torch.randn(B, M, N)  # dummy multichannel time-domain mixture
    print(f"input waveform x           : {tuple(x.shape)}  (B, M, N samples)")

    # STFT per microphone channel
    window = torch.hann_window(cfg.win_length)
    y = torch.stft(
        x.reshape(B * M, N),
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        win_length=cfg.win_length,
        window=window,
        return_complex=True,
        center=True,
    )  # [B*M, F, T]
    Fbins, T = y.shape[-2], y.shape[-1]
    y = y.reshape(B, M, Fbins, T)
    print(f"multichannel STFT Y(f,l)   : {tuple(y.shape)}  (B, M, F={Fbins}, T={T})")

    pipeline = BeamFusionPipeline(cfg)
    n_params = sum(p.numel() for p in pipeline.net.parameters())
    print(f"\nBeamFusionNet params        : {n_params/1e3:.1f} K"
          f"  (paper reports ~85 K — expect a gap given the APPROX choices above)\n")

    print("forward pass (per-stage shapes):")
    fused_spec, weights, z_beams = pipeline(y, verbose=True)

    # ISTFT back to time domain, just to show the full round trip
    wav_hat = torch.istft(
        fused_spec,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        win_length=cfg.win_length,
        window=window,
        length=N,
    )
    print(f"\nenhanced waveform (ISTFT)  : {tuple(wav_hat.shape)}")
