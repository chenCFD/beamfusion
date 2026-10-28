"""
feature.py — BeamFusion feature / target extraction  (arXiv:2510.24497, Sec. 3.2 + 4.3)

Reads the wavs written by generate_dataset.py and writes, per sample, two .npy files:

    feat_{idx:06d}.npy    [3P, F, T]  real
        = concat( Re{Z_p}, Im{Z_p}, |Z_p| )  over the P fixed beamformers
          (paper: "real, imaginary and T-F magnitude spectrum components ...
           Z_in in R^{3P x T x F} constructed by concatenating these three")
          channel order: [Re_0..Re_{P-1}, Im_0..Im_{P-1}, Mag_0..Mag_{P-1}]
          P = 5 : MWNG, DMA-I (null 90 deg), DMA-II (120), DMA-III (150), DMA-IV (180)
    target_{idx:06d}.npy  [2, F, T]   real
        = Re / Im of the STFT of the reference signal X_ref (loss, Eq. 11:
          L = E || X_fi - X_ref ||^2  in the T-F domain)

Layout is [channels, F, T] so one file can be fed straight into model.py
(which uses [B, C, F, T]).  Nothing is ERB-compressed here: ERB compression is the
first layer INSIDE the network (Fig. 1b), so it stays in model.py.

Front end (paper Sec. 4.3): 16 kHz, FFT 512, window 512, "overlap 128".
  The paper's wording is ambiguous.  A frame shift of 384 (= 512 - 128) leaves the
  25 %-overlap windows with gaps (no perfect reconstruction with any sensible
  window), so the default here is hop = 128 (125 frames/s), sqrt-Hann window, which
  is COLA and is a normal frame-online setting.  Change with --hop if you decide
  otherwise (and keep model.py / the C code in sync).
  Framing is causal: frame t covers samples [t*hop - (win-hop), t*hop + hop), i.e.
  (win-hop) zeros are prepended, so every real sample is covered by a full overlap
  and the streaming C implementation can use exactly the same framing.

Beamformers (fixed, computed once, closed form):
  * MWNG : h = d(theta_s) / (d^H d)                                (delay-and-sum)
  * DMA-p: single-null LCMV  h = C (C^H C + eps I)^-1 g,  C=[d(theta_s), d(theta_null)],
           g=[1,0]^T.   APPROX: the paper cites the exact DMA formulas of
           Benesty & Chen's book; this is the standard equivalent first-order
           (one null) design, not guaranteed identical.
  All of them satisfy h^H d(theta_s) = 1 (distortionless).

Steering-vector sign (IMPORTANT, checked by --selftest):
  The reference mic is mic 0 and the mics are ordered along +x, with the target at
  +x (azimuth 0).  Mics closer to the source hear the wavefront EARLIER, so with the
  usual STFT convention X(f) = sum x[n] e^{-j 2 pi f n/fs} the steering vector is
      d_m(f) = exp(+j 2 pi f m delta cos(theta) / c).
  (Eq. (1) of the paper writes e^{-j...}; that corresponds to the opposite STFT
  sign convention / mic ordering.  --selftest builds a plane wave in the TIME domain
  and shows that +1 recovers the source and -1 does not.)

Usage
-----
  python feature.py --selftest
  python feature.py --data-root ./dataset --out-root ./features
  python feature.py --data-root ./dataset --out-root ./features --splits train test --n-jobs 8
  python feature.py --data-root ./dataset --out-root ./features --dtype float32 --limit 4
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# 0. Config
# ---------------------------------------------------------------------------
@dataclass
class FeatConfig:
    sample_rate: int = 16_000
    n_fft: int = 512
    win_length: int = 512
    hop_length: int = 128            # see module docstring ("overlap 128" ambiguity)

    # array (must match generate_dataset.py)
    n_mics: int = 8
    mic_spacing: float = 0.01
    sound_speed: float = 343.0
    look_angle_deg: float = 0.0
    null_angles_deg: tuple = (90.0, 120.0, 150.0, 180.0)
    steering_sign: int = +1          # +1: see docstring. Only change if --selftest says so.
    lcmv_floor: float = 3e-3         # eigenvalue floor of C^H C, relative to its largest eigenvalue.
                                     # Caps ||h||^2 (noise amplification) at low freq; ~ ||h||^2 <= 1/(2*M*floor).
                                     # Smaller = deeper nulls but more low-freq noise gain; 0 = old (unregularised) LCMV.

    dtype: str = "float16"           # on-disk dtype of feature / target

    @property
    def n_freqs(self) -> int:
        return self.n_fft // 2 + 1

    @property
    def n_beams(self) -> int:
        return 1 + len(self.null_angles_deg)

    @property
    def pad_left(self) -> int:
        return self.win_length - self.hop_length


# ---------------------------------------------------------------------------
# 1. STFT / ISTFT (numpy, causal framing, sqrt-Hann)
# ---------------------------------------------------------------------------
def make_window(cfg: FeatConfig) -> np.ndarray:
    n = np.arange(cfg.win_length)
    hann = 0.5 - 0.5 * np.cos(2.0 * np.pi * n / cfg.win_length)   # periodic Hann
    return np.sqrt(hann)


def n_frames_for(n_samples: int, cfg: FeatConfig) -> int:
    total = n_samples + cfg.pad_left
    return int(np.ceil(max(total - cfg.win_length, 0) / cfg.hop_length)) + 1


def stft(x: np.ndarray, cfg: FeatConfig) -> np.ndarray:
    """x: [N] or [M, N] real  ->  [F, T] or [M, F, T] complex64."""
    squeeze = x.ndim == 1
    if squeeze:
        x = x[None]
    M, N = x.shape
    T = n_frames_for(N, cfg)
    total = (T - 1) * cfg.hop_length + cfg.win_length
    xp = np.zeros((M, total), dtype=np.float64)
    xp[:, cfg.pad_left:cfg.pad_left + N] = x
    idx = (np.arange(T)[:, None] * cfg.hop_length + np.arange(cfg.win_length)[None, :])  # [T, win]
    frames = xp[:, idx] * make_window(cfg)[None, None, :]                                # [M, T, win]
    X = np.fft.rfft(frames, n=cfg.n_fft, axis=-1)                                       # [M, T, F]
    X = np.ascontiguousarray(np.transpose(X, (0, 2, 1))).astype(np.complex64)           # [M, F, T]
    return X[0] if squeeze else X


def istft(X: np.ndarray, cfg: FeatConfig, n_samples: int) -> np.ndarray:
    """X: [F, T] complex -> [n_samples] real (inverse of stft above)."""
    F, T = X.shape
    w = make_window(cfg)
    frames = np.fft.irfft(X.T.astype(np.complex128), n=cfg.n_fft, axis=-1)[:, :cfg.win_length] * w
    total = (T - 1) * cfg.hop_length + cfg.win_length
    y = np.zeros(total)
    norm = np.zeros(total)
    for t in range(T):
        s = t * cfg.hop_length
        y[s:s + cfg.win_length] += frames[t]
        norm[s:s + cfg.win_length] += w ** 2
    y /= np.maximum(norm, 1e-8)
    return y[cfg.pad_left:cfg.pad_left + n_samples].astype(np.float32)


# ---------------------------------------------------------------------------
# 2. Fixed beamformers  (MWNG + 4 single-null LCMV "DMAs")
# ---------------------------------------------------------------------------
def steering_vector(angle_deg: float, cfg: FeatConfig) -> np.ndarray:
    """[M, F] complex, element 0 == 1 (mic 0 is the phase reference)."""
    freqs = np.fft.rfftfreq(cfg.n_fft, d=1.0 / cfg.sample_rate)          # [F]
    tau0 = cfg.mic_spacing / cfg.sound_speed
    m = np.arange(cfg.n_mics)[:, None]
    phase = cfg.steering_sign * 2j * np.pi * freqs[None, :] * m * tau0 * np.cos(np.deg2rad(angle_deg))
    return np.exp(phase)


def design_beamformers(cfg: FeatConfig) -> np.ndarray:
    """H: [P, M, F] complex128, applied as Z = h^H y.

    DMA-p: h = C A^-1 g, A = C^H C with its eigenvalues floored at
    lcmv_floor * lambda_max (per frequency bin).  At low frequency d_look and d_null are almost
    parallel, so lambda_min -> 0 and ||h||^2 grows ~ 1/f^2 (white-noise gain -28 dB at 31 Hz);
    the floor bounds that, and the beam degrades gracefully towards MWNG-like behaviour.
    h is re-scaled afterwards so that h^H d_look = 1 (distortionless) still holds exactly.
    """
    d_look = steering_vector(cfg.look_angle_deg, cfg)                    # [M, F]
    h_mwng = d_look / np.sum(np.abs(d_look) ** 2, axis=0, keepdims=True)  # [M, F]
    H = [h_mwng]
    g = np.array([1.0, 0.0], dtype=np.complex128)
    for null_deg in cfg.null_angles_deg:
        d_null = steering_vector(null_deg, cfg)
        C = np.stack([d_look, d_null], axis=-1).transpose(1, 0, 2)       # [F, M, 2]
        A = np.einsum("fmi,fmj->fij", C.conj(), C)                       # [F, 2, 2] Hermitian
        lam, V = np.linalg.eigh(A)                                       # ascending; lam [F,2], V [F,2,2]
        lam = np.maximum(lam, max(cfg.lcmv_floor, 1e-12) * lam[:, -1:])  # eigenvalue flooring (DC bin has lam_min = 0)
        A_inv = np.einsum("fik,fk,fjk->fij", V, 1.0 / lam, V.conj())     # V diag(1/lam) V^H
        w = np.einsum("fij,j->fi", A_inv, g)                             # [F, 2]
        h = np.einsum("fmi,fi->mf", C, w)                                # [M, F]
        gain = np.einsum("mf,mf->f", h.conj(), d_look)                   # h^H d_look (== 1 without flooring)
        h = h / np.conj(gain)[None, :]                                   # restore distortionless response
        H.append(h)
    return np.stack(H, axis=0)                                           # [P, M, F]


def apply_beamformers(H: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """H [P,M,F], Y [M,F,T] -> Z [P,F,T] complex64   (Z = h^H y)."""
    return np.einsum("pmf,mft->pft", H.conj(), Y.astype(np.complex128)).astype(np.complex64)


# ---------------------------------------------------------------------------
# 3. Feature / target for one sample
# ---------------------------------------------------------------------------
def make_feature(mixture: np.ndarray, H: np.ndarray, cfg: FeatConfig) -> np.ndarray:
    """mixture [M, N] -> feature [3P, F, T] float32  (Re, Im, |.|)."""
    Y = stft(mixture, cfg)                    # [M, F, T]
    Z = apply_beamformers(H, Y)               # [P, F, T]
    return np.concatenate([Z.real, Z.imag, np.abs(Z)], axis=0).astype(np.float32)


def make_target(reference: np.ndarray, cfg: FeatConfig) -> np.ndarray:
    """reference [N] -> target [2, F, T] float32  (Re, Im)."""
    X = stft(reference, cfg)
    return np.stack([X.real, X.imag], axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# 4. I/O
# ---------------------------------------------------------------------------
def read_wav(path: Path) -> np.ndarray:
    """Returns float32, shape [N] (mono) or [N, C]."""
    try:
        import soundfile as sf
        data, _ = sf.read(str(path), dtype="float32", always_2d=False)
        return data
    except ImportError:
        from scipy.io import wavfile
        _, data = wavfile.read(str(path))
        if data.dtype == np.int16:
            return (data.astype(np.float32) / 32768.0)
        if data.dtype == np.int32:
            return (data.astype(np.float32) / 2147483648.0)
        return data.astype(np.float32)


def find_samples(split_dir: Path, ref_prefix: str = "reference"):
    """[(idx, mixture_path, reference_path)] from mixture_XXXXXX.wav / <ref_prefix>_XXXXXX.wav."""
    out = []
    for mp in sorted(split_dir.glob("mixture_*.wav")):
        m = re.match(r"mixture_(\d+)\.wav$", mp.name)
        if not m:
            continue
        rp = split_dir / f"{ref_prefix}_{m.group(1)}.wav"
        if rp.exists():
            out.append((int(m.group(1)), mp, rp))
    return out


def _cast(x: np.ndarray, dtype: str, name: str) -> np.ndarray:
    if dtype == "float16":
        peak = float(np.max(np.abs(x))) if x.size else 0.0
        if peak > 6.0e4:
            raise ValueError(f"{name}: peak {peak:.3g} overflows float16; re-run with --dtype float32")
    return x.astype(np.dtype(dtype))


_H_CACHE = {}


def _get_H(cfg: FeatConfig) -> np.ndarray:
    key = json.dumps(asdict(cfg), sort_keys=True, default=list)
    if key not in _H_CACHE:
        _H_CACHE[key] = design_beamformers(cfg)
    return _H_CACHE[key]


def process_one(task):
    idx, mix_path, ref_path, out_dir, cfg, targets_only = task
    mixture = read_wav(mix_path)                     # [N, M]
    reference = read_wav(ref_path)                   # [N]
    if mixture.ndim != 2 or mixture.shape[1] != cfg.n_mics:
        raise ValueError(f"{mix_path}: expected [N,{cfg.n_mics}], got {mixture.shape}")
    if reference.ndim != 1 or reference.shape[0] != mixture.shape[0]:
        raise ValueError(f"{ref_path}: length/shape mismatch with mixture ({reference.shape} vs {mixture.shape})")

    feat_path = out_dir / f"feat_{idx:06d}.npy"
    targ_path = out_dir / f"target_{idx:06d}.npy"
    targ = make_target(reference, cfg)               # [2, F, T]
    if targets_only:
        # keep the existing feat_*.npy (they only depend on the mixture); just rewrite the target
        if not feat_path.exists():
            raise FileNotFoundError(f"--targets-only but {feat_path} does not exist")
        n_frames = int(np.load(feat_path, mmap_mode="r").shape[-1])
        if targ.shape[-1] != n_frames:
            raise ValueError(f"sample {idx}: target has {targ.shape[-1]} frames, existing feat has {n_frames}")
        if not np.isfinite(targ).all():
            raise ValueError(f"sample {idx}: NaN/Inf in target")
        np.save(targ_path, _cast(targ, cfg.dtype, f"target_{idx}"))
    else:
        H = _get_H(cfg)
        feat = make_feature(mixture.T, H, cfg)       # [3P, F, T]
        if not (np.isfinite(feat).all() and np.isfinite(targ).all()):
            raise ValueError(f"sample {idx}: NaN/Inf in features")
        np.save(feat_path, _cast(feat, cfg.dtype, f"feat_{idx}"))
        np.save(targ_path, _cast(targ, cfg.dtype, f"target_{idx}"))
        n_frames = int(feat.shape[-1])
    return {
        "sample_idx": idx,
        "n_samples": int(mixture.shape[0]),
        "n_frames": n_frames,
        "feat_path": str(feat_path),
        "target_path": str(targ_path),
        "mixture_path": str(mix_path),
        "reference_path": str(ref_path),
    }


def run_split(split: str, data_root: Path, out_root: Path, cfg: FeatConfig, n_jobs: int, limit: int | None,
              ref_prefix: str = "reference", targets_only: bool = False):
    samples = find_samples(data_root / split, ref_prefix)
    if not samples:
        print(f"[{split}] no mixture_*.wav / reference_*.wav pairs under {data_root / split} — skipped")
        return
    if limit:
        samples = samples[:limit]
    out_dir = out_root / split
    out_dir.mkdir(parents=True, exist_ok=True)

    T = n_frames_for(160_000, cfg)
    per = (3 * cfg.n_beams + 2) * cfg.n_freqs * T * np.dtype(cfg.dtype).itemsize / 1e6
    print(f"[{split}] {len(samples)} samples, ~{per:.1f} MB/sample for a 10 s clip "
          f"-> ~{per * len(samples) / 1e3:.1f} GB total ({cfg.dtype})")

    tasks = [(i, mp, rp, out_dir, cfg, targets_only) for i, mp, rp in samples]
    rows = []
    if n_jobs <= 1:
        for k, t in enumerate(tasks, 1):
            rows.append(process_one(t))
            if k % 50 == 0 or k == len(tasks):
                print(f"[{split}] {k}/{len(tasks)}")
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as pool:
            futs = [pool.submit(process_one, t) for t in tasks]
            for k, f in enumerate(as_completed(futs), 1):
                rows.append(f.result())
                if k % 50 == 0 or k == len(tasks):
                    print(f"[{split}] {k}/{len(tasks)}")

    rows.sort(key=lambda r: r["sample_idx"])
    with open(out_root / f"feature_manifest_{split}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[{split}] done -> {out_dir}  (+ feature_manifest_{split}.csv)")


# ---------------------------------------------------------------------------
# 5. Self-test (no dataset needed)
# ---------------------------------------------------------------------------
def _frac_advance(s: np.ndarray, adv_samples: float, half_len: int = 64) -> np.ndarray:
    """y[n] = s[n + adv]  via windowed-sinc, time domain (no FFT-sign convention involved)."""
    k = np.arange(-half_len, half_len + 1)
    h = np.sinc(adv_samples - k) * np.hanning(2 * half_len + 1)   # s(n+adv) = sum_k s[n+k] sinc(adv-k)
    # y[n] = sum_k s[n + k] * h[k]
    return np.convolve(s, h[::-1], mode="same")


def selftest():
    cfg = FeatConfig()
    ok = True
    print(f"P={cfg.n_beams} beams, F={cfg.n_freqs}, win={cfg.win_length}, hop={cfg.hop_length}")

    # (a) STFT / ISTFT perfect reconstruction
    rng = np.random.default_rng(0)
    x = rng.standard_normal(16000).astype(np.float32)
    err = np.max(np.abs(istft(stft(x, cfg), cfg, len(x)) - x))
    print(f"(a) STFT->ISTFT max abs error: {err:.2e}")
    ok &= err < 1e-4

    # (b) distortionless + null depth, from the weights alone
    H = design_beamformers(cfg)
    fbins = np.arange(4, cfg.n_freqs)      # skip DC-ish bins
    names = ["MWNG", "DMA-I", "DMA-II", "DMA-III", "DMA-IV"]
    print("(b) |h^H d(theta)| (mean over bins>=4, dB):   theta:  0     90    120   150   180")
    for p in range(cfg.n_beams):
        row = []
        for th in (0, 90, 120, 150, 180):
            d = steering_vector(th, cfg)
            g = np.abs(np.einsum("mf,mf->f", H[p].conj(), d))[fbins]
            row.append(20 * np.log10(np.mean(g) + 1e-12))
            if th == 0:
                ok &= np.allclose(g, 1.0, atol=1e-6)
        print(f"    {names[p]:8s}" + " ".join(f"{v:7.1f}" for v in row))
    # each DMA must have a deep null at its own angle -- except at low frequency, where the
    # eigenvalue floor (lcmv_floor) deliberately gives up the null to bound the noise gain
    nbins = np.arange(16, cfg.n_freqs)                # >= 500 Hz
    for p, th in zip(range(1, cfg.n_beams), cfg.null_angles_deg):
        d = steering_vector(th, cfg)
        g = np.abs(np.einsum("mf,mf->f", H[p].conj(), d))
        ok &= g[nbins].max() < 1e-3
        print(f"    {names[p]:8s} null depth @125/250/500/1000 Hz [dB]: " +
              " ".join(f"{20 * np.log10(g[int(round(fh / cfg.sample_rate * cfg.n_fft))] + 1e-12):7.1f}"
                       for fh in (125, 250, 500, 1000)))
    print("    max ||h||^2 over bins (noise gain, lower is safer): " +
          " ".join(f"{np.sum(np.abs(H[p]) ** 2, axis=0).max():.1f}" for p in range(cfg.n_beams)))

    # white-noise gain (how much sensor noise each beam amplifies)
    print("    white-noise gain 10log10(1/||h||^2) [dB] @ 200 / 1000 / 4000 Hz:")
    for p in range(cfg.n_beams):
        vals = []
        for fhz in (200, 1000, 4000):
            fb = int(round(fhz / cfg.sample_rate * cfg.n_fft))
            vals.append(10 * np.log10(1.0 / np.sum(np.abs(H[p][:, fb]) ** 2)))
        print(f"    {names[p]:8s}" + " ".join(f"{v:7.1f}" for v in vals))

    # (c) time-domain plane wave from +x (0 deg): mic m hears it EARLIER by m*tau0
    N = 32000
    s = rng.standard_normal(N).astype(np.float32)
    tau_samp = cfg.mic_spacing / cfg.sound_speed * cfg.sample_rate
    mix = np.stack([_frac_advance(s, m * tau_samp) for m in range(cfg.n_mics)])      # [M, N]
    ref = mix[0]
    X_ref = stft(ref, cfg)
    lo, hi = 8, 200                                   # ~250 Hz .. 6.2 kHz, away from edges
    sl = slice(lo, hi)
    for sign in (+1, -1):
        cfg_s = FeatConfig(steering_sign=sign)
        Z = apply_beamformers(design_beamformers(cfg_s), stft(mix, cfg_s))            # [P, F, T]
        rel = [np.linalg.norm(Z[p, sl, 10:-10] - X_ref[sl, 10:-10]) / np.linalg.norm(X_ref[sl, 10:-10])
               for p in range(cfg_s.n_beams)]
        print(f"(c) steering_sign={sign:+d}: rel. error of Z_p vs STFT(mic0)  " +
              " ".join(f"{r:.3f}" for r in rel))
        if sign == cfg.steering_sign:
            ok &= max(rel) < 0.05

    # (d) the same plane wave from 90 deg (broadside, no inter-mic delay) is suppressed by DMA-I
    mix90 = np.stack([s] * cfg.n_mics)
    Z90 = apply_beamformers(H, stft(mix90, cfg))
    X90 = stft(s, cfg)
    sl = slice(16, 200)                               # >= 500 Hz: below that the null is floored on purpose
    att = [20 * np.log10(np.linalg.norm(Z90[p, sl, 10:-10]) / np.linalg.norm(X90[sl, 10:-10]) + 1e-12)
           for p in range(cfg.n_beams)]
    print("(d) level of a 90-deg source after each beam (dB, rel. to input): " +
          " ".join(f"{a:.1f}" for a in att))
    ok &= att[1] < -40

    # (e) shapes
    feat = make_feature(mix, H, cfg)
    targ = make_target(ref, cfg)
    print(f"(e) feature {feat.shape}, target {targ.shape}")
    ok &= feat.shape[0] == 3 * cfg.n_beams and targ.shape[0] == 2 and feat.shape[1:] == targ.shape[1:]

    print("\nSELFTEST", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# 6. CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="BeamFusion feature/target extraction")
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--data-root", type=Path, help="folder produced by generate_dataset.py (has train/ and test/)")
    p.add_argument("--out-root", type=Path, default=None, help="default: <data-root>/features")
    p.add_argument("--splits", nargs="+", default=["train", "test"])
    p.add_argument("--n-jobs", type=int, default=4)
    p.add_argument("--limit", type=int, default=None, help="only the first N samples per split (smoke test)")
    p.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    p.add_argument("--hop", type=int, default=128)
    p.add_argument("--n-fft", type=int, default=512)
    p.add_argument("--ref-name", default="reference",
                   help="prefix of the reference wavs used for the TARGET: 'reference' (reverberant image, default) "
                        "or 'reference_direct' (made by direct_ref.py)")
    p.add_argument("--targets-only", action="store_true",
                   help="only rewrite target_*.npy (keep existing feat_*.npy and feature_config.json) -- fast re-target")
    return p.parse_args()


def main():
    args = parse_args()
    if args.selftest:
        sys.exit(selftest())
    if args.data_root is None:
        sys.exit("--data-root is required (or use --selftest)")

    cfg = FeatConfig(n_fft=args.n_fft, win_length=args.n_fft, hop_length=args.hop, dtype=args.dtype)
    out_root = args.out_root or (args.data_root / "features")
    out_root.mkdir(parents=True, exist_ok=True)
    if args.targets_only:
        # reuse the config the existing feat files were made with (do not touch it)
        d = json.load(open(out_root / "feature_config.json"))
        known = {f.name for f in fields(FeatConfig)}
        cfg = FeatConfig(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in d.items() if k in known})
    else:
        with open(out_root / "feature_config.json", "w") as f:
            json.dump(asdict(cfg), f, indent=2, default=list)

    for split in args.splits:
        run_split(split, args.data_root, out_root, cfg, args.n_jobs, args.limit, args.ref_name, args.targets_only)


if __name__ == "__main__":
    main()
