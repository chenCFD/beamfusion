"""
direct_ref.py — build a *direct-path-only* reference for BeamFusion evaluation / training.

Why
---
generate_dataset.py saves, as reference_XXXXXX.wav, the REVERBERANT target image at mic 0
(target speech through its full RIR).  The paper's loss text says the output should approximate
the "original direct sound component".  At 2 m in an 8x6x3 m room (T60 0.2-0.8 s) the source is
beyond the critical distance, so most of the energy in the reverberant image is reverberation,
which a distortionless beamformer is not supposed to (and mostly cannot) reproduce.

generate_dataset.py also saves dry_target_XXXXXX.wav (the dry speech, scaled by the same peak
normalisation as mixture/reference but NOT delay-aligned).  The direct path is simply the dry
signal, delayed by the propagation time and scaled, so it can be recovered without regenerating
anything:

  1. coarse lag    = argmax of the cross-correlation between reference and dry (0..max_lag samples)
  2. direct signal = least-squares FIR (2K+1 taps around that lag, K=8 -> +-0.5 ms) that maps dry
                     onto the reference.  The short window captures delay + fractional delay +
                     gain of the direct arrival, but not the reflections that follow it.
  3. DRR           = 10log10( |direct|^2 / |reference - direct|^2 )   (diagnostic)

CLI (writes reference_direct_XXXXXX.wav next to each reference_XXXXXX.wav, for feature.py
--ref-name reference_direct):
    python direct_ref.py --data-root /mnt/g/beamfusion/dataset --splits train test --n-jobs 8
"""

from __future__ import annotations

import argparse
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np


def build_direct(ref: np.ndarray, dry: np.ndarray, K: int = 8, max_lag: int = 800,
                 ridge: float = 1e-6):
    """ref, dry: [N] float.  Returns (direct [N] float32, lag, drr_db)."""
    n = min(len(ref), len(dry))
    ref = ref[:n].astype(np.float64)
    dry = dry[:n].astype(np.float64)

    nfft = 1 << int(np.ceil(np.log2(2 * n)))
    c = np.fft.irfft(np.fft.rfft(ref, nfft) * np.conj(np.fft.rfft(dry, nfft)), nfft)
    lag = int(np.argmax(np.abs(c[:max_lag])))                       # ref[t] ~ g * dry[t - lag]

    taps = np.arange(lag - K, lag + K + 1)
    X = np.zeros((n, taps.size))
    for j, d in enumerate(taps):
        if d >= 0:
            X[d:, j] = dry[:n - d]
        else:
            X[:n + d, j] = dry[-d:]
    A = X.T @ X
    A += ridge * np.trace(A) / taps.size * np.eye(taps.size)
    h = np.linalg.solve(A, X.T @ ref)
    direct = X @ h

    res = ref - direct
    drr = 10 * np.log10((direct @ direct + 1e-12) / (res @ res + 1e-12))
    return direct.astype(np.float32), lag, float(drr)


# ---------------------------------------------------------------------------
# CLI: write reference_direct_XXXXXX.wav
# ---------------------------------------------------------------------------
def _one(task):
    ref_path, dry_path, out_path = task
    from feature import read_wav
    ref, dry = read_wav(ref_path), read_wav(dry_path)
    direct, lag, drr = build_direct(ref, dry)
    if len(direct) < len(ref):                                      # keep the length of the reference
        direct = np.pad(direct, (0, len(ref) - len(direct)))
    try:
        import soundfile as sf
        sf.write(str(out_path), direct, 16000, subtype="FLOAT")     # float: no clipping / requantisation
    except ImportError:
        from scipy.io import wavfile
        wavfile.write(str(out_path), 16000, direct)
    return out_path.name, lag, drr


def main():
    p = argparse.ArgumentParser(description="write reference_direct_*.wav from reference_*.wav + dry_target_*.wav")
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--splits", nargs="+", default=["train", "test"])
    p.add_argument("--n-jobs", type=int, default=4)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    for split in args.splits:
        d = args.data_root / split
        tasks = []
        for rp in sorted(d.glob("reference_*.wav")):
            m = re.match(r"reference_(\d+)\.wav$", rp.name)
            if not m:
                continue
            dp = d / f"dry_target_{m.group(1)}.wav"
            if dp.exists():
                tasks.append((rp, dp, d / f"reference_direct_{m.group(1)}.wav"))
        if args.limit:
            tasks = tasks[:args.limit]
        if not tasks:
            print(f"[{split}] nothing to do under {d}")
            continue
        with ProcessPoolExecutor(max_workers=args.n_jobs) as pool:
            res = list(pool.map(_one, tasks, chunksize=8))
        lags = np.array([r[1] for r in res])
        drrs = np.array([r[2] for r in res])
        print(f"[{split}] wrote {len(res)} files | lag samples min/median/max {lags.min()}/{int(np.median(lags))}/{lags.max()}"
              f" | DRR dB mean {drrs.mean():.2f} (min {drrs.min():.2f}, max {drrs.max():.2f})")


if __name__ == "__main__":
    main()
