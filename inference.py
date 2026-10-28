"""
inference.py — run BeamFusion and every comparison method of the paper (arXiv:2510.24497,
Sec. 4.2 / Table 1) on 8-channel mixtures, save the enhanced wavs, and report averaged metrics.

Methods (same order as Table 1)
  Noisy      reference-microphone (mic 0) mixture, unprocessed
  MWNG       delay-and-sum style fixed beamformer
  DMA-I..IV  single-null fixed beamformers, nulls at 90 / 120 / 150 / 180 deg
  ACC        adaptive convex combination of the 5 fixed beams (exponentiated-gradient update)
  BeamFusion the trained network from train.py (best.pt)

Two references (--ref-type, default both)
  reverb   reference_*.wav : reverberant target image at mic 0 (what train.py was trained against)
  direct   direct-path-only reference rebuilt from dry_target_*.wav by direct_ref.py (the paper's
           text says the output should approximate the "direct sound component").  Also prints
           the mean DRR so you can see how much of the reference is reverberation.

Metrics (mean over all processed samples; Table 1 columns)
  dSNR     output SNR - input SNR.  SNR = 10log10(|ref|^2 / |est - ref|^2), `ref` being the
           reverberant target image at mic 0 that generate_dataset.py saved as reference_*.wav.
  STOI     pystoi.stoi(ref, est)  (needs `pip install pystoi`; otherwise skipped with a warning)
  SI-SDR   absolute scale-invariant SDR vs `ref`, and dSI-SDR = SI-SDR(est) - SI-SDR(noisy)

Things that are NOT identical to the paper (be careful when comparing numbers)
  * ACC: the exact update rule / step-size of ref. [21] is not public in the paper; this is a
    standard per-bin normalised exponentiated-gradient convex combination (minimise output power,
    weights on the simplex).  Tune with --acc-mu / --acc-floor.
  * dSNR: the paper does not define how the SNR of the nonlinear BeamFusion output is measured.
    generate_dataset.py only stores mixture + reference (no separate interference/noise
    components), so SNR here is the waveform-residual definition above.
  * The paper's tables use fixed T60 = 300 / 700 ms; this dataset draws T60 uniformly in
    [0.2, 0.8] s.  If manifest_<split>.csv is next to the data, results are also reported per
    T60 bin (--t60-edges).
  * Table 2 (extra fixed interferer) and Fig. 2 (SIR vs interference angle, BSS-eval) need
    differently generated test sets and are not covered here.

Usage
-----
  # batch: whole test split, averaged metrics, first 5 samples' wavs saved
  python inference.py --data-root ./dataset --split test --feat-root /mnt/g/beamfusion/features \
        --ckpt ./runs/beamfusion/best.pt --out-dir ./results

  # quick check on 20 samples, save every wav, skip the slow STOI
  python inference.py --data-root ./dataset --limit 20 --save-wavs -1 --no-stoi ...

  # one file (reference optional; with it you also get metrics)
  python inference.py --mixture ./some_mixture.wav --reference ./some_reference.wav \
        --feat-root /mnt/g/beamfusion/features --ckpt ./runs/beamfusion/best.pt --out-dir ./results_one

  # classical methods only (no torch / checkpoint needed)
  python inference.py --data-root ./dataset --no-beamfusion
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import fields
from pathlib import Path

import numpy as np

import feature as FE
from direct_ref import build_direct

METHODS = ["Noisy", "MWNG", "DMA-I", "DMA-II", "DMA-III", "DMA-IV", "ACC", "BeamFusion"]

# Table 1 of the paper, T60 = 300 ms (dSNR, STOI, dSI-SDR) -- printed only as a rough yardstick.
PAPER_T60_300 = {
    "Noisy": (None, 0.58, None), "MWNG": (0.28, 0.62, 0.22), "DMA-I": (5.31, 0.69, 5.50),
    "DMA-II": (6.02, 0.71, 6.16), "DMA-III": (4.58, 0.69, 4.68), "DMA-IV": (4.95, 0.69, 5.06),
    "ACC": (6.56, 0.73, 6.61), "BeamFusion": (10.24, 0.74, 7.13),
}


# ---------------------------------------------------------------------------
# config / io
# ---------------------------------------------------------------------------
def load_fcfg(feat_root: Path | None) -> FE.FeatConfig:
    """FeatConfig identical to the one the training features were made with (feature_config.json)."""
    if feat_root is None:
        print("[warn] --feat-root not given: using FeatConfig defaults "
              "(must match the features the model was trained on, incl. lcmv_floor).")
        return FE.FeatConfig()
    p = Path(feat_root) / "feature_config.json"
    d = json.load(open(p))
    known = {f.name for f in fields(FE.FeatConfig)}
    kw = {k: (tuple(v) if isinstance(v, list) else v) for k, v in d.items() if k in known}
    if "lcmv_floor" not in d:
        print("[warn] feature_config.json has no lcmv_floor -> these features were made with the OLD "
              "(unregularised) DMA design; the DMAs used here (current feature.py) will not match them.")
    return FE.FeatConfig(**kw)


def read_multichannel(path: Path) -> np.ndarray:
    x = FE.read_wav(path)
    return x[None] if x.ndim == 1 else np.ascontiguousarray(x.T)          # [M, N]


def write_wav(path: Path, x: np.ndarray, sr: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.asarray(x, dtype=np.float32)
    try:
        import soundfile as sf
        sf.write(str(path), x, sr, subtype="FLOAT")
    except ImportError:
        from scipy.io import wavfile
        wavfile.write(str(path), sr, x)


def read_t60(data_root: Path, split: str) -> dict[int, float]:
    p = data_root / f"manifest_{split}.csv"
    if not p.exists():
        return {}
    with open(p, newline="") as f:
        return {int(r["sample_idx"]): float(r["t60_s"]) for r in csv.DictReader(f)}


# ---------------------------------------------------------------------------
# ACC (adaptive convex combination, exponentiated gradient) -- APPROX of ref. [21]
# ---------------------------------------------------------------------------
def acc_fuse(Z: np.ndarray, mu: float = 0.2, floor: float = 1e-3) -> np.ndarray:
    """Z [P,F,T] complex (fixed-beam outputs) -> fused [F,T] complex64.

    Per frequency bin, weights alpha live on the simplex (so the result stays distortionless).
    Online: frame t is output with alpha(t); then alpha is updated by an exponentiated-gradient
    step on the output power |sum_p alpha_p Z_p|^2 (gradient_p = Re{conj(out) Z_p}, normalised by
    the mean beam power so mu is level independent).  `floor` mixes in a little uniform weight so
    a beam whose weight collapsed can come back (plain EG cannot recover from alpha -> 0)."""
    P, F, T = Z.shape
    Zc = Z.astype(np.complex128)
    pw = (np.abs(Zc) ** 2).mean(0) + 1e-12                                 # [F, T]
    a = np.full((P, F), 1.0 / P)
    out = np.empty((F, T), dtype=np.complex128)
    for t in range(T):
        z = Zc[:, :, t]
        o = (a * z).sum(0)
        out[:, t] = o
        g = (np.conj(o)[None, :] * z).real
        a = a * np.exp(np.clip(-mu * g / pw[:, t][None, :], -30.0, 30.0))
        a /= a.sum(0, keepdims=True)
        a = (1.0 - floor) * a + floor / P
    return out.astype(np.complex64)


# ---------------------------------------------------------------------------
# BeamFusion network wrapper (torch imported lazily)
# ---------------------------------------------------------------------------
class BeamFusionRunner:
    def __init__(self, ckpt: Path, fcfg: FE.FeatConfig, device: str | None):
        import torch
        from model import BeamFusionFromFeat, Config
        self.torch = torch
        cfg = Config()
        if fcfg.n_fft // 2 + 1 != cfg.n_freqs or fcfg.n_beams != cfg.n_beams:
            raise ValueError(f"feature config (F={fcfg.n_fft // 2 + 1}, P={fcfg.n_beams}) does not match "
                             f"model.py Config (F={cfg.n_freqs}, P={cfg.n_beams})")
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = BeamFusionFromFeat(cfg).to(self.device).eval()
        ck = torch.load(ckpt, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ck["model"])
        self.epoch = ck.get("epoch")
        n = sum(p.numel() for p in self.model.net.parameters())
        print(f"[BeamFusion] loaded {ckpt} (epoch {self.epoch}, best val {ck.get('best_val')}), "
              f"{n / 1e3:.1f}K params, device {self.device}")

    def __call__(self, Z: np.ndarray):
        """Z [P,F,T] complex -> (fused [F,T] complex64, weights [P,F,T] float32)."""
        torch = self.torch
        feat = np.concatenate([Z.real, Z.imag, np.abs(Z)], axis=0).astype(np.float32)   # [3P,F,T]
        with torch.no_grad():
            fused, w = self.model(torch.from_numpy(feat)[None].to(self.device))
        f = fused[0].float().cpu().numpy()
        return (f[0] + 1j * f[1]).astype(np.complex64), w[0].float().cpu().numpy()


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def si_sdr(est: np.ndarray, ref: np.ndarray, eps: float = 1e-8) -> float:
    est = est.astype(np.float64) - est.mean()
    ref = ref.astype(np.float64) - ref.mean()
    s = (est @ ref) / (ref @ ref + eps) * ref
    e = est - s
    return float(10 * np.log10((s @ s) / (e @ e + eps) + eps))


def snr_db(est: np.ndarray, ref: np.ndarray, eps: float = 1e-12) -> float:
    ref = ref.astype(np.float64)
    e = est.astype(np.float64) - ref
    return float(10 * np.log10((ref @ ref + eps) / (e @ e + eps)))


_STOI_WARNED = False


def stoi_score(ref: np.ndarray, est: np.ndarray, sr: int) -> float:
    global _STOI_WARNED
    try:
        from pystoi import stoi
    except ImportError:
        if not _STOI_WARNED:
            print("[warn] pystoi not installed -> STOI reported as nan (pip install pystoi)")
            _STOI_WARNED = True
        return float("nan")
    return float(stoi(ref, est, sr, extended=False))


# ---------------------------------------------------------------------------
# one sample
# ---------------------------------------------------------------------------
def enhance(mixture: np.ndarray, fcfg: FE.FeatConfig, H: np.ndarray, runner, args):
    """mixture [M,N] -> ({method: waveform [N]}, fusion weights [P,F,T] or None)."""
    N = mixture.shape[1]
    Y = FE.stft(mixture, fcfg)                                             # [M,F,T]
    Z = FE.apply_beamformers(H, Y)                                         # [P,F,T]
    out = {"Noisy": mixture[args.ref_mic].astype(np.float32)}
    for name, z in zip(METHODS[1:5 + 1], Z):                               # MWNG, DMA-I..IV
        out[name] = FE.istft(z, fcfg, N)
    if not args.no_acc:
        out["ACC"] = FE.istft(acc_fuse(Z, args.acc_mu, args.acc_floor), fcfg, N)
    weights = None
    if runner is not None:
        fused, weights = runner(Z)
        out["BeamFusion"] = FE.istft(fused, fcfg, N)
    return out, weights


def score(outs: dict, ref: np.ndarray, sr: int, use_stoi: bool) -> dict:
    snr_in, sd_in = snr_db(outs["Noisy"], ref), si_sdr(outs["Noisy"], ref)
    res = {}
    for name, est in outs.items():
        n = min(len(est), len(ref))
        est, r = est[:n], ref[:n]
        sd = si_sdr(est, r)
        res[name] = {
            "snr": snr_db(est, r), "dsnr": snr_db(est, r) - snr_in if name != "Noisy" else float("nan"),
            "sisdr": sd, "dsisdr": sd - sd_in if name != "Noisy" else float("nan"),
            "stoi": stoi_score(r, est, sr) if use_stoi else float("nan"),
        }
    return res


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------
def nanmean(v):
    v = [x for x in v if x == x]
    return float(np.mean(v)) if v else float("nan")


def fmt(x, nd=2):
    return "   -  " if x != x else f"{x:6.{nd}f}"


def summary_table(rows: list[dict], methods: list[str], title: str, paper: bool) -> str:
    lines = [f"\n{title}  (n={len({r['idx'] for r in rows})} samples)",
             f"{'Method':11s}{'dSNR(dB)':>10s}{'STOI':>8s}{'SI-SDR(dB)':>12s}{'dSI-SDR(dB)':>13s}"
             + (f"   | paper T60=300: dSNR  STOI  dSI-SDR" if paper else "")]
    for m in methods:
        rr = [r for r in rows if r["method"] == m]
        if not rr:
            continue
        line = (f"{m:11s}{fmt(nanmean([r['dsnr'] for r in rr])):>10s}{fmt(nanmean([r['stoi'] for r in rr]), 3):>8s}"
                f"{fmt(nanmean([r['sisdr'] for r in rr])):>12s}{fmt(nanmean([r['dsisdr'] for r in rr])):>13s}")
        if paper and m in PAPER_T60_300:
            a, b, c = PAPER_T60_300[m]
            line += f"   | {fmt(a) if a is not None else '   -  '}  {b:5.2f}  {fmt(c) if c is not None else '   -  '}"
        lines.append(line)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="BeamFusion + baselines: inference and metrics")
    g = p.add_argument_group("input (batch OR single file)")
    g.add_argument("--data-root", type=Path, help="folder from generate_dataset.py (has <split>/mixture_*.wav, reference_*.wav)")
    g.add_argument("--split", default="test")
    g.add_argument("--limit", type=int, default=None, help="only the first N samples")
    g.add_argument("--mixture", type=Path, help="single multichannel wav [N, M] instead of --data-root")
    g.add_argument("--reference", type=Path, default=None, help="reference wav for --mixture (enables metrics)")
    g.add_argument("--dry", type=Path, default=None, help="dry_target wav for --mixture (needed for the direct reference)")
    p.add_argument("--ref-type", choices=["reverb", "direct", "both"], default="both")
    p.add_argument("--feat-root", type=Path, default=None, help="feature.py out-root (reads feature_config.json) -- recommended")
    p.add_argument("--ckpt", type=Path, default=Path("./runs/beamfusion/best.pt"))
    p.add_argument("--no-beamfusion", action="store_true", help="skip the network (no torch / ckpt needed)")
    p.add_argument("--no-acc", action="store_true")
    p.add_argument("--acc-mu", type=float, default=0.2)
    p.add_argument("--acc-floor", type=float, default=1e-3)
    p.add_argument("--no-stoi", action="store_true", help="skip STOI (the slowest metric)")
    p.add_argument("--ref-mic", type=int, default=0)
    p.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    p.add_argument("--out-dir", type=Path, default=Path("./results"))
    p.add_argument("--save-wavs", type=int, default=5, help="save wavs for the first N samples (-1 = all, 0 = none)")
    p.add_argument("--t60-edges", type=float, nargs="+", default=[0.2, 0.4, 0.6, 0.8001],
                   help="T60 bin edges (s) for the per-reverberation tables (needs manifest_<split>.csv)")
    p.add_argument("--no-paper", action="store_true", help="do not print the paper's Table 1 numbers beside the average table")
    return p.parse_args()


def main():
    args = parse_args()
    fcfg = load_fcfg(args.feat_root)
    sr = fcfg.sample_rate
    H = FE.design_beamformers(fcfg)

    runner = None
    if not args.no_beamfusion:
        runner = BeamFusionRunner(args.ckpt, fcfg, args.device)
    methods = [m for m in METHODS if not (m == "ACC" and args.no_acc) and not (m == "BeamFusion" and runner is None)]

    if args.mixture is not None:
        samples, t60 = [(0, args.mixture, args.reference)], {}
        save_n = -1
    elif args.data_root is not None:
        samples = FE.find_samples(args.data_root / args.split)
        if not samples:
            raise FileNotFoundError(f"no mixture_*.wav / reference_*.wav pairs in {args.data_root / args.split}")
        t60 = read_t60(args.data_root, args.split)
        save_n = args.save_wavs
    else:
        raise SystemExit("give --data-root (batch) or --mixture (single file)")
    if args.limit:
        samples = samples[:args.limit]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows, w_sum, w_n, drrs = [], None, 0, []
    t0 = time.time()
    for k, (idx, mp, rp) in enumerate(samples):
        mix = read_multichannel(mp)
        if mix.shape[0] != fcfg.n_mics:
            raise ValueError(f"{mp.name}: {mix.shape[0]} channels, expected {fcfg.n_mics}")
        outs, weights = enhance(mix, fcfg, H, runner, args)
        ref = FE.read_wav(rp) if rp is not None else None

        refs = {}
        if ref is not None:
            if args.ref_type in ("reverb", "both"):
                refs["reverb"] = ref
            if args.ref_type in ("direct", "both"):
                dp = args.dry if args.mixture is not None else mp.with_name(mp.name.replace("mixture_", "dry_target_"))
                if dp is not None and Path(dp).exists():
                    direct, _, drr = build_direct(ref, FE.read_wav(dp))
                    refs["direct"] = direct
                    drrs.append(drr)
                elif k == 0:
                    print(f"[warn] {dp} not found -> direct-path reference skipped")

        if save_n < 0 or k < save_n:
            d = args.out_dir / "wavs" / (mp.stem if args.mixture is not None else f"{idx:06d}")
            for name, x in outs.items():
                write_wav(d / f"{name}.wav", x, sr)
            for rn, r in refs.items():
                write_wav(d / f"reference_{rn}.wav", r, sr)
        for rn, r in refs.items():
            for m, res in score(outs, r, sr, not args.no_stoi).items():
                rows.append({"idx": idx, "ref": rn, "t60": t60.get(idx, float("nan")), "method": m, **res})
        if weights is not None:
            pm = weights.mean(axis=(1, 2))
            w_sum = pm if w_sum is None else w_sum + pm
            w_n += 1
        if (k + 1) % 10 == 0 or k + 1 == len(samples):
            el = time.time() - t0
            print(f"  [{k + 1}/{len(samples)}] {el:.0f}s elapsed, ETA {el / (k + 1) * (len(samples) - k - 1):.0f}s", flush=True)

    if rows:
        with open(args.out_dir / "metrics_per_sample.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        text = ""
        for rn in ("reverb", "direct"):
            rr_ = [r for r in rows if r["ref"] == rn]
            if not rr_:
                continue
            label = {"reverb": "reference = REVERBERANT target image (train.py's target)",
                     "direct": "reference = DIRECT-PATH only (paper's wording)"}[rn]
            text += summary_table(rr_, methods, f"ALL SAMPLES | {label}", (not args.no_paper) and rn == "direct")
            if args.mixture is None and any(r["t60"] == r["t60"] for r in rr_):
                e = args.t60_edges
                for lo, hi in zip(e[:-1], e[1:]):
                    sub = [r for r in rr_ if r["t60"] == r["t60"] and lo <= r["t60"] < hi]
                    if sub:
                        text += summary_table(sub, methods, f"  [{rn}] T60 in [{lo:.1f}, {min(hi, 0.8):.1f}) s", False)
        if drrs:
            text += (f"\n\nDRR of the reverberant reference (direct / reverberation, mean over {len(drrs)} samples): "
                     f"{np.mean(drrs):+.2f} dB  (min {np.min(drrs):+.2f}, max {np.max(drrs):+.2f})")
        if w_sum is not None:
            names = ["MWNG", "DMA-I", "DMA-II", "DMA-III", "DMA-IV"][:len(w_sum)]
            text += "\n\nmean BeamFusion weight per beam (sums to 1): " + "  ".join(
                f"{n} {v / w_n:.3f}" for n, v in zip(names, w_sum))
        print(text)
        (args.out_dir / "summary.txt").write_text(text + "\n")
        summ = {rn: {m: {k: nanmean([r[k] for r in rows if r["method"] == m and r["ref"] == rn])
                          for k in ("dsnr", "stoi", "sisdr", "dsisdr")} for m in methods}
                for rn in sorted({r["ref"] for r in rows})}
        json.dump({"n_samples": len({r['idx'] for r in rows}), "mean_metrics": summ,
                   "mean_drr_db": float(np.mean(drrs)) if drrs else None,
                   "args": {k: str(v) for k, v in vars(args).items()}},
                  open(args.out_dir / "summary.json", "w"), indent=2)
        print(f"\nwrote {args.out_dir}/metrics_per_sample.csv, summary.txt, summary.json"
              + (f", wavs/ ({'all' if save_n < 0 else save_n} samples)" if save_n else ""))
    else:
        print(f"no reference given -> metrics skipped; enhanced wavs are in {args.out_dir / 'wavs'}")


if __name__ == "__main__":
    main()
