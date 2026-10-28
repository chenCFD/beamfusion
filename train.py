"""
train.py — train BeamFusion on the features written by feature.py.

Paper recipe (arXiv:2510.24497, Sec. 4.3) followed here:
  * loss      L = E || X_fi - X_ref ||_2^2   (Eq. 11), computed in the T-F domain on the
              fused spectrum  sum_p W_p Z_p  versus the reference STFT  (Re/Im MSE)
  * optimizer Adam, initial lr 1e-3
  * schedule  lr halved when the validation loss has not decreased for 5 consecutive
              epochs, minimum lr 1e-4
  * batch 30, 50 epochs, 10 s clips
Not in the paper (my additions, all optional / easy to turn off):
  * gradient-norm clipping (--grad-clip, default 5; 0 disables)
  * train-time random time-crop (--segment-frames, default 0 = full clip, as in the paper)

Network = model.py::BeamFusionFromFeat  (ERB compress -> encoder -> G-DPRNN -> decoder ->
ERB expand -> softmax over the P beams -> weighted sum).  The fixed beamformers are NOT run
here: feature.py already did that offline.

Validation prints, besides the loss, the SI-SDR (dB) of the fused output and of each of the
P fixed beams against the reference (everything ISTFT'd back to the time domain with exactly
the framing used by feature.py).  A working model should beat every single beam.

Usage
-----
  # smoke test (2 minutes): a few samples, 2 epochs
  python train.py --feat-root ./dataset/features --out-dir ./runs/smoke \
        --limit-train 16 --limit-val 4 --epochs 2 --batch-size 4 --num-workers 0

  # paper setting
  python train.py --feat-root ./dataset/features --out-dir ./runs/beamfusion

  # GPU memory too small for batch 30 x 10 s?  e.g.
  python train.py ... --batch-size 16            # or
  python train.py ... --segment-frames 500       # random 4 s crops (hop 128)

  python train.py --feat-root ./dataset/features --out-dir ./runs/beamfusion --resume ./runs/beamfusion/last.pt
  python train.py --feat-root ./dataset/features --out-dir ./runs/beamfusion --eval-only --ckpt ./runs/beamfusion/best.pt
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from dataset import build_test, build_train_val, load_feature_config, make_loader
from model import BeamFusionFromFeat, Config

BEAM_NAMES = ["MWNG", "DMA-I", "DMA-II", "DMA-III", "DMA-IV"]


# ---------------------------------------------------------------------------
# loss / metrics
# ---------------------------------------------------------------------------
def tf_mse_loss(fused: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Eq. (11): mean over (batch, F, T) of |X_fi - X_ref|^2 ; fused/target: [B, 2, F, T] (Re, Im)."""
    return ((fused - target) ** 2).sum(dim=1).mean()


def istft_ri(re: torch.Tensor, im: torch.Tensor, fcfg: dict) -> torch.Tensor:
    """Inverse of feature.py::stft.  re/im: [N, F, T] -> waveform [N, T*hop] (float32).
    sqrt-Hann synthesis window, overlap-add, divided by the summed squared window."""
    n_fft, win, hop = fcfg["n_fft"], fcfg["win_length"], fcfg["hop_length"]
    pad_left = win - hop
    N, _, T = re.shape
    n = torch.arange(win, device=re.device, dtype=torch.float32)
    w = torch.sqrt(0.5 - 0.5 * torch.cos(2.0 * math.pi * n / win))            # [win]
    X = torch.complex(re.float(), im.float())
    frames = torch.fft.irfft(X, n=n_fft, dim=1)[:, :win, :] * w[None, :, None]  # [N, win, T]
    total = (T - 1) * hop + win
    y = F.fold(frames, output_size=(1, total), kernel_size=(1, win), stride=(1, hop))      # [N,1,1,total]
    norm = F.fold((w ** 2)[None, :, None].expand(1, win, T).contiguous(),
                  output_size=(1, total), kernel_size=(1, win), stride=(1, hop))           # [1,1,1,total]
    y = (y / norm.clamp_min(1e-8)).reshape(N, total)
    return y[:, pad_left:pad_left + T * hop]


def si_sdr(est: torch.Tensor, ref: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Scale-invariant SDR in dB, per row.  est/ref: [N, L]."""
    est = est - est.mean(dim=-1, keepdim=True)
    ref = ref - ref.mean(dim=-1, keepdim=True)
    alpha = (est * ref).sum(-1, keepdim=True) / (ref.pow(2).sum(-1, keepdim=True) + eps)
    s = alpha * ref
    e = est - s
    return 10.0 * torch.log10(s.pow(2).sum(-1) / (e.pow(2).sum(-1) + eps) + eps)


@torch.no_grad()
def evaluate(model, loader, device, fcfg, P: int, with_sisdr: bool = True):
    model.eval()
    tot_loss, tot_n = 0.0, 0
    sd_fused, sd_beams = 0.0, torch.zeros(P)
    for feat, targ in loader:
        feat, targ = feat.to(device, non_blocking=True), targ.to(device, non_blocking=True)
        fused, _ = model(feat)
        B = feat.shape[0]
        tot_loss += tf_mse_loss(fused, targ).item() * B
        tot_n += B
        if with_sisdr:
            ref_w = istft_ri(targ[:, 0], targ[:, 1], fcfg)                       # [B, L]
            fus_w = istft_ri(fused[:, 0], fused[:, 1], fcfg)                     # [B, L]
            sd_fused += si_sdr(fus_w, ref_w).sum().item()
            re = feat[:, :P].reshape(B * P, *feat.shape[2:])
            im = feat[:, P:2 * P].reshape(B * P, *feat.shape[2:])
            beam_w = istft_ri(re, im, fcfg).reshape(B, P, -1)                    # [B, P, L]
            sd_beams += si_sdr(beam_w.reshape(B * P, -1),
                               ref_w[:, None].expand(B, P, -1).reshape(B * P, -1)
                               ).reshape(B, P).sum(0).cpu()
    out = {"loss": tot_loss / max(tot_n, 1)}
    if with_sisdr and tot_n:
        out["sisdr_fused"] = sd_fused / tot_n
        out["sisdr_beams"] = (sd_beams / tot_n).tolist()
    return out


def fmt_metrics(m: dict, P: int) -> str:
    s = f"loss {m['loss']:.5f}"
    if "sisdr_fused" in m:
        names = BEAM_NAMES if P == len(BEAM_NAMES) else [f"beam{i}" for i in range(P)]
        beams = " ".join(f"{n}:{v:.2f}" for n, v in zip(names, m["sisdr_beams"]))
        s += f" | SI-SDR dB  fused {m['sisdr_fused']:.2f} | {beams}"
    return s


# ---------------------------------------------------------------------------
# misc
# ---------------------------------------------------------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def check_compat(cfg: Config, fcfg: dict, feat_shape):
    if fcfg["n_fft"] // 2 + 1 != cfg.n_freqs:
        raise ValueError(f"feature n_fft={fcfg['n_fft']} -> F={fcfg['n_fft'] // 2 + 1} but model.py "
                         f"Config.n_freqs={cfg.n_freqs}")
    P = 1 + len(fcfg["null_angles_deg"])
    if P != cfg.n_beams:
        raise ValueError(f"feature P={P} beams but model.py Config.n_beams={cfg.n_beams}")
    if feat_shape[0] != 3 * P or feat_shape[1] != cfg.n_freqs:
        raise ValueError(f"unexpected feat shape {tuple(feat_shape)} (want [{3 * P}, {cfg.n_freqs}, T])")


def save_ckpt(path, model, opt, sched, epoch, best_val, args):
    torch.save({"model": model.state_dict(), "optim": opt.state_dict(), "sched": sched.state_dict(),
                "epoch": epoch, "best_val": best_val,
                "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}}, path)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Train BeamFusion from precomputed features")
    p.add_argument("--feat-root", type=Path, required=True, help="out-root of feature.py (has train/, test/, feature_config.json)")
    p.add_argument("--out-dir", type=Path, default=Path("./runs/beamfusion"))
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--min-lr", type=float, default=1e-4)
    p.add_argument("--lr-patience", type=int, default=4,
                   help="ReduceLROnPlateau patience; 4 => lr halved after 5 consecutive epochs without improvement")
    p.add_argument("--grad-clip", type=float, default=5.0, help="0 disables")
    p.add_argument("--val-count", type=int, default=200, help="samples held out from train/ for validation")
    p.add_argument("--segment-frames", type=int, default=0, help="train-time random crop in frames (0 = full clip)")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit-train", type=int, default=None)
    p.add_argument("--limit-val", type=int, default=None)
    p.add_argument("--limit-test", type=int, default=None)
    p.add_argument("--no-sisdr", action="store_true", help="skip SI-SDR during validation (faster)")
    p.add_argument("--resume", type=Path, default=None)
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--ckpt", type=Path, default=None, help="checkpoint for --eval-only (default: out-dir/best.pt)")
    p.add_argument("--log-every", type=int, default=50, help="iterations between progress prints")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cfg = Config()
    fcfg = load_feature_config(args.feat_root)
    P = cfg.n_beams

    train_ds, val_ds = build_train_val(args.feat_root, args.val_count, args.seed, args.segment_frames,
                                       args.limit_train, args.limit_val)
    test_ds = build_test(args.feat_root, args.limit_test)
    f0, t0 = train_ds[0] if len(train_ds) else val_ds[0]
    check_compat(cfg, fcfg, f0.shape)

    model = BeamFusionFromFeat(cfg).to(device)
    n_params = sum(p.numel() for p in model.net.parameters())
    print(f"device={device}  train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds) if test_ds else 0}")
    print(f"feat {tuple(f0.shape)}  target {tuple(t0.shape)}  hop={fcfg['hop_length']}  "
          f"network params={n_params / 1e3:.1f}K (paper ~85K; expect a gap, see model.py APPROX notes)")

    val_loader = make_loader(val_ds, args.batch_size, False, args.num_workers)
    test_loader = make_loader(test_ds, args.batch_size, False, args.num_workers) if test_ds else None

    # ------------------------------------------------------------------ eval only
    if args.eval_only:
        ck = torch.load(args.ckpt or (args.out_dir / "best.pt"), map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        for name, ld in (("val", val_loader), ("test", test_loader)):
            if ld is not None:
                print(f"[{name}] " + fmt_metrics(evaluate(model, ld, device, fcfg, P, not args.no_sisdr), P))
        return

    # ------------------------------------------------------------------ train
    train_loader = make_loader(train_ds, args.batch_size, True, args.num_workers, drop_last=len(train_ds) >= args.batch_size)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=args.lr_patience, min_lr=args.min_lr)

    start_epoch, best_val = 1, float("inf")
    if args.resume:
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optim"])
        sched.load_state_dict(ck["sched"])
        start_epoch, best_val = ck["epoch"] + 1, ck["best_val"]
        print(f"resumed from {args.resume} at epoch {start_epoch} (best val {best_val:.5f})")

    with open(args.out_dir / "args.json", "w") as f:
        json.dump({k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, f, indent=2)
    log_path = args.out_dir / "log.csv"
    new_log = not log_path.exists()
    log_f = open(log_path, "a", newline="")
    log = csv.writer(log_f)
    if new_log:
        log.writerow(["epoch", "train_loss", "val_loss", "val_sisdr_fused"] +
                     [f"val_sisdr_beam{i}" for i in range(P)] + ["lr", "seconds"])

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        t_start = time.time()
        run_loss, run_n = 0.0, 0
        for it, (feat, targ) in enumerate(train_loader, 1):
            feat, targ = feat.to(device, non_blocking=True), targ.to(device, non_blocking=True)
            fused, _ = model(feat)
            loss = tf_mse_loss(fused, targ)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at epoch {epoch} iter {it}; try --grad-clip, lower --lr, "
                                   f"or check the features with `python dataset.py`")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            run_loss += loss.item() * feat.shape[0]
            run_n += feat.shape[0]
            if it % args.log_every == 0:
                print(f"  epoch {epoch} it {it}/{len(train_loader)}  loss {run_loss / run_n:.5f}", flush=True)

        train_loss = run_loss / max(run_n, 1)
        m = evaluate(model, val_loader, device, fcfg, P, not args.no_sisdr)
        sched.step(m["loss"])
        lr = opt.param_groups[0]["lr"]
        secs = time.time() - t_start
        print(f"[epoch {epoch:3d}/{args.epochs}] train {train_loss:.5f} | val {fmt_metrics(m, P)} | "
              f"lr {lr:.2e} | {secs:.0f}s", flush=True)
        log.writerow([epoch, f"{train_loss:.6f}", f"{m['loss']:.6f}", f"{m.get('sisdr_fused', float('nan')):.3f}"] +
                     [f"{v:.3f}" for v in m.get("sisdr_beams", [float('nan')] * P)] + [f"{lr:.3e}", f"{secs:.0f}"])
        log_f.flush()

        if m["loss"] < best_val:
            best_val = m["loss"]
            save_ckpt(args.out_dir / "best.pt", model, opt, sched, epoch, best_val, args)
        save_ckpt(args.out_dir / "last.pt", model, opt, sched, epoch, best_val, args)
    log_f.close()

    # ------------------------------------------------------------------ final test
    print(f"\ntraining done. best val loss {best_val:.5f}")
    if test_loader is not None and (args.out_dir / "best.pt").exists():
        ck = torch.load(args.out_dir / "best.pt", map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        print(f"[test @ best epoch {ck['epoch']}] " + fmt_metrics(evaluate(model, test_loader, device, fcfg, P, not args.no_sisdr), P))


if __name__ == "__main__":
    main()
