"""
dataset.py — PyTorch Dataset / DataLoader helpers for BeamFusion training.

Reads the files written by feature.py:

    <feat_root>/<split>/feat_XXXXXX.npy     [3P, F, T]   (Re_0..Re_{P-1}, Im_0.., Mag_0..)
    <feat_root>/<split>/target_XXXXXX.npy   [2,  F, T]   (Re, Im of the reference STFT)
    <feat_root>/feature_config.json                      (n_fft / hop / ... used by feature.py)

Each item is (feat, target), both float32 (the files are usually float16 on disk).

Notes
-----
* .npy files are opened with mmap, so a random time-crop only reads the bytes it needs.
* Paper setup: full 10 s clips, batch size 30.  `segment_frames=0` keeps the whole clip.
  If GPU memory is tight, set e.g. segment_frames=250 (2 s at hop 128) for training only
  (validation / test always use the full clip).
* A validation set is held out from the *train* split by sample index (the paper does not
  describe its validation set).  Speakers are shared with train, so treat the val loss as a
  convergence monitor and the `test` split (speaker-disjoint, made by generate_dataset.py) as
  the honest number.

Smoke test:
    python dataset.py --feat-root ./dataset/features
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def load_feature_config(feat_root: Path) -> dict:
    p = Path(feat_root) / "feature_config.json"
    if not p.exists():
        raise FileNotFoundError(f"{p} not found — was feature.py run with --out-root {feat_root}?")
    with open(p) as f:
        return json.load(f)


def list_pairs(split_dir: Path):
    """Sorted [(feat_path, target_path)] for every feat_XXXXXX.npy that has a target_XXXXXX.npy."""
    pairs = []
    for fp in sorted(Path(split_dir).glob("feat_*.npy")):
        m = re.match(r"feat_(\d+)\.npy$", fp.name)
        if not m:
            continue
        tp = fp.with_name(f"target_{m.group(1)}.npy")
        if tp.exists():
            pairs.append((fp, tp))
    return pairs


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class BeamFusionDataset(Dataset):
    def __init__(self, pairs, segment_frames: int = 0, random_crop: bool = True):
        self.pairs = list(pairs)
        self.segment_frames = int(segment_frames)
        self.random_crop = random_crop

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        fp, tp = self.pairs[i]
        feat = np.load(fp, mmap_mode="r")        # [3P, F, T]
        targ = np.load(tp, mmap_mode="r")        # [2,  F, T]
        T = feat.shape[-1]
        if targ.shape[-1] != T or feat.shape[1] != targ.shape[1]:
            raise ValueError(f"shape mismatch: {fp.name} {feat.shape} vs {tp.name} {targ.shape}")

        start, stop = 0, T
        seg = self.segment_frames
        if seg and seg < T:
            if self.random_crop:
                start = int(torch.randint(0, T - seg + 1, (1,)).item())   # per-worker seeded by DataLoader
            stop = start + seg

        feat = np.ascontiguousarray(feat[..., start:stop], dtype=np.float32)
        targ = np.ascontiguousarray(targ[..., start:stop], dtype=np.float32)
        return torch.from_numpy(feat), torch.from_numpy(targ)


def collate_min_len(batch):
    """Stack to [B, C, F, T]; if lengths differ, crop everything to the shortest clip."""
    T = min(f.shape[-1] for f, _ in batch)
    feats = torch.stack([f[..., :T] for f, _ in batch], dim=0)
    targs = torch.stack([t[..., :T] for _, t in batch], dim=0)
    return feats, targs


# ---------------------------------------------------------------------------
# builders
# ---------------------------------------------------------------------------
def build_train_val(feat_root, val_count=200, seed=0, segment_frames=0,
                    limit_train=None, limit_val=None):
    pairs = list_pairs(Path(feat_root) / "train")
    if not pairs:
        raise FileNotFoundError(f"no feat_*.npy/target_*.npy pairs under {Path(feat_root) / 'train'}")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(pairs))
    n_val = min(val_count, max(len(pairs) - 1, 0))
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    train_pairs = [pairs[i] for i in sorted(train_idx)]
    val_pairs = [pairs[i] for i in sorted(val_idx)]
    if limit_train:
        train_pairs = train_pairs[:limit_train]
    if limit_val:
        val_pairs = val_pairs[:limit_val]
    train_ds = BeamFusionDataset(train_pairs, segment_frames=segment_frames, random_crop=True)
    val_ds = BeamFusionDataset(val_pairs, segment_frames=0, random_crop=False)   # full clips
    return train_ds, val_ds


def build_test(feat_root, limit=None):
    pairs = list_pairs(Path(feat_root) / "test")
    if limit:
        pairs = pairs[:limit]
    return BeamFusionDataset(pairs, segment_frames=0, random_crop=False) if pairs else None


def make_loader(ds, batch_size, shuffle, num_workers=4, drop_last=False):
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
        drop_last=drop_last, collate_fn=collate_min_len,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


# ---------------------------------------------------------------------------
# smoke test
# ---------------------------------------------------------------------------
def _main():
    p = argparse.ArgumentParser()
    p.add_argument("--feat-root", type=Path, required=True)
    p.add_argument("--val-count", type=int, default=200)
    p.add_argument("--segment-frames", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=4)
    args = p.parse_args()

    fcfg = load_feature_config(args.feat_root)
    print("feature_config:", {k: fcfg[k] for k in ("sample_rate", "n_fft", "hop_length", "dtype")})
    train_ds, val_ds = build_train_val(args.feat_root, args.val_count, segment_frames=args.segment_frames)
    test_ds = build_test(args.feat_root)
    print(f"train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds) if test_ds else 0}")

    loader = make_loader(train_ds, args.batch_size, shuffle=True, num_workers=0)
    feat, targ = next(iter(loader))
    print(f"feat   {tuple(feat.shape)} {feat.dtype}  min={feat.min():.3g} max={feat.max():.3g} "
          f"rms={feat.pow(2).mean().sqrt():.3g}")
    print(f"target {tuple(targ.shape)} {targ.dtype}  min={targ.min():.3g} max={targ.max():.3g} "
          f"rms={targ.pow(2).mean().sqrt():.3g}")
    P = feat.shape[1] // 3
    mag = feat[:, 2 * P:]
    err = (mag - torch.sqrt(feat[:, :P] ** 2 + feat[:, P:2 * P] ** 2)).abs().max()
    print(f"channels: 3P={feat.shape[1]} (P={P}); max |Mag - sqrt(Re^2+Im^2)| = {err:.3g} "
          f"(float16 rounding only)")
    assert torch.isfinite(feat).all() and torch.isfinite(targ).all()


if __name__ == "__main__":
    _main()
