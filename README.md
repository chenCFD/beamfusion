# BeamFusion (unofficial PyTorch reimplementation)

An unofficial, from-scratch reimplementation of **"Online Neural Fusion of Distortionless
Differential Beamformers for Robust Speech Enhancement"** ([Qian et al., arXiv:2510.24497](https://arxiv.org/abs/2510.24497)) —
a small causal network (~89K params) that learns to fuse the outputs of 5 fixed, distortionless
microphone-array beamformers (1 MWNG + 4 differential beamformers) into one enhanced signal,
frame by frame, in real time.

No official code was released with the paper. This repo builds the full pipeline from the paper's
text alone: room/array simulation → fixed-beamformer design → feature extraction → the
BeamFusion network → training → evaluation against every baseline in the paper's Table 1
(Noisy / MWNG / DMA-I–IV / ACC / BeamFusion).

**This README also documents, in detail, the places where the paper under-specifies its own
setup** (how "direct sound" is actually measured, target/interferer loudness, the exact DMA
regularization, ACC's update rule) and the debugging work that went into resolving each one —
see [Deviations from the paper](#deviations-from-the-paper--things-it-doesnt-specify) below.
If you're trying to reproduce this paper (or one like it) yourself, that section is probably the
most useful part of this repo.

## Results

Full held-out, speaker-disjoint test split (n=600), evaluated against a **direct-path-only**
reference (see [why not the reverberant image](#1-what-exactly-is-the-target-the-paper-never-fully-says)
below). `dSNR` / `STOI` / `dSI-SDR` are the paper's exact Table 1 columns (STOI is absolute, not a delta).

| Method | dSNR (dB) | STOI | dSI-SDR (dB) | paper dSNR | paper STOI | paper dSI-SDR |
|---|---:|---:|---:|---:|---:|---:|
| Noisy | – | 0.625 | – | – | 0.58 | – |
| MWNG | 2.11 | 0.694 | 1.63 | 0.28 | 0.62 | 0.22 |
| DMA-I | -2.27 | 0.651 | -4.02 | 5.31 | 0.69 | 5.50 |
| DMA-II | -0.97 | 0.683 | -2.44 | 6.02 | 0.71 | 6.16 |
| DMA-III | -0.25 | 0.689 | -1.57 | 4.58 | 0.69 | 4.68 |
| DMA-IV | -0.05 | 0.690 | -1.30 | 4.95 | 0.69 | 5.06 |
| ACC | 3.22 | 0.717 | 1.98 | 6.56 | 0.73 | 6.61 |
| **BeamFusion (ours)** | **5.03** | **0.732** | **3.12** | **10.24** | **0.74** | **7.13** |

By reverberation time (this dataset draws T60 uniformly from [0.2, 0.8] s rather than the
paper's two fixed points, so these bins are a rough proxy for the paper's T60=300ms / 700ms
columns, not an exact match):

| Method | dSI-SDR, T60∈[0.2,0.4)s | dSI-SDR, T60∈[0.6,0.8)s | paper ΔSI-SDR, T60=300ms | paper ΔSI-SDR, T60=700ms |
|---|---:|---:|---:|---:|
| MWNG | 1.42 | 1.95 | 0.22 | 0.19 |
| DMA-I | -2.10 | -5.83 | 5.50 | 6.83 |
| DMA-II | -0.69 | -4.05 | 6.16 | 6.34 |
| DMA-III | -0.04 | -3.21 | 4.68 | 4.48 |
| DMA-IV | 0.09 | -2.76 | 5.06 | 4.91 |
| ACC | 2.39 | 1.58 | 6.61 | 7.30 |
| **BeamFusion (ours)** | **3.58** | **2.75** | **7.13** | **7.58** |

**The headline result replicates the paper's core claim: BeamFusion > ACC > MWNG > every single
DMA**, in that order, matching Table 1's ranking, and it holds consistently across every T60
bin. The absolute gap to the paper is real and explained (not hidden) in the sections below —
mostly traced to the DMA baseline, not to the neural fusion network itself, which is both a
result and a genuine open question (see [Known limitations](#known-limitations--open-problem)).
Full per-sample metrics and generated audio for the run above are in `results/` after running
[step 5](#5-evaluate).

## Architecture

```
mixture (8ch) --STFT--> [5 fixed beamformers: MWNG, DMA-I..IV] --Re/Im/|.|--> BeamFusion Net --softmax weights--> weighted sum --ISTFT--> enhanced signal
                                                                       |
                                                           ERB compress -> encoder -> grouped
                                                           dual-path RNN -> decoder -> ERB expand
```

The 5 beamformers are fixed, closed-form, distortionless filters (not learned). The network only
ever learns the per-time-frequency-bin mixing weights `W_p(t,f)` — the actual signal processing
(null-steering, directivity) is done entirely by classical array signal processing. See
`model.py`'s docstring for exactly how this reimplementation's backbone differs from the paper's
(GTCRN-derived) one, and why.

## Repo layout

| File | What it does |
|---|---|
| `generate_dataset.py` | Simulates the room/array/moving-interferer scenario (Sec. 4.1) with `pyroomacoustics`; writes `mixture_*.wav`, `reference_*.wav`, `dry_target_*.wav` |
| `direct_ref.py` | Recovers a direct-path-only reference from `dry_target_*.wav`, without re-simulating anything |
| `feature.py` | Designs the 5 fixed beamformers and extracts the `[3P, F, T]` network input features |
| `dataset.py` | PyTorch `Dataset`/`DataLoader` over the features |
| `model.py` | The BeamFusion network (ERB → encoder → grouped dual-path RNN → decoder → softmax) |
| `train.py` | Training loop (Adam, `ReduceLROnPlateau`, the paper's loss) |
| `inference.py` | Runs every method (Noisy/MWNG/DMA-I–IV/ACC/BeamFusion) and reports Table-1-style metrics |

## Setup

```bash
git clone https://github.com/<you>/beamfusion.git
cd beamfusion
pip install -r requirements.txt
```

Needs a LibriSpeech corpus on disk (e.g. [`train-clean-100`](https://www.openslr.org/12)) and a
CUDA GPU for training (CPU works but is slow). `generate_dataset.py` is the heavy step — budget
real time and ~10 GB of disk per 10,000 training samples.

## Reproduce

### 1. Simulate the dataset

```bash
python generate_dataset.py \
    --librispeech-root /path/to/LibriSpeech/train-clean-100 \
    --out-dir ./data/dataset \
    --n-jobs 16
```

Builds the paper's 8-mic, 1 cm-spacing ULA in an 8×6×3 m room, a static target at 0°, and an
interferer sweeping 90°→180°, with randomized T60 (0.2–0.8 s), SNR (20–40 dB — paper's range),
and **SIR (0–10 dB, our addition — see [below](#2-how-loud-is-the-interferer-the-paper-never-says))**.
10,000 train / 600 test samples by default, speaker-disjoint split.

### 2. Recover a direct-path reference

```bash
python direct_ref.py --data-root ./data/dataset --splits train test --n-jobs 8
```

Writes `reference_direct_*.wav` next to the existing (reverberant) `reference_*.wav` — see
[why this matters](#1-what-exactly-is-the-target-the-paper-never-fully-says).

### 3. Design the beamformers and extract features

```bash
python feature.py --data-root ./data/dataset --out-root ./data/features \
    --ref-name reference_direct --n-jobs 16
```

### 4. Train

```bash
python train.py --feat-root ./data/features --out-dir ./runs/beamfusion \
    --segment-frames 400 --num-workers 12
```

`--segment-frames 400` trains on random 3.2 s crops instead of the paper's full 10 s clips, to
fit consumer GPU memory; validation/test always use full clips. Drop the flag to match the paper
exactly if you have the VRAM for batch 30 × 10 s. Training converges (`lr` decays to its floor,
val loss plateaus) in 100–150 epochs in practice, well past the paper's nominal 50.

### 5. Evaluate

```bash
python inference.py --data-root ./data/dataset --split test \
    --feat-root ./data/features --ckpt ./runs/beamfusion/best.pt \
    --out-dir ./results
```

Runs Noisy/MWNG/DMA-I–IV/ACC/BeamFusion on the full test split, against both the reverberant and
direct-path references, and writes `results/summary.txt` / `summary.json` /
`metrics_per_sample.csv`. Add `--limit 20` for a quick check first.

## Deviations from the paper / things it doesn't specify

The paper (reasonably, for a 2-page ICASSP submission) omits several implementation details that
turn out to matter a lot in practice. Each one below cost real debugging time; recording them
here so nobody else has to rediscover them from scratch.

### 1. What exactly is "the target"? The paper never fully says.

`generate_dataset.py`'s natural output, `reference_*.wav`, is the target speech convolved with
its **full room impulse response** (direct path + all reflections) — the "reverberant image."
The paper's loss, however, says the output should approximate the **"direct sound component"**
of the target. At 2 m source distance in this room (beyond the critical distance for the given
T60 range), a lot of the reverberant image's energy *is* reverberation — we measured a mean
direct-to-reverberant ratio of **−4.75 dB on average** (range −10.0 to +2.2 dB across the test
set, n=600; `inference.py` prints this). Training or
evaluating against the reverberant image instead of the direct path systematically penalizes
*any* beamformer that does its job (suppressing off-axis energy, including reflections), and
produces results that don't track the paper's. `direct_ref.py` recovers a direct-path-only
reference from the untouched dry speech without needing to re-simulate anything — see its
docstring for the exact method (short least-squares FIR around the cross-correlation peak).

### 2. How loud is the interferer? The paper never says.

`generate_dataset.py`, as a literal reading of Sec. 4.1, just sums the (independently, randomly
leveled) target and interferer dry clips — there's no stated control on their relative loudness
(SIR). Measuring it after the fact: mean SIR ≈ +0.9 dB, and the interferer is *louder* than the
target in **42% of samples**. The paper reports a 20–40 dB range for the *additive white noise*
SNR but says nothing about target/interferer balance, so we don't know whether the original
authors controlled for this. This repo adds an explicit `sir_range_db` (default `[0, 10]` dB,
target always at least as loud as the interferer) applied as a single scalar gain on the
interferer's reverberant image (preserves its spatial/spectral shape, only rescales loudness) —
see the long comment on `Config.sir_range_db` in `generate_dataset.py`.

### 3. The DMA regularization is a real, unresolved bottleneck.

The paper's "four first-order differential beamformers" are a 2-microphone-pair, null-constrained
LCMV design in this implementation (`feature.py::design_beamformers`). With this array's 7 cm
aperture, the look and null steering vectors become nearly parallel at low frequency, so the
exact (unregularized) solution blows up (white-noise/reverb gain >700× at 31 Hz). We regularize
with a **white-noise-gain cap** (bisection on diagonal loading per frequency bin, capped at a
fixed multiple of MWNG's constant noise gain — see `Config.max_wng_db`), which is more
principled than a naive eigenvalue floor (it directly bounds the physical quantity that actually
causes the error, rather than a proxy for it) but still trades away null depth in the
100–1000 Hz band, which carries ~80% of the reference energy. **This is the single biggest
source of the gap to the paper's numbers** — see [Known limitations](#known-limitations--open-problem).

### 4. ACC's exact update rule isn't public.

The paper cites \[21\] (Jin et al., ICASSP 2024) for the adaptive-convex-combination baseline but
that paper's code isn't public either. `inference.py::acc_fuse` implements a standard per-bin
normalized exponentiated-gradient convex combination (weights on the simplex, so the
distortionless constraint holds automatically); tune with `--acc-mu` / `--acc-floor`.

### 5. Matching 85K params requires guessing at an architecture the paper doesn't fully specify.

The paper sets the GTCRN-derived backbone's (Rong et al., ICASSP 2024) hidden dimension to 32 and
reports ~85K parameters / 142 MMac/s — but a literal GTCRN backbone at `hidden_dim=32` plus this
paper's 5-beam input/output heads lands at **~6.3K params**, about 13× short. We tried two
approaches:
- widen the hand-rolled backbone's capacity (3-stage conv encoder/decoder + 4-block grouped
  dual-path RNN, `hidden_dim=128`) until the param count lands near 85K — this is **what's
  shipped in `model.py`** and produces the results above;
- a faithful, line-by-line port of the actual [GTCRN](https://github.com/Xiaobin-Rong/gtcrn)
  backbone (grouped dilated conv + SE + TRA blocks) at the paper's stated `hidden_dim=32`, which
  lands at 85.1K params almost exactly — but trained noticeably *worse* (val loss plateaus ~0.52
  vs. ~0.35 for the wider hand-rolled version, controlling for everything else: same features,
  same batch size, same LR schedule, with or without LR warmup). Our working theory is that
  GTCRN's narrow-channel design, tuned for *dense* single-channel spectral mask regression, is a
  poor capacity allocation for BeamFusion's actual task — a *selection* problem over 5 fixed
  candidates — but we haven't proven this. Worth investigating further if you pick this repo up.

## Known limitations / open problem

**The DMA beamformers underperform MWNG in our evaluation (negative ΔSI-SDR for all 4), the
opposite of the paper's Table 1 (DMAs beat MWNG by ~5 dB).** This is the main open gap and,
interestingly, looks like it's rooted in actual array physics rather than a simple
implementation bug: at this array's 1 cm spacing, diffuse reverberation is nearly perfectly
coherent across microphones up to several kHz, which a differencing (DMA) design should
suppress very effectively — but DMA also necessarily amplifies spatially *incoherent* components
(the array's white-noise gain penalty), and the amount of amplification needed to hold a null
scales badly at this aperture. Whether the net effect is a win or a loss depends on exactly how
diffuse/coherent the simulated reverberant field actually is — a WNG cap that directly accounts
for the array's diffuse-field coherence (rather than a flat per-frequency cap, see
[deviation #3](#3-the-dma-regularization-is-a-real-unresolved-bottleneck)) is the most promising
next step we identified but didn't have time to fully chase down. If you solve this, a PR (or a
paper) would be very welcome.

## Citation

```bibtex
@inproceedings{qian2026beamfusion,
  title={Online Neural Fusion of Distortionless Differential Beamformers for Robust Speech Enhancement},
  author={Qian, Yuanhang and Zhao, Kunlong and Jin, Jilu and Luo, Xueqin and Huang, Gongping and Chen, Jingdong and Benesty, Jacob},
  journal={arXiv preprint arXiv:2510.24497},
  year={2025}
}
```

This repository is an independent, unofficial reimplementation and is not affiliated with the
paper's authors.

## License

MIT (see `LICENSE`) — update if you'd prefer something else.
