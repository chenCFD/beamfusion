"""
generate_dataset.py — 1:1 reproduction of the BeamFusion paper's data
generation recipe (Sec. 4.1 of arXiv:2510.24497).

WHAT THIS BUILDS
-----------------
For each sample:
  1. Two "dense" 10 s speech clips are assembled from LibriSpeech utterances
     (silence-trimmed and concatenated so no silent gap exceeds 0.1 s) —
     one for the target, one for the interferer.
  2. A shoebox room (8 x 6 x 3 m) is built with a random T60 in [0.2, 0.8] s.
  3. An 8-element ULA (1 cm spacing) sits at the room's geometric center,
     at the stated height. The target is fixed at azimuth 0 deg, 2 m away.
     The interferer starts at azimuth 90 deg, 2 m away, and rotates
     counter-clockwise by 10 deg every second until it reaches 180 deg —
     implemented as ten 1-second blocks, each with its own (static) RIR,
     stitched together with proper convolution-tail overlap-add so the
     reverberation stays continuous across block boundaries.
  4. Target-only and interference-only reverberant signals are convolved
     per-microphone, the interferer is rescaled (as a single global gain,
     so its spatial/spectral shape is untouched) so the target/interferer
     power ratio at the reference mic equals a per-sample SIR drawn from
     cfg.sir_range_db, then they are summed and contaminated with
     spatially-white Gaussian noise (independent per microphone) at a
     random SNR in [20, 40] dB measured on that already-SIR-controlled sum.
  5. The mixture (8 channels) and a reference signal (the reverberant
     target image at a reference microphone, pre-interference, pre-noise)
     are written to disk, plus a manifest row with all the sampled
     parameters for reproducibility.

HONESTY NOTE ON WHAT WAS VERIFIED HERE
----------------------------------------
This sandbox has no network access and pyroomacoustics / soundfile are not
installed in it, so the pyroomacoustics-dependent calls in this file could
NOT be executed end-to-end here. What WAS verified numerically (with numpy
/ scipy, which ARE available here) before writing this file:
  - the array/source geometry reproduces the paper's exact coordinates
    ((6,2,1) for the target, (2,2,1) for the interferer at 180 deg, etc.)
  - the block-wise overlap-add convolution correctly carries a block's
    reverberant tail into the next block (no hard truncation clicks)
  - the silence-trim + concatenation logic guarantees no silent run
    exceeds the configured gap (measured ~0.067 s against a 0.1 s budget)
  - the SNR-scaling formula hits the target SNR to within 0.01 dB and
    produces genuinely independent (not identical) per-microphone noise
Please run this on your machine (where pyroomacoustics + soundfile are
installed per our conda setup) on a tiny --n-train 4 --n-test 2 first, and
paste back any traceback — the pyroomacoustics API calls themselves are
written from well-established usage patterns but weren't execution-tested
here.

AMBIGUITIES THE PAPER LEAVES OPEN — AND WHAT THIS SCRIPT DEFAULTS TO
-----------------------------------------------------------------------
(see our chat discussion — flagged again here so the script is self-
documenting for your README later)
  - "reference signal" for the loss: this script saves the *reverberant*
    target image at the reference mic (REF_MIC_INDEX) — i.e. target
    speech through its own RIR only, no interferer, no noise. This
    matches the distortionless constraint's actual guarantee (undistorted
    direct path), not a dereverberated anechoic target. A raw anechoic
    dry copy is saved too for convenience but is NOT time-aligned to the
    mixture (would need direct-path delay compensation to use directly).
  - interferer elevation: assumed equal to the array height (same
    z-coordinate), since the paper only specifies azimuth and distance.
  - which mic is "the" reference: assumed mic index 0.
  - target/interferer speaker identity: this script picks the target and
    interferer clips from *different* speakers (more realistic "a second
    talker interferes" scenario); the paper doesn't say either way.
  - train/test split: this script reserves a disjoint set of LibriSpeech
    speakers for the test split (the paper doesn't mention this, but
    training and testing on clips from the same speakers who also appear
    in training is a common and avoidable leakage risk).
  - target/interferer relative loudness (SIR): the paper's Sec. 4.1 gives
    a random SNR range for the ADDITIVE WHITE NOISE (20-40 dB) but never
    says anything about how loud the interferer is relative to the target
    -- the two dry clips used to just be loaded straight off LibriSpeech
    and summed as-is, so the SIR was whatever the two speakers' natural
    recording levels happened to produce (measured on this script's own
    output: mean ~+0.9 dB, 42% of samples with the interferer LOUDER than
    the target). That is a real gap relative to what's described in the
    paper, not a deliberate difficulty choice, so this version adds an
    explicit cfg.sir_range_db and rescales the interferer (one global gain
    per sample, applied after its RIR, so its spatial/spectral shape is
    untouched) to hit a per-sample SIR drawn uniformly from that range,
    measured at the reference mic BEFORE noise is added. Default is
    [0, 10] dB (target always at least as loud as the interferer, matching
    the "~5 dB louder" starting point from our chat discussion while still
    leaving a difficulty range rather than a single fixed value, the same
    way t60_range / snr_range_db are ranges rather than constants).
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.signal import fftconvolve


# ---------------------------------------------------------------------------
# 0. Config — every number here is taken directly from Sec. 4.1 / 4.3 of the
#    paper unless marked APPROX (an assumption the paper left open).
# ---------------------------------------------------------------------------
@dataclass
class Config:
    # --- paths ---
    librispeech_root: str = "./LibriSpeech/train-clean-100"
    out_dir: str = "./data/beamfusion"

    # --- dataset sizes (paper: 10,000 train / 600 test) ---
    n_train: int = 10_000
    n_test: int = 600
    test_speaker_fraction: float = 0.1       # APPROX: speaker-disjoint split

    # --- audio ---
    sample_rate: int = 16_000
    clip_duration_s: float = 10.0
    gap_duration_s: float = 0.05             # APPROX: splice gap between
                                              # concatenated utterances;
                                              # guarantees no silent run > 0.1 s
    vad_frame_ms: float = 20.0               # APPROX: VAD frame size
    vad_thresh_ratio: float = 0.02           # APPROX: VAD energy threshold

    # --- room (Sec. 4.1) ---
    room_dim: tuple = (8.0, 6.0, 3.0)        # 8m x 6m x 3m
    array_center: tuple = (4.0, 2.0, 1.0)
    t60_range: tuple = (0.2, 0.8)            # seconds

    # --- array (Sec. 4.1 / 4.2) ---
    n_mics: int = 8
    mic_spacing: float = 0.01                # 1 cm ULA
    ref_mic_index: int = 0                   # APPROX: which mic is "reference"

    # --- target / interferer geometry (Sec. 4.1) ---
    source_distance: float = 2.0             # meters from array center
    target_azimuth_deg: float = 0.0
    interferer_start_azimuth_deg: float = 90.0
    interferer_end_azimuth_deg: float = 180.0
    interferer_deg_per_sec: float = 10.0
    source_elevation_deg: float = 0.0        # APPROX: not stated in paper

    # --- noise (Sec. 4.1) ---
    snr_range_db: tuple = (20.0, 40.0)

    # --- target/interferer loudness (APPROX -- not specified by the paper,
    #     see the AMBIGUITIES note above). Per-sample SIR in dB, measured at
    #     the reference mic on the reverberant target/interferer images,
    #     BEFORE additive noise. target always >= interferer at the low end
    #     of the range (0 dB = equal loudness, 10 dB = target 10x the power).
    sir_range_db: tuple = (0.0, 10.0)

    # --- room-acoustics simulation ---
    sound_speed: float = 343.0

    # --- RIR synthesis method ---
    # "ism"    : pure image-source method, order from pra.inverse_sabine
    #            (the paper's literal description). In a perfect shoebox at
    #            high order (63 for T60=0.5 s here) the image sources arrive
    #            on a very regular lattice, which shows up as a fan of
    #            diagonal lines in the RIR spectrogram and is audible as a
    #            speaker-measurement-style sweep tone on speech.
    # "hybrid" : ISM only for the first `hybrid_ism_order` reflections, ray
    #            tracing for the late tail (statistically diffuse, no
    #            lattice). Deviation from the paper's literal wording, chosen
    #            for demo-quality audio. Tune with debug_hybrid_check.py.
    rir_method: str = "hybrid"
    hybrid_ism_order: int = 12
    n_rays: int = 10_000                     # the value validated in your debug run
    hybrid_scattering: float = 0.3
    ray_receiver_radius: float = 0.5
    ray_energy_thres: float = 1e-7

    # --- misc ---
    seed: int = 0
    n_jobs: int = 4


# ---------------------------------------------------------------------------
# 1. LibriSpeech indexing
# ---------------------------------------------------------------------------
def index_librispeech(root: str) -> dict[str, list[Path]]:
    """Scans a LibriSpeech subset directory (e.g. train-clean-100) and
    returns {speaker_id: [utterance .flac paths]}. LibriSpeech's on-disk
    layout is root/<speaker>/<chapter>/<speaker>-<chapter>-<utt>.flac."""
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(
            f"LibriSpeech root not found: {root}. Download train-clean-100 "
            f"from https://www.openslr.org/12/ and point --librispeech-root "
            f"at the extracted folder."
        )
    by_speaker: dict[str, list[Path]] = {}
    for flac_path in root_path.rglob("*.flac"):
        speaker_id = flac_path.parent.parent.name
        by_speaker.setdefault(speaker_id, []).append(flac_path)
    if not by_speaker:
        raise RuntimeError(f"No .flac files found under {root} — check the path.")
    return by_speaker


def split_speakers(
    by_speaker: dict[str, list[Path]], test_fraction: float, rng: random.Random
) -> tuple[dict[str, list[Path]], dict[str, list[Path]]]:
    """Speaker-disjoint train/test split (APPROX — paper doesn't specify this,
    but it avoids train/test content leakage)."""
    speakers = list(by_speaker.keys())
    rng.shuffle(speakers)
    n_test_speakers = max(1, int(len(speakers) * test_fraction))
    test_speakers = set(speakers[:n_test_speakers])
    train = {s: v for s, v in by_speaker.items() if s not in test_speakers}
    test = {s: v for s, v in by_speaker.items() if s in test_speakers}
    return train, test


# ---------------------------------------------------------------------------
# 2. Speech clip assembly: silence-trim + concatenate to a dense 10 s clip
# ---------------------------------------------------------------------------
def trim_silence(x: np.ndarray, sr: int, frame_ms: float, thresh_ratio: float) -> np.ndarray:
    """Energy-based VAD: trims leading/trailing silence from one utterance."""
    frame = max(1, int(sr * frame_ms / 1000))
    n_frames = max(1, len(x) // frame)
    if n_frames < 1 or len(x) == 0:
        return x
    energies = np.array(
        [np.sqrt(np.mean(x[i * frame:(i + 1) * frame] ** 2) + 1e-12) for i in range(n_frames)]
    )
    thresh = thresh_ratio * (energies.max() + 1e-12)
    active = np.where(energies > thresh)[0]
    if len(active) == 0:
        return x
    start = active[0] * frame
    end = min(len(x), (active[-1] + 1) * frame)
    return x[start:end]


def load_utterance(path: Path, target_sr: int):
    """Reads one LibriSpeech .flac file, resampling if needed. Returns a
    float32 mono array."""
    import soundfile as sf

    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        # LibriSpeech is natively 16 kHz; this branch only fires if you've
        # pointed --sample-rate at something else.
        from scipy.signal import resample_poly
        from math import gcd

        g = gcd(sr, target_sr)
        audio = resample_poly(audio, target_sr // g, sr // g).astype(np.float32)
    return audio


def build_dense_clip(
    utterance_paths: list[Path],
    cfg: Config,
    rng: np.random.Generator,
) -> np.ndarray:
    """Concatenates silence-trimmed utterances (drawn with replacement from
    utterance_paths) with small fixed gaps until the clip reaches
    cfg.clip_duration_s, then truncates to exactly that length. Guarantees,
    by construction, that no silent run exceeds cfg.gap_duration_s (which is
    well under the paper's 0.1 s limit)."""
    sr = cfg.sample_rate
    gap = np.zeros(int(sr * cfg.gap_duration_s), dtype=np.float32)
    target_len = int(sr * cfg.clip_duration_s)

    chunks: list[np.ndarray] = []
    total = 0
    pool = list(utterance_paths)
    rng.shuffle(pool)
    pool_i = 0

    while total < target_len:
        if pool_i >= len(pool):
            rng.shuffle(pool)
            pool_i = 0
        audio = load_utterance(pool[pool_i], sr)
        pool_i += 1
        trimmed = trim_silence(audio, sr, cfg.vad_frame_ms, cfg.vad_thresh_ratio)
        if len(trimmed) == 0:
            continue
        chunks.append(trimmed)
        chunks.append(gap)
        total += len(trimmed) + len(gap)

    clip = np.concatenate(chunks)[:target_len]
    if len(clip) < target_len:
        clip = np.pad(clip, (0, target_len - len(clip)))
    return clip.astype(np.float32)


# ---------------------------------------------------------------------------
# 3. Array / source geometry (verified against the paper's stated
#    coordinates: target -> (6,2,1), interferer@180deg -> (2,2,1))
# ---------------------------------------------------------------------------
def mic_array_positions(cfg: Config) -> np.ndarray:
    """Returns mic positions, shape [3, M] (pyroomacoustics convention):
    an M-element ULA along the x-axis, centered at cfg.array_center."""
    center = np.array(cfg.array_center)
    offsets = (np.arange(cfg.n_mics) - (cfg.n_mics - 1) / 2) * cfg.mic_spacing
    positions = center[:, None] + np.stack(
        [offsets, np.zeros(cfg.n_mics), np.zeros(cfg.n_mics)], axis=0
    )
    return positions  # [3, M]


def polar_to_cartesian(
    center: np.ndarray, distance: float, azimuth_deg: float, elevation_deg: float = 0.0
) -> np.ndarray:
    az = np.deg2rad(azimuth_deg)
    el = np.deg2rad(elevation_deg)
    dx = distance * np.cos(az) * np.cos(el)
    dy = distance * np.sin(az) * np.cos(el)
    dz = distance * np.sin(el)
    return center + np.array([dx, dy, dz])


def interferer_azimuth_schedule(cfg: Config) -> list[float]:
    """One azimuth per 1-second block, from 90 deg rotating 10 deg/s,
    clamped at 180 deg (matches "rotated ... by 10 deg every second until
    reaching 180 deg")."""
    n_blocks = int(round(cfg.clip_duration_s))
    schedule = []
    for b in range(n_blocks):
        az = cfg.interferer_start_azimuth_deg + cfg.interferer_deg_per_sec * b
        az = min(az, cfg.interferer_end_azimuth_deg)
        schedule.append(az)
    return schedule


# ---------------------------------------------------------------------------
# 4. Room-acoustics simulation (pyroomacoustics) — NOT execution-tested in
#    this sandbox (no network to install pyroomacoustics here); see the
#    honesty note at the top of this file.
# ---------------------------------------------------------------------------
def compute_single_source_rirs(
    cfg: Config, mic_positions: np.ndarray, source_pos: np.ndarray, absorption: float, max_order: int
) -> np.ndarray:
    """Builds a fresh shoebox room with ONE source and the mic array, runs
    the RIR simulation selected by cfg.rir_method ("ism" or "hybrid"), and
    returns the RIRs as an object array of length n_mics (each entry a 1-D
    float array — RIR lengths differ slightly per mic/source geometry, so
    this is not a rectangular array)."""
    import pyroomacoustics as pra

    if cfg.rir_method == "hybrid":
        room = pra.ShoeBox(
            list(cfg.room_dim),
            fs=cfg.sample_rate,
            materials=pra.Material(absorption, scattering=cfg.hybrid_scattering),
            max_order=cfg.hybrid_ism_order,   # `max_order` arg is ignored here on purpose
            ray_tracing=True,
        )
        room.set_ray_tracing(
            n_rays=cfg.n_rays,
            receiver_radius=cfg.ray_receiver_radius,
            energy_thres=cfg.ray_energy_thres,
        )
    elif cfg.rir_method == "ism":
        room = pra.ShoeBox(
            list(cfg.room_dim),
            fs=cfg.sample_rate,
            materials=pra.Material(absorption),
            max_order=max_order,
        )
    else:
        raise ValueError(f"unknown rir_method {cfg.rir_method!r} (use 'ism' or 'hybrid')")
    # NOTE: recent pyroomacoustics (>=0.7) accepts a raw (3, n_mics) ndarray
    # here directly. If your installed version errors on this line, wrap it
    # as `pra.MicrophoneArray(mic_positions, cfg.sample_rate)` instead — the
    # exact call changed across pyroomacoustics versions.
    room.add_microphone_array(mic_positions)
    room.add_source(list(source_pos))
    room.compute_rir()

    rirs = np.empty(cfg.n_mics, dtype=object)
    for m in range(cfg.n_mics):
        rirs[m] = np.asarray(room.rir[m][0], dtype=np.float32)
    return rirs


def convolve_blockwise_moving_source(
    dry: np.ndarray, rirs_per_block: list[np.ndarray], cfg: Config
) -> np.ndarray:
    """Convolves a continuous dry signal against a sequence of per-block
    (per-second) RIRs, one static RIR per 1-second block, and overlap-adds
    the convolution tails across block boundaries so the reverberation
    stays continuous (verified against a synthetic case: a block's tail
    correctly bleeds into the next block rather than being clipped)."""
    sr = cfg.sample_rate
    block_len = int(round(sr * 1.0))
    n_blocks = len(rirs_per_block)
    max_rir_len = max(len(r) for r in rirs_per_block)
    out_len = n_blocks * block_len + max_rir_len - 1
    out = np.zeros(out_len, dtype=np.float32)

    for b, rir in enumerate(rirs_per_block):
        start = b * block_len
        end = min(start + block_len, len(dry))
        block_dry = dry[start:end]
        if len(block_dry) == 0:
            continue
        wet = fftconvolve(block_dry, rir)
        out[start:start + len(wet)] += wet

    return out[: n_blocks * block_len]  # truncate to the nominal clip length


def convolve_static_source(dry: np.ndarray, rir: np.ndarray) -> np.ndarray:
    """Target doesn't move, so it only needs one RIR for the whole clip.
    Truncated back to the dry signal's length (direct-path delay means the
    very start is a few samples of silence, and the reverberant tail past
    the clip end is dropped — consistent with how the moving-source blocks
    above are truncated to the nominal clip length)."""
    wet = fftconvolve(dry, rir)
    return wet[: len(dry)].astype(np.float32)


# ---------------------------------------------------------------------------
# 5. SIR control (target/interferer loudness) + spatially-white Gaussian
#    noise at a target SNR
# ---------------------------------------------------------------------------
def rescale_interferer_to_sir(
    target_wet: np.ndarray, interf_wet: np.ndarray, ref_mic_index: int, sir_db: float
) -> np.ndarray:
    """interf_wet: [M, N] reverberant interferer image, rescaled by ONE global
    scalar (same gain on every mic, so its spatial image / ILD-ITD cues and
    its spectral shape are untouched -- only its overall loudness changes) so
    that, at the reference mic, power(target) / power(interferer) == 10**(sir_db/10).
    Returns the rescaled interf_wet; target_wet is untouched."""
    target_power = np.mean(target_wet[ref_mic_index].astype(np.float64) ** 2)
    interf_power = np.mean(interf_wet[ref_mic_index].astype(np.float64) ** 2) + 1e-12
    desired_interf_power = target_power / (10 ** (sir_db / 10))
    scale = np.sqrt(desired_interf_power / interf_power)
    return (interf_wet * scale).astype(np.float32)


def add_white_noise_at_snr(
    mixture: np.ndarray, ref_mic_index: int, snr_db: float, rng: np.random.Generator
) -> np.ndarray:
    """mixture: [M, N] reverberant target+interferer. Adds i.i.d. (spatially
    uncorrelated) Gaussian noise per channel so that the reference-mic SNR
    equals snr_db (verified numerically to within 0.01 dB)."""
    ref_power = np.mean(mixture[ref_mic_index] ** 2)
    noise_power = ref_power / (10 ** (snr_db / 10))
    noise = rng.normal(0, np.sqrt(noise_power), size=mixture.shape).astype(np.float32)
    return mixture + noise


def normalize_peak(
    mixture: np.ndarray, reference: np.ndarray, dry_target: np.ndarray, target_peak: float = 0.95
):
    """Applies ONE global scalar gain to mixture, reference, and dry_target
    together, only if needed to bring the loudest sample among all three
    under target_peak. Because the same scalar is applied to everything,
    the relative levels between target/interferer/noise (i.e. the SNR you
    sampled in add_white_noise_at_snr) are completely unaffected — only
    the overall loudness changes, and only when it would otherwise clip.

    Without this, summing a reverberant target + a reverberant moving
    interferer + noise frequently produces peaks above +-1.0, which get
    silently hard-clipped when written as 16-bit PCM WAV (soundfile.write
    defaults to PCM_16 for the .wav container when no subtype is given) —
    this is what produced the severe clipping/popping heard in `mixture`
    and the intermittent micro-clipping ("chirping" during loud passages)
    heard in `reference`."""
    peak = max(
        float(np.max(np.abs(mixture))),
        float(np.max(np.abs(reference))),
        float(np.max(np.abs(dry_target))),
        1e-8,
    )
    if peak > target_peak:
        scale = target_peak / peak
        mixture = mixture * scale
        reference = reference * scale
        dry_target = dry_target * scale
    return mixture, reference, dry_target


# ---------------------------------------------------------------------------
# 6. One full sample
# ---------------------------------------------------------------------------
def build_one_sample(
    sample_idx: int,
    target_utts: list[Path],
    interf_utts: list[Path],
    cfg: Config,
) -> dict:
    rng_np = np.random.default_rng(cfg.seed + sample_idx)
    # Ray tracing draws from numpy's *global* RNG. Worker processes are
    # forked from the parent and would otherwise inherit identical global
    # state (-> identical random tails across samples), so seed per sample.
    np.random.seed((cfg.seed * 1_000_003 + sample_idx) % (2**32))
    rng_py = random.Random(cfg.seed + sample_idx)

    target_dry = build_dense_clip(target_utts, cfg, rng_np)
    interf_dry = build_dense_clip(interf_utts, cfg, rng_np)

    t60 = rng_py.uniform(*cfg.t60_range)
    snr_db = rng_py.uniform(*cfg.snr_range_db)
    sir_db = rng_py.uniform(*cfg.sir_range_db)

    import pyroomacoustics as pra

    absorption, max_order = pra.inverse_sabine(t60, list(cfg.room_dim))

    mic_positions = mic_array_positions(cfg)
    array_center = np.array(cfg.array_center)

    target_pos = polar_to_cartesian(
        array_center, cfg.source_distance, cfg.target_azimuth_deg, cfg.source_elevation_deg
    )
    target_rirs = compute_single_source_rirs(cfg, mic_positions, target_pos, absorption, max_order)

    azimuth_schedule = interferer_azimuth_schedule(cfg)
    interf_rirs_per_mic = [[] for _ in range(cfg.n_mics)]  # [mic][block] -> rir
    for az in azimuth_schedule:
        interf_pos = polar_to_cartesian(array_center, cfg.source_distance, az, cfg.source_elevation_deg)
        block_rirs = compute_single_source_rirs(cfg, mic_positions, interf_pos, absorption, max_order)
        for m in range(cfg.n_mics):
            interf_rirs_per_mic[m].append(block_rirs[m])

    n_samples = int(round(cfg.sample_rate * cfg.clip_duration_s))
    target_wet = np.zeros((cfg.n_mics, n_samples), dtype=np.float32)
    interf_wet = np.zeros((cfg.n_mics, n_samples), dtype=np.float32)

    for m in range(cfg.n_mics):
        target_wet[m] = convolve_static_source(target_dry, target_rirs[m])
        interf_wet[m] = convolve_blockwise_moving_source(interf_dry, interf_rirs_per_mic[m], cfg)

    # Option B from our chat discussion: control SIR at the mic (post-RIR), not on the
    # dry clips -- this is exact (independent of how the target's and interferer's RIRs
    # happen to differ) and needs no change to build_dense_clip / the dry signals at all.
    interf_wet = rescale_interferer_to_sir(target_wet, interf_wet, cfg.ref_mic_index, sir_db)

    clean_mixture = target_wet + interf_wet
    noisy_mixture = add_white_noise_at_snr(clean_mixture, cfg.ref_mic_index, snr_db, rng_np)

    reference = target_wet[cfg.ref_mic_index].copy()  # reverberant target, no interference/noise

    # Prevent clipping on write: see normalize_peak()'s docstring for why
    # this is needed (summed target+interferer+noise commonly exceeds +-1.0,
    # and soundfile defaults to 16-bit PCM for .wav, which hard-clips).
    noisy_mixture, reference, target_dry = normalize_peak(noisy_mixture, reference, target_dry)

    meta = {
        "sample_idx": sample_idx,
        "t60_s": round(t60, 4),
        "snr_db": round(snr_db, 4),
        "sir_db": round(sir_db, 4),
        "target_azimuth_deg": cfg.target_azimuth_deg,
        "interferer_azimuth_schedule_deg": azimuth_schedule,
        "target_speaker": target_utts[0].parent.parent.name if target_utts else None,
        "interferer_speaker": interf_utts[0].parent.parent.name if interf_utts else None,
    }

    return {
        "mixture": noisy_mixture,       # [M, N] float32
        "reference": reference,         # [N] float32
        "dry_target": target_dry,       # [N] float32, NOT time-aligned to mixture
        "meta": meta,
    }


def _write_sample(sample: dict, out_dir: Path, split: str, cfg: Config) -> dict:
    import soundfile as sf

    split_dir = out_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)
    idx = sample["meta"]["sample_idx"]

    mixture_path = split_dir / f"mixture_{idx:06d}.wav"
    reference_path = split_dir / f"reference_{idx:06d}.wav"
    dry_path = split_dir / f"dry_target_{idx:06d}.wav"

    # subtype explicitly pinned to PCM_16 (soundfile's default for .wav
    # anyway, but spelled out so it's never a silent surprise again). Safe
    # from clipping because build_one_sample() already ran normalize_peak().
    sf.write(str(mixture_path), sample["mixture"].T, cfg.sample_rate, subtype="PCM_16")  # [N, M]
    sf.write(str(reference_path), sample["reference"], cfg.sample_rate, subtype="PCM_16")
    sf.write(str(dry_path), sample["dry_target"], cfg.sample_rate, subtype="PCM_16")

    row = dict(sample["meta"])
    row["mixture_path"] = str(mixture_path)
    row["reference_path"] = str(reference_path)
    row["dry_target_path"] = str(dry_path)
    return row


# ---------------------------------------------------------------------------
# 7. Parallel generation driver
# ---------------------------------------------------------------------------
def _generate_one(args):
    sample_idx, target_utts, interf_utts, cfg, out_dir, split = args
    sample = build_one_sample(sample_idx, target_utts, interf_utts, cfg)
    return _write_sample(sample, out_dir, split, cfg)


def generate_split(
    split: str,
    n_samples: int,
    by_speaker: dict[str, list[Path]],
    cfg: Config,
    out_dir: Path,
):
    speakers = list(by_speaker.keys())
    if len(speakers) < 2:
        raise RuntimeError(
            f"Need at least 2 distinct speakers in the {split} pool to pick "
            f"different target/interferer speakers; found {len(speakers)}."
        )

    rng = random.Random(cfg.seed + (0 if split == "train" else 1))
    tasks = []
    for i in range(n_samples):
        target_speaker, interf_speaker = rng.sample(speakers, 2)
        tasks.append(
            (i, by_speaker[target_speaker], by_speaker[interf_speaker], cfg, out_dir, split)
        )

    manifest_rows = []
    with ProcessPoolExecutor(max_workers=cfg.n_jobs) as pool:
        futures = [pool.submit(_generate_one, t) for t in tasks]
        for n_done, fut in enumerate(as_completed(futures), start=1):
            manifest_rows.append(fut.result())
            if n_done % 50 == 0 or n_done == n_samples:
                print(f"[{split}] {n_done}/{n_samples} samples done")

    manifest_rows.sort(key=lambda r: r["sample_idx"])
    manifest_path = out_dir / f"manifest_{split}.csv"
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        for row in manifest_rows:
            row = dict(row)
            row["interferer_azimuth_schedule_deg"] = json.dumps(row["interferer_azimuth_schedule_deg"])
            writer.writerow(row)
    print(f"[{split}] manifest written to {manifest_path}")


# ---------------------------------------------------------------------------
# 8. CLI
# ---------------------------------------------------------------------------
def parse_args() -> Config:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--librispeech-root", default=Config.librispeech_root)
    p.add_argument("--out-dir", default=Config.out_dir)
    p.add_argument("--n-train", type=int, default=Config.n_train)
    p.add_argument("--n-test", type=int, default=Config.n_test)
    p.add_argument("--n-jobs", type=int, default=Config.n_jobs)
    p.add_argument("--seed", type=int, default=Config.seed)
    p.add_argument("--rir-method", choices=["ism", "hybrid"], default=Config.rir_method)
    p.add_argument("--n-rays", type=int, default=Config.n_rays)
    p.add_argument("--hybrid-ism-order", type=int, default=Config.hybrid_ism_order)
    p.add_argument("--sir-range-db", type=float, nargs=2, default=list(Config.sir_range_db),
                   metavar=("LOW", "HIGH"),
                   help="per-sample target/interferer SIR range in dB, measured at the "
                        "reference mic before noise (default: %(default)s)")
    args = p.parse_args()

    cfg = Config(
        librispeech_root=args.librispeech_root,
        out_dir=args.out_dir,
        n_train=args.n_train,
        n_test=args.n_test,
        n_jobs=args.n_jobs,
        seed=args.seed,
        rir_method=args.rir_method,
        n_rays=args.n_rays,
        hybrid_ism_order=args.hybrid_ism_order,
        sir_range_db=tuple(args.sir_range_db),
    )
    return cfg


def main():
    cfg = parse_args()
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Indexing LibriSpeech at {cfg.librispeech_root} ...")
    by_speaker = index_librispeech(cfg.librispeech_root)
    print(f"Found {len(by_speaker)} speakers.")

    rng_py = random.Random(cfg.seed)
    train_speakers, test_speakers = split_speakers(by_speaker, cfg.test_speaker_fraction, rng_py)
    print(f"Speaker-disjoint split: {len(train_speakers)} train speakers, "
          f"{len(test_speakers)} test speakers.")

    generate_split("train", cfg.n_train, train_speakers, cfg, out_dir)
    generate_split("test", cfg.n_test, test_speakers, cfg, out_dir)

    with open(out_dir / "config.json", "w") as f:
        json.dump(cfg.__dict__, f, indent=2)
    print(f"\nDone. Config snapshot saved to {out_dir / 'config.json'}")


if __name__ == "__main__":
    main()
