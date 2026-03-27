#!/usr/bin/env python3
"""Standalone spectrogram-saving pipeline for x-foster song analysis.

This script loads `priority_bird_songpaths.json` (produced by
`audio_files_for_select_families.py`), reads each audio file, selects a middle
window around the strongest detected note cluster, and saves a spectrogram PNG
per file together with a CSV manifest.

Designed to mirror the logic used in `debug_notebook.ipynb`.
"""

from __future__ import annotations

import csv
import json
import os
import re
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.ioff()
import numpy as np
from scipy.io import wavfile
from scipy.signal import ShortTimeFFT, butter, filtfilt, get_window

try:
    import soundfile as sf
except ImportError:  # pragma: no cover
    sf = None

from tools.evfuncs import load_cbin

EPSILON = 1e-12


# -----------------------------
# Paths / configuration
# -----------------------------
DEFAULT_SONG_RESULTS_JSON = "file_management/priority_bird_songpaths.json"
DEFAULT_OUTPUT_ROOT = "file_management/xfoster_specs"
DEFAULT_MANIFEST_PATH = "file_management/xfoster_specs/spectrogram_manifest.csv"


# -----------------------------
# IO helpers
# -----------------------------
def load_song_results(filepath: str) -> Dict[str, List[Dict[str, Any]]]:
    """Load song_results dictionary from JSON."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"Loaded song_results from: {filepath}")
        return data
    except Exception as e:
        print(f"Error loading file: {e}")
        return {}


def read_audio_file(filepath: str) -> Tuple[np.ndarray, int]:
    """Read .cbin, .wav, or extensionless audio files.

    Returns
    -------
    audio : np.ndarray
        Mono float64 audio.
    sr : int
        Sample rate.
    """
    ext = os.path.splitext(filepath)[1].lower()

    if ext == ".cbin":
        audio, sr = load_cbin(filepath)
        if audio.ndim > 1:
            audio = audio[:, 0]
        return audio.astype(np.float64), int(sr)

    if sf is not None:
        try:
            audio, sr = sf.read(filepath, always_2d=False)
            if audio.ndim > 1:
                audio = audio[:, 0]
            return audio.astype(np.float64), int(sr)
        except Exception:
            pass

    sr, audio = wavfile.read(filepath)
    if audio.ndim > 1:
        audio = audio[:, 0]
    return audio.astype(np.float64), int(sr)


# -----------------------------
# Signal processing
# -----------------------------
def bandpass(audio: np.ndarray, sr: int, low: float = 500, high: float = 10000, order: int = 8) -> np.ndarray:
    nyq = sr / 2
    high = min(high, nyq - 1000)
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, audio)


def smooth_envelope(audio: np.ndarray, sr: int, smooth_ms: float = 2.0) -> np.ndarray:
    x = bandpass(audio, sr)
    x = x**2
    win = max(1, int(round(sr * smooth_ms / 1000.0)))
    kernel = np.ones(win, dtype=float) / win
    env = np.convolve(x, kernel, mode="same")
    return env


def segment_notes(
    env: np.ndarray,
    sr: int,
    threshold: float,
    min_int_ms: float = 2.0,
    min_dur_ms: float = 20.0,
) -> Tuple[np.ndarray, np.ndarray]:
    mask = env > threshold
    trans = np.diff(mask.astype(np.int8))
    on = np.where(trans == 1)[0] + 1
    off = np.where(trans == -1)[0] + 1

    if len(on) == 0 or len(off) == 0:
        return np.array([]), np.array([])

    if off[0] < on[0]:
        off = off[1:]
    if len(on) > len(off):
        on = on[: len(off)]
    elif len(off) > len(on):
        off = off[: len(on)]

    keep = [0]
    for i in range(1, len(on)):
        gap_ms = (on[i] - off[i - 1]) * 1000.0 / sr
        if gap_ms > min_int_ms:
            keep.append(i)
    on = on[keep]
    off = off[keep]

    dur_ms = (off - on) * 1000.0 / sr
    keep = dur_ms > min_dur_ms
    on = on[keep]
    off = off[keep]

    return on / sr, off / sr


def pick_middle_window(
    audio: np.ndarray,
    sr: int,
    window_sec: float = 0.75,
    threshold_mode: str = "percentile",
) -> Tuple[Tuple[float, float], np.ndarray, float, np.ndarray, np.ndarray]:
    env = smooth_envelope(audio, sr, smooth_ms=2.0)

    if np.max(env) <= 0:
        mid = len(audio) / (2 * sr)
        start_t = max(0.0, mid - window_sec / 2)
        end_t = min(len(audio) / sr, mid + window_sec / 2)
        return (start_t, end_t), env, 0.0, np.array([]), np.array([])

    env = env / np.max(env)

    if threshold_mode == "percentile":
        threshold = np.percentile(env, 90) * 0.05
    else:
        threshold = 0.05

    onsets, offsets = segment_notes(env, sr, threshold)

    if len(onsets) == 0:
        mid = len(audio) / (2 * sr)
    else:
        durations = offsets - onsets
        i = int(np.argmax(durations))
        mid = 0.5 * (onsets[i] + offsets[i])

    half = window_sec / 2.0
    start_t = max(0.0, mid - half)
    end_t = min(len(audio) / sr, mid + half)

    return (start_t, end_t), env, threshold, onsets, offsets


def make_song_spectrogram(
    audio: np.ndarray,
    fs: int,
    nfft: int = 1024,
    hop: int = 1,
    min_freq: float = 400,
    max_freq: float = 10000,
    p_low: float = 2,
    p_high: float = 98,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Birdsong-style spectrogram without fixed max duration or target-shape padding."""
    if audio is None or len(audio) == 0:
        return np.array([]), np.array([]), np.array([])

    w = get_window("hann", Nx=nfft)
    stft = ShortTimeFFT(w, hop=hop, fs=fs)

    Sx = stft.stft(audio)
    t = stft.t(len(audio))
    f = stft.f

    keep_t = t >= 0
    if len(keep_t) > 0 and nfft // 2 < len(keep_t):
        keep_t[-(nfft // 2) :] = False

    t = t[keep_t]
    Sx = Sx[:, keep_t]

    spec = np.log(np.abs(Sx) + EPSILON)

    keep_f = (f >= min_freq) & (f <= max_freq)
    f_sel = f[keep_f]
    spec = spec[keep_f, :]

    finite_vals = spec[np.isfinite(spec)]
    if finite_vals.size == 0:
        return np.zeros_like(spec), f_sel, t

    lo, hi = np.percentile(finite_vals, [p_low, p_high])
    if hi <= lo:
        hi = lo + EPSILON

    spec_norm = (spec - lo) / (hi - lo)
    spec_norm = np.clip(spec_norm, 0, 1)

    return spec_norm, f_sel, t


# -----------------------------
# Plotting / saving helpers
# -----------------------------
def safe_basename(filename: str) -> str:
    base = Path(filename).stem
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", base)
    return base


def save_one_spectrogram_record(
    bird: str,
    filepath: str,
    output_root: str,
    window_sec: float = 2,
    threshold_mode: str = "percentile",
    nfft: int = 1024,
    hop: int = 1,
    min_freq: float = 400,
    max_freq: float = 10000,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Save one spectrogram PNG for one file."""
    try:
        audio, sr = read_audio_file(filepath)

        (start_t, end_t), env, threshold, onsets, offsets = pick_middle_window(
            audio,
            sr,
            window_sec=window_sec,
            threshold_mode=threshold_mode,
        )

        s1 = int(round(start_t * sr))
        s2 = int(round(end_t * sr))
        audio_segment = audio[s1:s2]

        spec, f_sel, t_sel = make_song_spectrogram(
            audio_segment,
            fs=sr,
            nfft=nfft,
            hop=hop,
            min_freq=min_freq,
            max_freq=max_freq,
        )

        bird_dir = os.path.join(output_root, bird)
        os.makedirs(bird_dir, exist_ok=True)

        out_name = safe_basename(Path(filepath).name) + ".png"
        out_path = os.path.join(bird_dir, out_name)

        if os.path.exists(out_path) and not overwrite:
            return {
                "bird": bird,
                "filepath": filepath,
                "output_path": out_path,
                "status": "skipped_exists",
                "start_t": start_t,
                "end_t": end_t,
                "threshold": threshold,
                "num_onsets": len(onsets),
                "num_offsets": len(offsets),
            }

        fig = None
        try:
            fig, ax = plt.subplots(figsize=(10, 4))

            if spec.size > 0:
                ax.imshow(
                    spec,
                    origin="lower",
                    aspect="auto",
                    cmap="magma",
                    vmin=0,
                    vmax=1,
                    extent=[0, end_t - start_t, f_sel[0], f_sel[-1]],
                )

            for o, off in zip(onsets, offsets):
                if start_t <= o <= end_t:
                    ax.axvline(o - start_t, color="lime", linestyle="--", linewidth=1, alpha=0.7)
                if start_t <= off <= end_t:
                    ax.axvline(off - start_t, color="red", linestyle="--", linewidth=1, alpha=0.7)

            ax.set_title(f"{bird} | {Path(filepath).name}")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Frequency (Hz)")
            ax.set_ylim(0, min(max_freq, sr / 2))
            plt.tight_layout()
            fig.savefig(out_path, dpi=400, bbox_inches="tight")


        finally:
            if fig is not None:
                plt.close(fig)

        return {
            "bird": bird,
            "filepath": filepath,
            "output_path": out_path,
            "status": "saved",
            "start_t": start_t,
            "end_t": end_t,
            "threshold": threshold,
            "num_onsets": len(onsets),
            "num_offsets": len(offsets),
        }

    except Exception as e:
        return {
            "bird": bird,
            "filepath": filepath,
            "output_path": "",
            "status": f"error: {e}",
            "start_t": None,
            "end_t": None,
            "threshold": None,
            "num_onsets": None,
            "num_offsets": None,
        }


def build_spectrogram_pipeline(
    song_results: Dict[str, List[Dict[str, Any]]],
    output_root: Optional[str] = None,
    window_sec: float = 1,
    threshold_mode: str = "percentile",
    nfft: int = 1024,
    hop: int = 1,
    min_freq: float = 400,
    max_freq: float = 10000,
    overwrite: bool = False,
    max_files_per_bird: int = 5,
    use_parallel: bool = True,
    max_workers: Optional[int] = None,
    manifest_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Save one spectrogram PNG per audio file in song_results."""
    if output_root is None:
        output_root = os.path.join(os.getcwd(), "xfoster_specs")
    os.makedirs(output_root, exist_ok=True)

    tasks: List[Tuple[str, str]] = []
    for bird, files in song_results.items():
        for file_info in files[:max_files_per_bird]:
            tasks.append((bird, file_info["filepath"]))

    results: List[Dict[str, Any]] = []

    if use_parallel and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futures = [
                ex.submit(
                    save_one_spectrogram_record,
                    bird,
                    filepath,
                    output_root,
                    window_sec,
                    threshold_mode,
                    nfft,
                    hop,
                    min_freq,
                    max_freq,
                    overwrite,
                )
                for bird, filepath in tasks
            ]
            for fut in as_completed(futures):
                results.append(fut.result())
    else:
        for bird, filepath in tasks:
            results.append(
                save_one_spectrogram_record(
                    bird,
                    filepath,
                    output_root,
                    window_sec,
                    threshold_mode,
                    nfft,
                    hop,
                    min_freq,
                    max_freq,
                    overwrite,
                )
            )

    if manifest_path is None:
        manifest_path = os.path.join(output_root, "spectrogram_manifest.csv")

    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "bird",
                "filepath",
                "output_path",
                "status",
                "start_t",
                "end_t",
                "threshold",
                "num_onsets",
                "num_offsets",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"Saved manifest to: {manifest_path}")
    print(f"Output root: {output_root}")
    print(f"Total files processed: {len(results)}")
    print(f"Saved: {sum(r['status'] == 'saved' for r in results)}")
    print(f"Skipped existing: {sum(r['status'] == 'skipped_exists' for r in results)}")
    print(f"Errors: {sum(str(r['status']).startswith('error') for r in results)}")

    return results


# -----------------------------
# Optional quick inspection helpers
# -----------------------------
def test_song_file_simple(
    filepath: str,
    params: Optional[Any] = None,
    window_sec: float = 0.75,
    threshold_mode: str = "percentile",
) -> Dict[str, Any]:
    """Test segmentation on one file and plot waveform + envelope + spectrogram."""
    audio, sr = read_audio_file(filepath)

    (start_t, end_t), env, threshold, onsets, offsets = pick_middle_window(
        audio, sr, window_sec=window_sec, threshold_mode=threshold_mode
    )

    s1 = int(round(start_t * sr))
    s2 = int(round(end_t * sr))
    audio_segment = audio[s1:s2]

    nfft = getattr(params, "nfft", 1024)
    hop = getattr(params, "hop", 256)
    min_freq = getattr(params, "min_freq", 500)
    max_freq = getattr(params, "max_freq", 10000)

    spec, f_sel, t_sel = make_song_spectrogram(
        audio_segment,
        fs=sr,
        nfft=nfft,
        hop=hop,
        min_freq=min_freq,
        max_freq=max_freq,
    )

    fig, axes = plt.subplots(3, 1, figsize=(15, 10), gridspec_kw={"height_ratios": [1, 1, 2]})

    full_t = np.arange(len(audio)) / sr
    axes[0].plot(full_t, audio, linewidth=0.8)
    axes[0].axvspan(start_t, end_t, alpha=0.2)
    for o, off in zip(onsets, offsets):
        axes[0].axvline(o, color="green", alpha=0.25, linewidth=1)
        axes[0].axvline(off, color="red", alpha=0.25, linewidth=1)
    axes[0].set_title(os.path.basename(filepath))
    axes[0].set_ylabel("Amplitude")

    env_t = np.arange(len(env)) / sr
    axes[1].plot(env_t, env, linewidth=0.8)
    axes[1].axhline(threshold, linestyle="--")
    axes[1].axvspan(start_t, end_t, alpha=0.2)
    for o, off in zip(onsets, offsets):
        axes[1].axvline(o, color="green", alpha=0.25, linewidth=1)
        axes[1].axvline(off, color="red", alpha=0.25, linewidth=1)
    axes[1].set_ylabel("Envelope")

    if spec.size > 0:
        axes[2].imshow(
            spec,
            origin="lower",
            aspect="auto",
            cmap="magma",
            vmin=0,
            vmax=1,
            extent=[0, end_t - start_t, f_sel[0], f_sel[-1]],
        )
        axes[2].set_ylabel("Frequency (Hz)")
        axes[2].set_xlabel("Time (s)")
        axes[2].set_title("Song spectrogram (cropped middle window)")

    plt.tight_layout()
    plt.show()

    print(f"Sample rate: {sr}")
    print(f"Threshold: {threshold:.4f}")
    print(f"Selected window: {start_t:.3f}s to {end_t:.3f}s")
    print(f"Window duration: {end_t - start_t:.3f}s")
    print(f"Spec shape: {spec.shape}")

    return {
        "audio": audio,
        "sr": sr,
        "snippet": audio_segment,
        "spec": spec,
        "f": f_sel,
        "t": t_sel,
        "window": (start_t, end_t),
        "threshold": threshold,
        "onsets": onsets,
        "offsets": offsets,
    }


# -----------------------------
# CLI entry point
# -----------------------------
def main() -> None:
    song_results_path = DEFAULT_SONG_RESULTS_JSON
    output_root = DEFAULT_OUTPUT_ROOT
    manifest_path = DEFAULT_MANIFEST_PATH

    if not os.path.exists(song_results_path):
        raise FileNotFoundError(
            f"Could not find song results JSON: {song_results_path}"
        )

    song_results = load_song_results(song_results_path)
    if not song_results:
        raise RuntimeError("song_results is empty; nothing to process")

    results = build_spectrogram_pipeline(
        song_results=song_results,
        output_root=output_root,
        window_sec=3,
        threshold_mode="percentile",
        nfft=1024,
        hop=1,
        min_freq=400,
        max_freq=10000,
        overwrite=True,
        max_files_per_bird=5,
        use_parallel=True,
        max_workers=2,
        manifest_path=manifest_path,
    )

    # Save the full per-file result payload as JSON alongside the manifest.
    out_json = os.path.join(output_root, "spectrogram_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results JSON to: {out_json}")


if __name__ == "__main__":
    main()
