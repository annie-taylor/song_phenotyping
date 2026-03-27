#!/usr/bin/env python3
"""
Standalone song segmentation + spectrogram-saving pipeline for x-foster analysis.

What this script does:
1) Loads `file_management/priority_bird_songpaths.json`
2) Reads each audio file
3) Scores the file for song-likeness using envelope-based segmentation
4) Keeps only files that pass the song criterion
5) Saves a clean spectrogram PNG for each passing file
6) Writes a CSV manifest + JSON results

Important:
- Saved spectrograms do NOT show onset/offset overlays.
- Segmentation is used only for validation right now.
- Diagnostic plotting helpers are included for debugging only.
- Sequential processing is the default to reduce RAM pressure.
"""

from __future__ import annotations

import csv
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.ioff()

import numpy as np
from scipy.io import wavfile
from scipy.signal import ShortTimeFFT, butter, filtfilt, get_window, spectrogram

try:
    import soundfile as sf
except ImportError:  # pragma: no cover
    sf = None

from tools.evfuncs import load_cbin

EPSILON = 1e-12

# -----------------------------------------------------------------------------
# Paths / configuration
# -----------------------------------------------------------------------------
DEFAULT_SONG_RESULTS_JSON = "file_management/priority_bird_songpaths.json"
DEFAULT_OUTPUT_ROOT = "file_management/xfoster_specs"
DEFAULT_MANIFEST_PATH = "file_management/xfoster_specs/spectrogram_manifest.csv"
DEFAULT_RESULTS_JSON = "file_management/xfoster_specs/spectrogram_results.json"

# Optional aliases for birds that appear under multiple IDs in source systems.
BIRD_ALIASES = {
    "ye1tut0": ["ye1tut0", "ye1", "y1"],
}

# -----------------------------------------------------------------------------
# IO helpers
# -----------------------------------------------------------------------------
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


def resolve_audio_path(filepath: str) -> str:
    """Resolve stale relative/absolute paths when possible."""
    p = Path(filepath)
    if p.exists():
        return str(p)

    cwd_candidate = Path.cwd() / p
    if cwd_candidate.exists():
        return str(cwd_candidate.resolve())

    script_dir = Path(__file__).resolve().parent
    script_candidate = script_dir / p
    if script_candidate.exists():
        return str(script_candidate.resolve())

    return str(p)


def read_audio_file(filepath: str) -> Tuple[np.ndarray, int]:
    """
    Read .cbin, .wav, or extensionless audio files.

    Returns
    -------
    audio : np.ndarray
        Mono float64 audio.
    sr : int
        Sample rate.
    """
    filepath = resolve_audio_path(filepath)
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


# -----------------------------------------------------------------------------
# Signal processing
# -----------------------------------------------------------------------------
def bandpass(
    audio: np.ndarray,
    sr: int,
    low: float = 500,
    high: float = 10000,
    order: int = 8,
) -> np.ndarray:
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
    """Threshold-based syllable segmentation on the smoothed envelope."""
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


def score_song_candidate(
    audio: np.ndarray,
    sr: int,
    window_sec: float = 2.0,
    min_segments: int = 5,
    step_sec: float = 0.25,
    threshold_mode: str = "percentile",
) -> Dict[str, Any]:
    """
    Score a file for song-likeness using envelope-based segmentation and a
    sliding-window count of detected segments.
    """
    env = smooth_envelope(audio, sr, smooth_ms=2.0)

    if np.max(env) <= 0:
        return {
            "passed": False,
            "max_segments": 0,
            "best_window": None,
            "onsets": np.array([]),
            "offsets": np.array([]),
            "threshold": 0.0,
        }

    env = env / np.max(env)

    if threshold_mode == "percentile":
        threshold = np.percentile(env, 90) * 0.3
    else:
        threshold = 0.05

    onsets, offsets = segment_notes(env, sr, threshold)

    if len(onsets) == 0:
        return {
            "passed": False,
            "max_segments": 0,
            "best_window": None,
            "onsets": onsets,
            "offsets": offsets,
            "threshold": threshold,
        }

    duration = len(audio) / sr
    if duration <= window_sec:
        count = len(onsets)
        passed = count >= min_segments
        return {
            "passed": passed,
            "max_segments": count,
            "best_window": (0.0, min(window_sec, duration)) if passed else None,
            "onsets": onsets,
            "offsets": offsets,
            "threshold": threshold,
        }

    best_count = 0
    best_start = 0.0

    starts = np.arange(0, duration - window_sec + 1e-9, step_sec)
    for start_t in starts:
        end_t = start_t + window_sec
        overlap = (onsets < end_t) & (offsets > start_t)
        count = int(np.sum(overlap))
        if count > best_count:
            best_count = count
            best_start = start_t

    passed = best_count >= min_segments
    best_window = (best_start, best_start + window_sec) if passed else None

    return {
        "passed": passed,
        "max_segments": best_count,
        "best_window": best_window,
        "onsets": onsets,
        "offsets": offsets,
        "threshold": threshold,
    }


# -----------------------------------------------------------------------------
# Spectrogram helpers
# -----------------------------------------------------------------------------
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
    """
    Birdsong-style spectrogram without fixed max duration or target-shape padding.
    """
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


def spectrogram_for_plot(audio: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fallback spectrogram for diagnostic plotting."""
    f, tt, Sxx = spectrogram(audio, fs=sr, nperseg=1024, noverlap=768)
    Sxx = 10 * np.log10(Sxx + EPSILON)
    return f, tt, Sxx


# -----------------------------------------------------------------------------
# Optional debugging plot
# -----------------------------------------------------------------------------
def plot_segmentation_summary(
    audio: np.ndarray,
    fs: int,
    onsets: np.ndarray,
    offsets: np.ndarray,
    threshold: Optional[float] = None,
    out_path: Optional[str] = None,
    title: str = "",
) -> None:
    """
    Optional diagnostic plot of waveform + envelope + spectrogram.

    This is for debugging only. The production song-classification / spectrogram
    save path does not project detected segments onto saved PNGs.
    """
    fig = None
    try:
        env = smooth_envelope(audio, fs, smooth_ms=2.0)
        if np.max(env) > 0:
            env = env / np.max(env)

        t = np.arange(len(audio)) / fs
        te = np.arange(len(env)) / fs

        fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=False)

        axes[0].plot(t, audio)
        for o, off in zip(onsets, offsets):
            axes[0].axvspan(o, off, alpha=0.2)
        axes[0].set_title(title or "Waveform with detected segments")
        axes[0].set_ylabel("Amplitude")

        axes[1].plot(te, env)
        if threshold is not None:
            axes[1].axhline(threshold, linestyle="--")
        for o, off in zip(onsets, offsets):
            axes[1].axvspan(o, off, alpha=0.2)
        axes[1].set_title("Smoothed envelope + threshold")
        axes[1].set_ylabel("Normalized envelope")

        f, tt, Sxx = spectrogram_for_plot(audio, fs)
        axes[2].pcolormesh(tt, f, Sxx, shading="auto", cmap="magma")
        for o, off in zip(onsets, offsets):
            axes[2].axvspan(o, off, alpha=0.2)
        axes[2].set_ylim(400, min(10000, fs / 2))
        axes[2].set_title("Spectrogram with detected segments")
        axes[2].set_xlabel("Time (s)")
        axes[2].set_ylabel("Frequency (Hz)")

        plt.tight_layout()
        if out_path:
            fig.savefig(out_path, dpi=400, bbox_inches="tight")
        else:
            plt.show()
    finally:
        if fig is not None:
            plt.close(fig)


# -----------------------------------------------------------------------------
# Main per-file processing
# -----------------------------------------------------------------------------
def safe_basename(filename: str) -> str:
    base = Path(filename).stem
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", base)
    return base


def process_one_file(
    bird: str,
    filepath: str,
    output_root: str,
    window_sec: float = 2.0,
    min_segments: int = 5,
    step_sec: float = 0.25,
    threshold_mode: str = "percentile",
    overwrite: bool = False,
) -> Dict[str, Any]:
    """
    Process a single file and return a manifest row.

    The segmentation result is used for validation now; saved spectrogram PNGs are
    kept clean and do not display onset/offset overlays.
    """
    fig = None
    try:
        filepath = resolve_audio_path(filepath)
        audio, fs = read_audio_file(filepath)

        score = score_song_candidate(
            audio,
            fs,
            window_sec=window_sec,
            min_segments=min_segments,
            step_sec=step_sec,
            threshold_mode=threshold_mode,
        )

        passed = bool(score["passed"])
        n_segments = int(score["max_segments"])
        threshold = float(score["threshold"])
        best_window = score["best_window"]
        onsets = score["onsets"]
        offsets = score["offsets"]

        bird_dir = os.path.join(output_root, bird)
        os.makedirs(bird_dir, exist_ok=True)

        out_name = safe_basename(Path(filepath).name) + ".png"
        out_path = os.path.join(bird_dir, out_name)

        if not passed:
            return {
                "bird": bird,
                "filepath": filepath,
                "output_path": "",
                "status": "non_song",
                "n_segments": n_segments,
                "threshold": threshold,
                "best_window_start": best_window[0] if best_window else None,
                "best_window_end": best_window[1] if best_window else None,
                "n_onsets": int(len(onsets)),
                "n_offsets": int(len(offsets)),
            }

        if os.path.exists(out_path) and not overwrite:
            return {
                "bird": bird,
                "filepath": filepath,
                "output_path": out_path,
                "status": "skipped_exists",
                "n_segments": n_segments,
                "threshold": threshold,
                "best_window_start": best_window[0] if best_window else None,
                "best_window_end": best_window[1] if best_window else None,
                "n_onsets": int(len(onsets)),
                "n_offsets": int(len(offsets)),
            }

        start_t, end_t = best_window
        s1 = int(round(start_t * fs))
        s2 = int(round(end_t * fs))
        audio_segment = audio[s1:s2]

        spec, f_sel, t_sel = make_song_spectrogram(
            audio_segment,
            fs=fs,
            nfft=1024,
            hop=1,
            min_freq=400,
            max_freq=10000,
        )

        fig, ax = plt.subplots(figsize=(10, 4))
        try:
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

            #ax.set_title(f"{bird} | {Path(filepath).name}")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Frequency (Hz)")
            ax.set_ylim(400, min(10000, fs / 2))
            plt.tight_layout()
            fig.savefig(out_path, dpi=400, bbox_inches="tight")
        finally:
            plt.close(fig)

        return {
            "bird": bird,
            "filepath": filepath,
            "output_path": out_path,
            "status": "song",
            "n_segments": n_segments,
            "threshold": threshold,
            "best_window_start": start_t,
            "best_window_end": end_t,
            "n_onsets": int(len(onsets)),
            "n_offsets": int(len(offsets)),
        }

    except Exception as e:
        if fig is not None:
            plt.close(fig)
        return {
            "bird": bird,
            "filepath": filepath,
            "output_path": "",
            "status": f"error: {repr(e)}",
            "n_segments": None,
            "threshold": None,
            "best_window_start": None,
            "best_window_end": None,
            "n_onsets": None,
            "n_offsets": None,
        }


def build_spectrogram_pipeline(
    song_results: Dict[str, List[Dict[str, Any]]],
    output_root: Optional[str] = None,
    window_sec: float = 2.0,
    min_segments: int = 5,
    step_sec: float = 0.25,
    threshold_mode: str = "percentile",
    overwrite: bool = False,
    max_files_per_bird: int = 5,
    use_parallel: bool = False,
    max_workers: Optional[int] = None,
    manifest_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Save one spectrogram PNG per passing audio file in song_results.
    """
    if output_root is None:
        output_root = os.path.join(os.getcwd(), "xfoster_specs")
    os.makedirs(output_root, exist_ok=True)

    tasks: List[Tuple[str, str]] = []
    for bird, files in song_results.items():
        for file_info in files[:max_files_per_bird]:
            tasks.append((bird, file_info["filepath"]))

    results: List[Dict[str, Any]] = []

    if use_parallel and len(tasks) > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futures = [
                ex.submit(
                    process_one_file,
                    bird,
                    filepath,
                    output_root,
                    window_sec,
                    min_segments,
                    step_sec,
                    threshold_mode,
                    overwrite,
                )
                for bird, filepath in tasks
            ]
            for fut in as_completed(futures):
                results.append(fut.result())
    else:
        for bird, filepath in tasks:
            row = process_one_file(
                bird=bird,
                filepath=filepath,
                output_root=output_root,
                window_sec=window_sec,
                min_segments=min_segments,
                step_sec=step_sec,
                threshold_mode=threshold_mode,
                overwrite=overwrite,
            )
            results.append(row)
            print(f"{bird}: {row['status']} | n_segments={row['n_segments']}")

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
                "n_segments",
                "threshold",
                "best_window_start",
                "best_window_end",
                "n_onsets",
                "n_offsets",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"Saved manifest to: {manifest_path}")
    print(f"Output root: {output_root}")
    print(f"Total files processed: {len(results)}")
    print(f"Song files: {sum(r['status'] == 'song' for r in results)}")
    print(f"Non-song files: {sum(r['status'] == 'non_song' for r in results)}")
    print(f"Skipped existing: {sum(r['status'] == 'skipped_exists' for r in results)}")
    print(f"Errors: {sum(str(r['status']).startswith('error') for r in results)}")

    return results


# -----------------------------------------------------------------------------
# Convenience helpers
# -----------------------------------------------------------------------------
def load_birds_from_txt(txt_path: str) -> List[str]:
    with open(txt_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def expand_bird_aliases(bird: str) -> List[str]:
    return BIRD_ALIASES.get(bird, [bird])


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    song_results_path = DEFAULT_SONG_RESULTS_JSON
    output_root = DEFAULT_OUTPUT_ROOT
    manifest_path = DEFAULT_MANIFEST_PATH

    if not os.path.exists(song_results_path):
        raise FileNotFoundError(f"Could not find song results JSON: {song_results_path}")

    song_results = load_song_results(song_results_path)
    if not song_results:
        raise RuntimeError("song_results is empty; nothing to process")

    results = build_spectrogram_pipeline(
        song_results=song_results,
        output_root=output_root,
        window_sec=6.0,
        min_segments=8,
        step_sec=0.25,
        threshold_mode="percentile",
        overwrite=True,
        max_files_per_bird=5,
        use_parallel=False,
        max_workers=2,
        manifest_path=manifest_path,
    )

    with open(DEFAULT_RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results JSON to: {DEFAULT_RESULTS_JSON}")


if __name__ == "__main__":
    main()
