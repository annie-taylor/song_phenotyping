"""HTML-based song catalogs for visual inspection (Stage F utility).

Replaces the ReportLab PDF approach with matplotlib + base64 PNG embedded in
HTML.  Produces two separate catalog types:

1. **Song catalog** — one continuous spectrogram per song, with manual labels
   above and automated cluster labels below, sorted chronologically.

2. **Syllable-type catalog** — for each unique label, a grid of individual
   syllable spectrograms drawn directly from the Stage A HDF5 arrays (no
   audio files required).

Public API
----------
- :func:`generate_song_catalog` — chronological song view for one bird.
- :func:`generate_syllable_type_catalog` — per-label spectrogram grid.
- :func:`generate_sequencing_catalog` — transition matrices and repeat stats.
- :func:`generate_cluster_quality_catalog` — per-cluster feature stats,
  discriminative feature violins, outgoing transitions, and repeat-ramping
  analysis.  Output file uses the name ``syllable_characteristics``.
- :func:`generate_all_catalogs` — convenience wrapper that calls all of the above.
- :class:`CatalogConfig` — configurable display parameters.

Examples
--------
>>> from song_phenotyping.catalog import generate_all_catalogs
>>> results = generate_all_catalogs("/Volumes/Extreme SSD/pipeline_runs/or18or24", rank=0)
>>> results['song_catalog']
'.../or18or24/syllable_data/html/or18or24_song_catalog_rank0.html'
"""

from __future__ import annotations

import base64
import gc
import io
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import tables
from scipy.ndimage import zoom as _ndimage_zoom
from scipy.stats import linregress as _linregress

from song_phenotyping.tools.audio_utils import read_audio_file
from song_phenotyping.tools.label_handler import LabelHandler, LabelType
from song_phenotyping.signal import butter_bandpass_filter_sos, get_song_spec, rms_norm
from song_phenotyping.tools.spectrogram_configs import SpectrogramParams
from song_phenotyping.tools.system_utils import replace_macaw_root

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class CatalogConfig:
    """Shared configuration for both catalog types."""
    # Song catalog
    n_songs: int = 30               # max songs to include
    song_duration: float = 6.0      # seconds of audio to show per song
    song_fig_width: float = 14.0    # inches
    song_fig_height: float = 3.5    # inches

    # Syllable-type catalog
    n_per_type: int = 30            # max syllables shown per label type
    grid_cols: int = 6              # columns in the syllable grid
    syl_fig_size: float = 1.2       # inches per syllable thumbnail

    # Shared
    dpi: int = 120
    overwrite: bool = True


# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------

# Fixed palette so label colours are consistent across figures
_PALETTE = plt.cm.tab20.colors  # 20 distinct colours

def _label_color(label) -> tuple:
    """Return a consistent RGB colour for a label."""
    idx = hash(str(label)) % len(_PALETTE)
    return _PALETTE[idx]


def _fig_to_b64(fig: plt.Figure, dpi: int) -> str:
    """Render a matplotlib Figure to a base64-encoded PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode('ascii')
    plt.close(fig)
    gc.collect()
    return encoded


# ---------------------------------------------------------------------------
# Timestamp parsing (carried over from song_catalog_pdf.py)
# ---------------------------------------------------------------------------

def _parse_timestamp(filename: str) -> Optional[Dict[str, Any]]:
    """Extract recording datetime from an audio filename stem."""
    patterns = [
        r'(\w+)_(\d{8})_(\d{6})',
        r'(\w+)_(\d{6})_(\d{6})',
        r'(\d{8})_(\d{6})_(\w+)',
        r'(\d{6})_(\d{6})_(\w+)',
        r'(\w+)_(\d{8})-(\d{6})',
        r'(\w+)_(\d{6})-(\d{6})',
    ]
    for pattern in patterns:
        m = re.search(pattern, filename)
        if not m:
            continue
        groups = m.groups()
        bird_name, date_str, time_str = (
            (groups[2], groups[0], groups[1]) if pattern.startswith(r'(\d')
            else (groups[0], groups[1], groups[2])
        )
        try:
            date_fmt = _detect_date_fmt(date_str)
            date_obj = datetime.strptime(date_str, date_fmt)
            time_str = re.sub(r'\D', '', time_str).ljust(6, '0')[:6]
            h, mn, s = int(time_str[:2]), int(time_str[2:4]), int(time_str[4:6])
            h, mn, s = h % 24, mn % 60, s % 60
            time_str_clean = f'{h:02d}{mn:02d}{s:02d}'
            time_obj = datetime.strptime(time_str_clean, '%H%M%S')
            full_dt = datetime.combine(date_obj.date(), time_obj.time())
            return {
                'datetime': full_dt,
                'date': date_obj.strftime('%Y-%m-%d'),
                'time': time_obj.strftime('%H:%M:%S'),
                'bird': bird_name,
            }
        except ValueError:
            continue
    return None


def _detect_date_fmt(s: str) -> str:
    current_year = datetime.now().year
    if len(s) == 8:
        y = int(s[:4])
        if 1900 <= y <= current_year:
            return '%Y%m%d'
    elif len(s) == 6:
        return '%d%m%y'
    raise ValueError(f'Unrecognised date string: {s}')


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _read_audio_filename_from_h5(h5file) -> Optional[str]:
    try:
        raw = h5file.root.audio_filename.read()
        name = raw[0].decode('utf-8') if isinstance(raw[0], bytes) else str(raw[0])
        if not Path(name).is_file():
            name = replace_macaw_root(name)
        return name if Path(name).is_file() else None
    except Exception:
        return None


def _load_auto_labels_from_db(
    syllable_db: Optional[pd.DataFrame],
    song_filename: str,
    rank: int,
) -> Optional[np.ndarray]:
    """Return auto cluster labels for one song file, or None."""
    if syllable_db is None:
        return None
    song_data = syllable_db[syllable_db['song_file'] == song_filename]
    if song_data.empty:
        return None
    prefix = f'cluster_rank{rank}_'
    cols = [c for c in song_data.columns if c.startswith(prefix)]
    if not cols:
        return None
    raw = song_data[cols[0]].values
    return np.array([int(v) if pd.notna(v) else -1 for v in raw])


def _sorted_syllable_files(bird_path: Path) -> List[Tuple[Path, Dict]]:
    """Return syllable HDF5 files sorted chronologically."""
    from song_phenotyping.tools.pipeline_paths import SPECS_DIR
    specs_dir = bird_path / SPECS_DIR
    if not specs_dir.exists():
        return []
    files = list(specs_dir.glob('syllables_*.h5'))
    results = []
    for f in files:
        try:
            ts = _parse_timestamp(f.stem) or {
                'datetime': None, 'date': '?', 'time': '?', 'bird': bird_path.name
            }
        except Exception:
            ts = {'datetime': None, 'date': '?', 'time': '?', 'bird': bird_path.name}
        results.append((f, ts))
    results.sort(key=lambda x: x[1]['datetime'] or datetime.max)
    return results


def _load_syllable_db(bird_path: Path) -> Optional[pd.DataFrame]:
    from song_phenotyping.tools.pipeline_paths import STAGES_DIR
    csv = bird_path / STAGES_DIR / 'syllable_database' / 'syllable_features.csv'
    if csv.exists():
        try:
            return pd.read_csv(csv)
        except Exception as e:
            logger.warning(f'Could not load syllable DB: {e}')
    return None


# ---------------------------------------------------------------------------
# Song catalog — figure rendering
# ---------------------------------------------------------------------------

def _render_song_figure(
    spec: np.ndarray,
    duration: float,
    onsets_ms: np.ndarray,
    offsets_ms: np.ndarray,
    t_start_s: float,
    manual_labels: Optional[np.ndarray],
    auto_labels: Optional[np.ndarray],
    bird_name: str,
    rank: int,
    config: CatalogConfig,
) -> str:
    """Return a base64 PNG of a single continuous-song spectrogram with labels."""
    fig, ax = plt.subplots(figsize=(config.song_fig_width, config.song_fig_height))

    ax.imshow(
        spec, aspect='auto', origin='lower', cmap='plasma',
        extent=[0, duration, 0, spec.shape[0]],
    )
    ax.set_xlim(0, duration)
    ax.set_ylim(-28, spec.shape[0] + 28)
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.set_yticks([])
    ax.set_xlabel('Time (s)', fontsize=9)

    for i, (on, off) in enumerate(zip(onsets_ms, offsets_ms)):
        cx = (on + off) / 2 / 1000 - t_start_s
        if not (0 <= cx <= duration):
            continue
        if manual_labels is not None and i < len(manual_labels):
            lbl = manual_labels[i]
            if str(lbl) not in ('s', 'z', '\r'):
                c = _label_color(lbl)
                ax.text(cx, spec.shape[0] + 4, str(lbl), fontsize=8,
                        ha='center', va='bottom', color='black',
                        bbox=dict(facecolor=c, edgecolor='none', alpha=0.85,
                                  boxstyle='round,pad=0.15'))
        if auto_labels is not None and i < len(auto_labels):
            lbl = auto_labels[i]
            if int(lbl) >= 0:
                c = _label_color(lbl)
                ax.text(cx, -4, str(lbl), fontsize=8,
                        ha='center', va='top', color='black',
                        bbox=dict(facecolor=c, edgecolor='none', alpha=0.85,
                                  boxstyle='round,pad=0.15'))

    label_str = []
    if manual_labels is not None:
        label_str.append('manual')
    if auto_labels is not None:
        label_str.append(f'auto rank {rank}')
    ax.set_title(
        f'{bird_name}  —  {" + ".join(label_str) or "no labels"}',
        fontsize=9, pad=16,
    )

    fig.tight_layout()
    return _fig_to_b64(fig, config.dpi)


# ---------------------------------------------------------------------------
# Syllable-type catalog — figure rendering
# ---------------------------------------------------------------------------

def _render_syllable_grid(
    specs: List[np.ndarray],
    label: Any,
    label_source: str,
    n_cols: int,
    syl_fig_size: float,
    dpi: int,
) -> str:
    """Return a base64 PNG grid of syllable spectrograms for one label."""
    n = len(specs)
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * syl_fig_size, n_rows * syl_fig_size),
        squeeze=False,
    )
    color = _label_color(label)
    fig.patch.set_facecolor((*color[:3], 0.08))

    for idx, ax in enumerate(axes.flat):
        if idx < n:
            spec = specs[idx]
            ax.imshow(spec, aspect='auto', origin='lower', cmap='plasma')
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(0.4)
        else:
            ax.set_visible(False)

    fig.suptitle(f'{label_source} label "{label}"  (n={n})',
                 fontsize=9, y=1.01)
    fig.tight_layout(pad=0.2)
    return _fig_to_b64(fig, dpi)


# ---------------------------------------------------------------------------
# HTML template helpers
# ---------------------------------------------------------------------------

_HTML_HEAD = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{title}</title>
<style>
  body {{ font-family: sans-serif; background: #1a1a2e; color: #eee;
          margin: 0; padding: 16px; }}
  h1   {{ font-size: 1.4em; margin-bottom: 4px; }}
  .meta {{ font-size: 0.8em; color: #aaa; margin: 2px 0 8px 0; }}
  .song {{ background: #16213e; border-radius: 6px; padding: 10px 12px;
           margin-bottom: 14px; }}
  .song img {{ max-width: 100%; display: block; border-radius: 4px; }}
  .type-section {{ background: #16213e; border-radius: 6px; padding: 10px 12px;
                   margin-bottom: 20px; }}
  .type-section img {{ max-width: 100%; display: block; border-radius: 4px; }}
  .summary {{ font-size: 0.85em; color: #9ea; margin-bottom: 16px; }}
</style>
</head>
<body>
<h1>{title}</h1>
<p class="meta">Generated: {generated} &nbsp;|&nbsp; Bird: {bird} &nbsp;|&nbsp; {params}</p>
<p class="summary">{summary}</p>
"""

_HTML_FOOT = "</body></html>\n"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_song_catalog(
    bird_path: str,
    rank: int = 0,
    config: Optional[CatalogConfig] = None,
) -> str:
    """Generate an HTML song catalog for one bird.

    Each entry shows a continuous-song spectrogram spanning
    ``config.song_duration`` seconds, with manual syllable labels rendered
    above the spectrogram and automated cluster labels below.  Songs are
    sorted chronologically using timestamps parsed from the audio filenames.

    Parameters
    ----------
    bird_path : str
        Path to the bird's root directory (must contain
        ``syllable_data/specs/`` from Stage A).
    rank : int, optional
        Clustering rank to use for automated labels (0 = best-ranked
        HDBSCAN result).  Default is ``0``.
    config : CatalogConfig or None, optional
        Display parameters.  Uses default :class:`CatalogConfig` when
        ``None``.

    Returns
    -------
    str
        Absolute path to the generated HTML file, or ``''`` on failure.
    """
    cfg = config or CatalogConfig()
    from song_phenotyping.tools.pipeline_paths import CATALOG_DIR
    bird_path = Path(bird_path)
    bird_name = bird_path.name
    out_dir = bird_path / CATALOG_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'{bird_name}_song_catalog_rank{rank}.html'

    if out_path.exists() and not cfg.overwrite:
        return str(out_path)

    syllable_db = _load_syllable_db(bird_path)
    spec_params = SpectrogramParams()
    spec_params.max_dur = cfg.song_duration

    sorted_files = _sorted_syllable_files(bird_path)
    if not sorted_files:
        logger.warning(f'No syllable files found for {bird_name}')
        return ''

    sorted_files = sorted_files[:cfg.n_songs]
    sections = []
    n_rendered = 0

    for syl_file, ts_info in sorted_files:
        try:
            with tables.open_file(str(syl_file), 'r') as hf:
                onsets  = hf.root.onsets.read().astype(float)   # ms
                offsets = hf.root.offsets.read().astype(float)  # ms
                audio_path = _read_audio_filename_from_h5(hf)
                manual_raw = (
                    hf.root.manual.read()
                    if hasattr(hf.root, 'manual') else None
                )

            if audio_path is None:
                logger.debug(f'Audio not found for {syl_file.name}, skipping')
                continue

            # Manual labels
            handler_m = LabelHandler(LabelType.MANUAL)
            manual_labels = (
                np.array(handler_m.normalize_labels(list(manual_raw)))
                if manual_raw is not None and len(manual_raw) > 0 else None
            )

            # Auto labels
            auto_labels = _load_auto_labels_from_db(
                syllable_db, syl_file.name, rank
            )

            if manual_labels is None and auto_labels is None:
                continue

            # Load audio and generate spectrogram
            audio, fs = read_audio_file(audio_path)
            audio = rms_norm(audio)
            audio = butter_bandpass_filter_sos(
                audio, lowcut=spec_params.min_freq,
                highcut=spec_params.max_freq, fs=fs, order=5,
            )

            t_start = max(onsets[0] / 1000 - 0.25, 0.0)
            t_end   = min(t_start + cfg.song_duration,
                          (len(audio) - 1) / fs)

            spec, _, _ = get_song_spec(
                t1=t_start, t2=t_end, audio=audio,
                params=spec_params, fs=fs, downsample=False,
            )

            # Trim to syllables within window
            mask = (onsets >= t_start * 1000) & (onsets <= t_end * 1000)
            onsets_w  = onsets[mask]
            offsets_w = offsets[mask]
            manual_w  = manual_labels[mask] if manual_labels is not None else None
            auto_w    = auto_labels[mask]   if auto_labels   is not None else None

            if len(onsets_w) < 2:
                continue

            b64 = _render_song_figure(
                spec=spec,
                duration=t_end - t_start,
                onsets_ms=onsets_w,
                offsets_ms=offsets_w,
                t_start_s=t_start,
                manual_labels=manual_w,
                auto_labels=auto_w,
                bird_name=bird_name,
                rank=rank,
                config=cfg,
            )

            date_str = ts_info.get('date', '?')
            time_str = ts_info.get('time', '?')
            sections.append(
                f'<div class="song">'
                f'<p class="meta">{date_str} &nbsp; {time_str} &nbsp; '
                f'| &nbsp; {syl_file.name}</p>'
                f'<img src="data:image/png;base64,{b64}" '
                f'alt="song spectrogram" />'
                f'</div>\n'
            )
            n_rendered += 1

        except Exception as e:
            logger.error(f'Error rendering {syl_file.name}: {e}', exc_info=True)
            continue

    if not sections:
        logger.warning(f'No songs rendered for {bird_name}')
        return ''

    html_parts = [
        _HTML_HEAD.format(
            title=f'{bird_name} — Song Catalog',
            generated=datetime.now().strftime('%Y-%m-%d %H:%M'),
            bird=bird_name,
            params=f'rank={rank} &nbsp; duration={cfg.song_duration}s',
            summary=f'{n_rendered} songs rendered (of {len(sorted_files)} available)',
        )
    ]
    html_parts.extend(sections)
    html_parts.append(_HTML_FOOT)

    out_path.write_text(''.join(html_parts), encoding='utf-8')
    logger.info(f'Song catalog written: {out_path}')
    return str(out_path)


def generate_syllable_type_catalog(
    bird_path: str,
    rank: int = 0,
    label_source: str = 'auto',
    config: Optional[CatalogConfig] = None,
) -> str:
    """Generate an HTML syllable-type catalog for one bird.

    For each unique label, renders a grid of up to ``config.n_per_type``
    individual syllable spectrograms drawn directly from the Stage A HDF5
    arrays — no audio files required.

    Parameters
    ----------
    bird_path : str
        Path to the bird's root directory.
    rank : int, optional
        Clustering rank to use when *label_source* is ``'auto'``.
        Default is ``0``.
    label_source : {'auto', 'manual'}, optional
        Which label set to group syllables by.  Default is ``'auto'``.
    config : CatalogConfig or None, optional
        Display parameters.  Uses default :class:`CatalogConfig` when
        ``None``.

    Returns
    -------
    str
        Absolute path to the generated HTML file, or ``''`` on failure.
    """
    cfg = config or CatalogConfig()
    from song_phenotyping.tools.pipeline_paths import CATALOG_DIR, SPECS_DIR
    bird_path = Path(bird_path)
    bird_name = bird_path.name
    out_dir = bird_path / CATALOG_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    src_tag = f'rank{rank}' if label_source == 'auto' else 'manual'
    out_path = out_dir / f'{bird_name}_syllable_types_{label_source}_{src_tag}.html'

    if out_path.exists() and not cfg.overwrite:
        return str(out_path)

    syllable_db = _load_syllable_db(bird_path) if label_source == 'auto' else None
    specs_dir   = bird_path / SPECS_DIR

    if not specs_dir.exists():
        logger.warning(f'No specs directory for {bird_name}')
        return ''

    # Collect spectrograms grouped by label
    label_to_specs: Dict[Any, List[np.ndarray]] = {}

    for syl_file in sorted(specs_dir.glob('syllables_*.h5')):
        try:
            with tables.open_file(str(syl_file), 'r') as hf:
                specs_arr = hf.root.spectrograms.read()   # (n, freq, time)
                manual_raw = (
                    hf.root.manual.read()
                    if hasattr(hf.root, 'manual') else None
                )

            if label_source == 'manual':
                if manual_raw is None or len(manual_raw) == 0:
                    continue
                handler = LabelHandler(LabelType.MANUAL)
                labels = handler.normalize_labels(list(manual_raw))
            else:
                labels_arr = _load_auto_labels_from_db(
                    syllable_db, syl_file.name, rank
                )
                if labels_arr is None:
                    continue
                labels = list(labels_arr)

            if len(labels) != len(specs_arr):
                n = min(len(labels), len(specs_arr))
                labels, specs_arr = labels[:n], specs_arr[:n]

            for lbl, spec in zip(labels, specs_arr):
                # Skip boundary/noise tokens
                if str(lbl) in ('s', 'z', '\r') or (
                    label_source == 'auto' and int(lbl) < 0
                ):
                    continue
                label_to_specs.setdefault(lbl, []).append(spec)

        except Exception as e:
            logger.error(f'Error loading {syl_file.name}: {e}', exc_info=True)
            continue

    if not label_to_specs:
        logger.warning(f'No labelled syllables found for {bird_name}')
        return ''

    # Sort labels (strings alphabetically, ints numerically)
    try:
        sorted_labels = sorted(label_to_specs.keys(), key=lambda x: int(x))
    except (TypeError, ValueError):
        sorted_labels = sorted(label_to_specs.keys(), key=str)

    sections = []
    for lbl in sorted_labels:
        all_specs = label_to_specs[lbl]
        subset = all_specs[:cfg.n_per_type]
        b64 = _render_syllable_grid(
            specs=subset,
            label=lbl,
            label_source=label_source,
            n_cols=cfg.grid_cols,
            syl_fig_size=cfg.syl_fig_size,
            dpi=cfg.dpi,
        )
        sections.append(
            f'<div class="type-section">'
            f'<img src="data:image/png;base64,{b64}" '
            f'alt="syllable grid for label {lbl}" />'
            f'</div>\n'
        )

    total_syls = sum(len(v) for v in label_to_specs.values())
    html_parts = [
        _HTML_HEAD.format(
            title=f'{bird_name} — Syllable Types ({label_source})',
            generated=datetime.now().strftime('%Y-%m-%d %H:%M'),
            bird=bird_name,
            params=(
                f'source={label_source}'
                + (f' rank={rank}' if label_source == 'auto' else '')
                + f' &nbsp; n_per_type={cfg.n_per_type}'
            ),
            summary=(
                f'{len(sorted_labels)} label types &nbsp;|&nbsp; '
                f'{total_syls} total syllables'
            ),
        )
    ]
    html_parts.extend(sections)
    html_parts.append(_HTML_FOOT)

    out_path.write_text(''.join(html_parts), encoding='utf-8')
    logger.info(f'Syllable-type catalog written: {out_path}')
    return str(out_path)


# ---------------------------------------------------------------------------
# Sequencing catalog (transition matrices, repeat counts, syllable proportions)
# ---------------------------------------------------------------------------

def generate_sequencing_catalog(
    bird_path: str,
    rank: int = 0,
    config: Optional[CatalogConfig] = None,
) -> str:
    """Generate an HTML sequencing catalog for one bird.

    Reads the automated phenotype pickle from Stage E and renders transition
    matrices, repeat counts, syllable proportions, and summary statistics as
    embedded base64 PNGs inside a single HTML file.

    Parameters
    ----------
    bird_path : str
        Path to the bird's root directory (must contain
        ``stages/05_phenotype/`` from Stage E).
    rank : int, optional
        Clustering rank to use (0 = best-ranked result).  Default is ``0``.
    config : CatalogConfig or None, optional
        Display parameters.  Uses default :class:`CatalogConfig` when ``None``.

    Returns
    -------
    str
        Absolute path to the generated HTML file, or ``''`` on failure.
    """
    cfg = config or CatalogConfig()
    from song_phenotyping.tools.pipeline_paths import CATALOG_DIR, PHENOTYPE_DIR
    bird_path = Path(bird_path)
    bird_name = bird_path.name
    out_dir = bird_path / CATALOG_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'{bird_name}_sequencing_rank{rank}.html'

    if out_path.exists() and not cfg.overwrite:
        return str(out_path)

    pkl_path = bird_path / PHENOTYPE_DIR / f'automated_phenotype_data_rank{rank}.pkl'
    if not pkl_path.exists():
        logger.warning(f'Sequencing catalog: pkl not found: {pkl_path}')
        return ''

    try:
        data = pd.read_pickle(str(pkl_path))
    except Exception as e:
        logger.warning(f'Sequencing catalog: failed to load pkl ({e})', exc_info=True)
        return ''

    pheno = data.get('phenotype_results', data)
    sections = []

    # ------------------------------------------------------------------
    # 1. Summary stats table
    # ------------------------------------------------------------------
    stat_keys = [
        ('repertoire_size', 'Repertoire size'),
        ('n_songs', 'Songs'),
        ('n_syllables_total', 'Total syllables'),
        ('entropy', 'Transition entropy'),
        ('entropy_scaled', 'Entropy (scaled)'),
        ('has_intro_notes', 'Has intro notes'),
        ('intro_note_labels', 'Intro note syllables'),
        ('mean_intro_count_per_song', 'Mean intro notes per song'),
        ('std_intro_count_per_song', 'Std intro notes per song'),
        ('intro_recurs_in_song', 'Intro note recurs in song'),
        ('repeat_bool', 'Has repeats'),
        ('dyad_bool', 'Has dyads'),
        ('num_dyad', 'Dyad count (excluded from mean/median)'),
        ('num_longer_reps', 'Longer repeats'),
        ('mean_repeat_syls', 'Mean repeat length'),
        ('median_repeat_syls', 'Median repeat length'),
        ('var_repeat_syls', 'Repeat length variance'),
    ]
    rows_html = []
    for key, label in stat_keys:
        val = pheno.get(key)
        if val is None:
            continue
        if isinstance(val, float):
            val_str = f'{val:.4f}'
        else:
            val_str = str(val)
        rows_html.append(
            f'<tr><td style="padding:4px 12px;color:#aaa;">{label}</td>'
            f'<td style="padding:4px 12px;">{val_str}</td></tr>'
        )
    if rows_html:
        tbl = (
            '<div class="type-section"><h2>Summary Statistics</h2>'
            '<table style="border-collapse:collapse;font-size:0.9em;">'
            + ''.join(rows_html)
            + '</table></div>'
        )
        sections.append(tbl)

    # ------------------------------------------------------------------
    # 1b. Syllable identity table
    # ------------------------------------------------------------------
    try:
        intro_labels = set(pheno.get('intro_note_labels') or [])
        repeat_counts_raw = pheno.get('repeat_counts')
        repeating_syls = set(repeat_counts_raw.columns.tolist()) if (
            repeat_counts_raw is not None and hasattr(repeat_counts_raw, 'columns')
            and not repeat_counts_raw.empty
        ) else set()

        # Determine dyad-dominated syllables: length-2 repeats > 50% of all repeats for that syllable
        dyad_syls = set()
        if repeat_counts_raw is not None and not getattr(repeat_counts_raw, 'empty', True):
            for syl in repeat_counts_raw.columns:
                col = repeat_counts_raw[syl]
                total = col.sum()
                if total > 0 and 2 in col.index:
                    if col[2] / total > 0.5:
                        dyad_syls.add(syl)

        syllable_counts_map = pheno.get('syllable_counts') or {}
        all_syls = sorted(syllable_counts_map.keys(), key=lambda s: -syllable_counts_map.get(s, 0))
        if all_syls:
            id_rows = []
            for syl in all_syls:
                tags = []
                if syl in intro_labels:
                    tags.append('<span style="background:#5b7fa6;color:#fff;padding:1px 5px;'
                                'border-radius:3px;font-size:0.8em;">intro</span>')
                if syl in dyad_syls:
                    tags.append('<span style="background:#8a6f3e;color:#fff;padding:1px 5px;'
                                'border-radius:3px;font-size:0.8em;">dyad</span>')
                elif syl in repeating_syls:
                    tags.append('<span style="background:#4a7a4a;color:#fff;padding:1px 5px;'
                                'border-radius:3px;font-size:0.8em;">repeats</span>')
                tag_str = ' '.join(tags) if tags else '<span style="color:#666;">—</span>'
                lbl_color = _css_color(syl)
                id_rows.append(
                    f'<tr>'
                    f'<td style="padding:3px 10px;text-align:center;">'
                    f'<span style="background:{lbl_color};padding:2px 7px;border-radius:3px;">{syl}</span></td>'
                    f'<td style="padding:3px 10px;text-align:right;">{syllable_counts_map.get(syl, "")}</td>'
                    f'<td style="padding:3px 10px;">{tag_str}</td>'
                    f'</tr>'
                )
            id_tbl = (
                '<table style="border-collapse:collapse;font-size:0.9em;">'
                '<tr style="border-bottom:1px solid #555;">'
                '<th style="padding:3px 10px;">Syllable</th>'
                '<th style="padding:3px 10px;">Count</th>'
                '<th style="padding:3px 10px;">Identity</th>'
                '</tr>'
                + ''.join(id_rows)
                + '</table>'
                '<p style="color:#888;font-size:0.8em;margin-top:6px;">'
                'Dyad: &gt;50% of repeats have length 2 (excluded from mean/median repeat stats). '
                'Repeats: appears in significant repeat distribution. '
                'Intro note stats reflect all songs regardless of repeat classification.</p>'
            )
            sections.append(
                f'<div class="type-section"><h2>Syllable Identity</h2>{id_tbl}</div>'
            )
    except Exception as e:
        logger.warning(f'Sequencing catalog: syllable identity table failed ({e})', exc_info=True)

    # ------------------------------------------------------------------
    # 2. Syllable proportions bar chart
    # ------------------------------------------------------------------
    syllable_counts = pheno.get('syllable_counts')
    if syllable_counts:
        try:
            labels_list = sorted(syllable_counts.keys(),
                                 key=lambda k: -syllable_counts[k])
            counts = [syllable_counts[k] for k in labels_list]
            total = sum(counts) or 1
            proportions = [c / total for c in counts]
            colors = [_label_color(lbl) for lbl in labels_list]

            fig, ax = plt.subplots(figsize=(8, max(3, len(labels_list) * 0.4)))
            bars = ax.barh(range(len(labels_list)), proportions, color=colors)
            ax.set_yticks(range(len(labels_list)))
            ax.set_yticklabels([str(l) for l in labels_list])
            ax.set_xlabel('Proportion')
            ax.set_title(f'Syllable Proportions — {bird_name} (rank {rank})')
            ax.invert_yaxis()
            fig.tight_layout()
            b64 = _fig_to_b64(fig, cfg.dpi)
            sections.append(
                f'<div class="type-section"><h2>Syllable Proportions</h2>'
                f'<img src="data:image/png;base64,{b64}"></div>'
            )
        except Exception as e:
            logger.warning(f'Sequencing catalog: proportions plot failed ({e})', exc_info=True)

    # ------------------------------------------------------------------
    # 3 & 4. Transition matrices
    # ------------------------------------------------------------------
    for matrix_key, title, cmap in [
        ('transition_counts', 'Transition Counts', 'Blues'),
        ('transition_matrix', 'Transition Probabilities (1st order)', 'viridis'),
    ]:
        matrix = pheno.get(matrix_key)
        if matrix is None or (hasattr(matrix, 'empty') and matrix.empty):
            continue
        try:
            arr = matrix.values if hasattr(matrix, 'values') else np.array(matrix)
            tick_labels = list(matrix.index) if hasattr(matrix, 'index') else []
            n = arr.shape[0]
            fig, ax = plt.subplots(figsize=(max(6, n * 0.6), max(5, n * 0.6)))
            im = ax.imshow(arr, cmap=cmap, aspect='auto')
            plt.colorbar(im, ax=ax, shrink=0.8)
            if tick_labels:
                def _fmt_tick(t, i, n_ticks):
                    if i == 0:
                        return 'START'
                    if i == n_ticks - 1:
                        return 'END'
                    return str(t)
                fmt_labels = [_fmt_tick(t, i, n) for i, t in enumerate(tick_labels)]
                ax.set_xticks(range(n))
                ax.set_xticklabels(fmt_labels, rotation=45, ha='right', fontsize=8)
                ax.set_yticks(range(n))
                ax.set_yticklabels(fmt_labels, fontsize=8)
            # Annotate cells for small matrices
            if n <= 12 and matrix_key == 'transition_counts':
                for i in range(n):
                    for j in range(n):
                        ax.text(j, i, int(arr[i, j]), ha='center', va='center',
                                fontsize=7, color='white' if arr[i, j] > arr.max() * 0.5 else 'black')
            ax.set_title(f'{title} — {bird_name} (rank {rank})')
            ax.set_xlabel('To syllable')
            ax.set_ylabel('From syllable')
            fig.tight_layout()
            b64 = _fig_to_b64(fig, cfg.dpi)
            sections.append(
                f'<div class="type-section"><h2>{title}</h2>'
                f'<img src="data:image/png;base64,{b64}"></div>'
            )
        except Exception as e:
            logger.warning(f'Sequencing catalog: {matrix_key} plot failed ({e})', exc_info=True)

    # ------------------------------------------------------------------
    # 5. Repeat counts heatmap
    # ------------------------------------------------------------------
    repeat_counts = pheno.get('repeat_counts')
    if repeat_counts is not None and hasattr(repeat_counts, 'empty') and not repeat_counts.empty:
        try:
            arr = repeat_counts.values.astype(float)   # shape: (n_repeat_lengths, n_syllables)
            syls = list(repeat_counts.columns)
            repeat_lens = list(repeat_counts.index)
            # Drop repeat-length rows that are all zero
            nonzero_rows = arr.any(axis=1)
            arr = arr[nonzero_rows]
            repeat_lens = [repeat_lens[i] for i in range(len(repeat_lens)) if nonzero_rows[i]]
            fig, ax = plt.subplots(figsize=(max(6, len(syls) * 0.6), max(3, len(repeat_lens) * 0.5)))
            im = ax.imshow(arr, cmap='Reds', aspect='auto')
            plt.colorbar(im, ax=ax, shrink=0.8)
            ax.set_xticks(range(len(syls)))
            ax.set_xticklabels([str(s) for s in syls], rotation=45, ha='right', fontsize=8)
            ax.set_yticks(range(len(repeat_lens)))
            ax.set_yticklabels([str(r) for r in repeat_lens], fontsize=8)
            ax.set_title(f'Repeat Counts — {bird_name} (rank {rank})')
            ax.set_xlabel('Syllable')
            ax.set_ylabel('Repeat length')
            fig.tight_layout()
            b64 = _fig_to_b64(fig, cfg.dpi)
            sections.append(
                f'<div class="type-section"><h2>Repeat Counts</h2>'
                f'<img src="data:image/png;base64,{b64}"></div>'
            )
        except Exception as e:
            logger.warning(f'Sequencing catalog: repeat counts plot failed ({e})', exc_info=True)

    if not sections:
        logger.warning(f'Sequencing catalog: no sections generated for {bird_name} rank {rank}')
        return ''

    n_clusters = pheno.get('repertoire_size', '?')
    html_parts = [
        _HTML_HEAD.format(
            title=f'{bird_name} — Sequencing (rank {rank})',
            generated=datetime.now().strftime('%Y-%m-%d %H:%M'),
            bird=bird_name,
            params=f'rank={rank}',
            summary=f'{n_clusters} syllable types',
        )
    ]
    html_parts.extend(sections)
    html_parts.append(_HTML_FOOT)

    out_path.write_text(''.join(html_parts), encoding='utf-8')
    logger.info(f'Sequencing catalog written: {out_path}')
    return str(out_path)


# ---------------------------------------------------------------------------
# Cluster quality catalog — helpers
# ---------------------------------------------------------------------------

# Acoustic features included in the quality catalog
_CQ_FEATURES = [
    'duration_ms',
    'spectral_centroid_mean',
    'spectral_bandwidth_mean',
    'spectral_rolloff_mean',
    'rms_energy_mean',
    'prev_syllable_gap_ms',
    'next_syllable_gap_ms',
]

_EIGENSYL_WIDTH = 64   # time bins after normalisation
_EIGENSYL_MAX_N = 100  # max spectrograms per cluster used for eigensyllable


def _time_normalise(spec: np.ndarray, target_width: int = _EIGENSYL_WIDTH) -> np.ndarray:
    """Resize *spec* to ``(n_freq, target_width)`` via bilinear interpolation."""
    n_freq, n_time = spec.shape
    if n_time == target_width:
        return spec
    return _ndimage_zoom(spec, (1.0, target_width / n_time), order=1)


def _ncc_align(spec: np.ndarray, ref: np.ndarray, max_shift_frac: float = 0.15) -> np.ndarray:
    """Shift *spec* along the time axis to maximise NCC with *ref*.

    Parameters
    ----------
    spec, ref : ndarray, shape (n_freq, n_time)
        Both must be the same shape after time-normalisation.
    max_shift_frac : float
        Maximum allowed shift as a fraction of the time width.

    Returns
    -------
    ndarray
        Shifted copy of *spec*.
    """
    n_time = spec.shape[1]
    max_shift = max(1, int(n_time * max_shift_frac))
    ref_norm = ref / (np.linalg.norm(ref) + 1e-12)

    best_shift = 0
    best_ncc = -np.inf
    for shift in range(-max_shift, max_shift + 1):
        rolled = np.roll(spec, shift, axis=1)
        ncc = np.sum(ref_norm * rolled) / (np.linalg.norm(rolled) + 1e-12)
        if ncc > best_ncc:
            best_ncc = ncc
            best_shift = shift
    return np.roll(spec, best_shift, axis=1)


def _compute_eigensyllable(
        specs: List[np.ndarray],
        max_n: int = _EIGENSYL_MAX_N,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (mean, std) eigensyllable for a collection of spectrograms.

    Steps:
    1. Cap at *max_n* instances.
    2. Time-normalise all to :data:`_EIGENSYL_WIDTH` bins.
    3. Find medoid (min mean MSE to all others).
    4. NCC-align all to medoid (±15 % shift).
    5. Return pixelwise mean and std.

    Parameters
    ----------
    specs : list of ndarray, shape (n_freq, n_time)
        Raw (unnormalised) syllable spectrograms.
    max_n : int
        Maximum number of instances to use.

    Returns
    -------
    mean_spec, std_spec : ndarray, shape (n_freq, n_time_norm)
    """
    if not specs:
        raise ValueError('No spectrograms provided')
    sample = specs[:max_n]

    # Time-normalise
    normed = np.stack([_time_normalise(s) for s in sample])  # (n, freq, width)

    # Find medoid via MSE
    n = len(normed)
    if n == 1:
        medoid_idx = 0
    else:
        flat = normed.reshape(n, -1)
        dists = np.sum(
            (flat[:, np.newaxis, :] - flat[np.newaxis, :, :]) ** 2,
            axis=-1,
        )  # (n, n)
        np.fill_diagonal(dists, 0.0)
        medoid_idx = int(np.argmin(dists.sum(axis=1)))

    ref = normed[medoid_idx]

    # NCC-align all to medoid
    aligned = np.stack([_ncc_align(s, ref) for s in normed])

    mean_spec = aligned.mean(axis=0)
    std_spec  = aligned.std(axis=0)
    return mean_spec, std_spec


def _add_repeat_position(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    """Add a ``position_in_repeat`` column to *df*.

    Within each song (identified by ``song_file`` + ``position_in_song``),
    consecutive rows sharing the same label form a "run".  The first
    occurrence in a run gets position 1, the second position 2, etc.
    Rows with label ``-1`` (noise) always get position ``NaN``.

    Parameters
    ----------
    df : DataFrame
        Must contain ``song_file``, ``position_in_song``, and *label_col*.
    label_col : str
        Column with cluster labels.

    Returns
    -------
    DataFrame
        A copy of *df* with an added ``position_in_repeat`` column.
    """
    df = df.sort_values(['song_file', 'position_in_song']).copy()
    pos_in_rep = np.full(len(df), np.nan)

    for _, song_df in df.groupby('song_file', sort=False):
        idxs = song_df.index.tolist()
        labels = song_df[label_col].tolist()
        run_pos = 1
        for k, idx in enumerate(idxs):
            lbl = labels[k]
            if pd.isna(lbl) or lbl == -1:
                run_pos = 1
                continue
            if k > 0 and labels[k - 1] == lbl:
                pos_in_rep[df.index.get_loc(idx)] = run_pos
                run_pos += 1
            else:
                run_pos = 1
                pos_in_rep[df.index.get_loc(idx)] = run_pos
                run_pos += 1

    df['position_in_repeat'] = pos_in_rep
    return df


def _render_eigensyllable(mean_spec: np.ndarray, std_spec: np.ndarray,
                          label: Any, dpi: int) -> str:
    """Return a base64 PNG showing mean + std eigensyllable side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(6, 2))
    vmin = mean_spec.min()
    vmax = mean_spec.max()
    axes[0].imshow(mean_spec, aspect='auto', origin='lower', cmap='plasma',
                   vmin=vmin, vmax=vmax)
    axes[0].set_title(f'Eigensyllable (label {label})', fontsize=8)
    axes[0].axis('off')
    axes[1].imshow(std_spec, aspect='auto', origin='lower', cmap='viridis')
    axes[1].set_title('Std deviation map', fontsize=8)
    axes[1].axis('off')
    fig.tight_layout(pad=0.5)
    return _fig_to_b64(fig, dpi)


# ---------------------------------------------------------------------------
# Cluster quality catalog — main function
# ---------------------------------------------------------------------------

def generate_cluster_quality_catalog(
    bird_path: str,
    rank: int = 0,
    config: Optional[CatalogConfig] = None,
) -> str:
    """Generate an HTML syllable characteristics catalog for one bird.

    Presents per-cluster acoustic feature statistics, spectrogram thumbnails,
    eigensyllables, discriminative feature distributions, transition outflows,
    and (for repeat clusters) a ramping analysis.

    Parameters
    ----------
    bird_path : str
        Path to the bird's root directory (the run-scoped path, same as
        passed to :func:`generate_all_catalogs`).
    rank : int, optional
        Clustering rank to use.  Default is ``0``.
    config : CatalogConfig or None, optional
        Display parameters.  Uses default :class:`CatalogConfig` when ``None``.

    Returns
    -------
    str
        Absolute path to the generated HTML file, or ``''`` on failure.
    """
    cfg = config or CatalogConfig()
    from song_phenotyping.tools.pipeline_paths import (
        CATALOG_DIR, SPECS_DIR, STAGES_DIR, PHENOTYPE_DIR,
    )
    bird_path = Path(bird_path)
    bird_name = bird_path.name
    out_dir = bird_path / CATALOG_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'{bird_name}_syllable_characteristics_rank{rank}.html'

    if out_path.exists() and not cfg.overwrite:
        return str(out_path)

    # ------------------------------------------------------------------
    # Load phenotype pickle (for transition data and intro notes)
    # ------------------------------------------------------------------
    _pheno_data = {}
    _pheno_intro_labels = set()
    _pheno_transition_matrix = None
    _pheno_transition_counts = None
    _pheno_repeat_counts = None
    _pheno_repeating_syls = set()
    _pheno_dyad_syls = set()
    try:
        pkl_path = bird_path / PHENOTYPE_DIR / f'automated_phenotype_data_rank{rank}.pkl'
        if pkl_path.exists():
            _raw = pd.read_pickle(str(pkl_path))
            _pheno_data = _raw.get('phenotype_results', _raw)
            _pheno_intro_labels = set(_pheno_data.get('intro_note_labels') or [])
            _pheno_transition_matrix = _pheno_data.get('transition_matrix')
            _pheno_transition_counts = _pheno_data.get('transition_counts')
            _pheno_repeat_counts = _pheno_data.get('repeat_counts')
            if _pheno_repeat_counts is not None and not getattr(_pheno_repeat_counts, 'empty', True):
                _pheno_repeating_syls = set(_pheno_repeat_counts.columns.tolist())
                for _syl in _pheno_repeat_counts.columns:
                    _col = _pheno_repeat_counts[_syl]
                    _tot = _col.sum()
                    if _tot > 0 and 2 in _col.index and _col[2] / _tot > 0.5:
                        _pheno_dyad_syls.add(_syl)
    except Exception as _e:
        logger.warning(f'Syllable characteristics catalog: could not load pkl ({_e})')

    # ------------------------------------------------------------------
    # Load syllable_features.csv
    # ------------------------------------------------------------------
    csv_path = bird_path / STAGES_DIR / 'syllable_database' / 'syllable_features.csv'
    if not csv_path.exists():
        logger.warning(f'Syllable characteristics catalog: syllable_features.csv not found: {csv_path}')
        return ''

    try:
        df_all = pd.read_csv(csv_path)
    except Exception as e:
        logger.warning(f'Syllable characteristics catalog: could not load CSV ({e})')
        return ''

    # Find cluster label column for this rank
    prefix = f'cluster_rank{rank}_'
    label_cols = [c for c in df_all.columns if c.startswith(prefix)]
    if not label_cols:
        logger.warning(f'Syllable characteristics catalog: no cluster_rank{rank}_* column found in CSV')
        return ''
    label_col = label_cols[0]

    # Present features available in this CSV
    features = [f for f in _CQ_FEATURES if f in df_all.columns]
    if not features:
        logger.warning('Syllable characteristics catalog: none of the expected feature columns found')
        return ''

    # Identify unique cluster labels (include -1 noise)
    df_all[label_col] = df_all[label_col].fillna(-1).astype(int)
    cluster_labels = sorted(df_all[label_col].unique())

    # ------------------------------------------------------------------
    # Load spectrograms grouped by cluster label
    # ------------------------------------------------------------------
    specs_dir = bird_path / SPECS_DIR
    label_to_specs: Dict[int, List[np.ndarray]] = {lbl: [] for lbl in cluster_labels}
    hash_to_label: Dict[str, int] = {}
    if 'hash_id' in df_all.columns:
        hash_to_label = dict(zip(df_all['hash_id'].astype(str), df_all[label_col]))

    for syl_file in sorted(specs_dir.glob('syllables_*.h5')) if specs_dir.exists() else []:
        try:
            with tables.open_file(str(syl_file), 'r') as hf:
                specs_arr = hf.root.spectrograms.read()   # (n, freq, time)
                if hasattr(hf.root, 'hashes'):
                    hashes = [
                        h.decode('utf-8') if isinstance(h, bytes) else str(h)
                        for h in hf.root.hashes.read()
                    ]
                else:
                    hashes = []
            for i, spec in enumerate(specs_arr):
                if i < len(hashes) and hashes[i] in hash_to_label:
                    lbl = hash_to_label[hashes[i]]
                    if lbl in label_to_specs:
                        label_to_specs[lbl].append(spec)
        except Exception as e:
            logger.warning(f'Cluster quality catalog: error loading {syl_file.name}: {e}')

    sections = []
    n_with_specs = sum(1 for v in label_to_specs.values() if v)

    # ------------------------------------------------------------------
    # Between-cluster: summary table
    # ------------------------------------------------------------------
    try:
        summary_rows = []
        for lbl in cluster_labels:
            grp = df_all[df_all[label_col] == lbl]
            row = {'label': lbl, 'n': len(grp)}
            for feat in features:
                vals = grp[feat].dropna()
                row[f'{feat}_mean'] = vals.mean() if len(vals) else np.nan
                row[f'{feat}_std']  = vals.std()  if len(vals) else np.nan
            summary_rows.append(row)
        summary_df = pd.DataFrame(summary_rows)

        # HTML table
        header_cells = '<th>Label</th><th>N</th>' + ''.join(
            f'<th>{f}<br><span style="font-weight:normal;color:#aaa;">mean±std</span></th>'
            for f in features
        )
        data_rows = []
        for _, r in summary_df.iterrows():
            cells = (
                f'<td style="text-align:center;">'
                f'<span style="background:{_css_color(r["label"])};padding:2px 6px;'
                f'border-radius:3px;">{int(r["label"])}</span></td>'
                f'<td style="text-align:right;">{int(r["n"])}</td>'
            )
            for feat in features:
                m = r.get(f'{feat}_mean', np.nan)
                s = r.get(f'{feat}_std',  np.nan)
                cells += (
                    f'<td style="text-align:right;">'
                    f'{"N/A" if np.isnan(m) else f"{m:.2f}"}'
                    f'&nbsp;<span style="color:#888;">±&nbsp;'
                    f'{"N/A" if np.isnan(s) else f"{s:.2f}"}</span></td>'
                )
            data_rows.append(f'<tr>{cells}</tr>')

        tbl_html = (
            '<table style="border-collapse:collapse;font-size:0.85em;'
            'white-space:nowrap;">'
            f'<tr style="border-bottom:2px solid #444;">{header_cells}</tr>'
            + ''.join(data_rows)
            + '</table>'
        )
        sections.append(
            f'<div class="type-section"><h2>Cluster Summary</h2>{tbl_html}</div>'
        )
    except Exception as e:
        logger.warning(f'Syllable characteristics catalog: summary table failed ({e})', exc_info=True)

    # ------------------------------------------------------------------
    # Between-cluster: z-scored feature profile heatmap
    # ------------------------------------------------------------------
    z_matrix = np.zeros((len(cluster_labels), len(features)))  # fallback for violin plots
    feat_matrix = np.zeros((len(cluster_labels), len(features)))
    try:
        feat_matrix = summary_df[[f'{f}_mean' for f in features]].values.astype(float)
        col_means = np.nanmean(feat_matrix, axis=0)
        col_stds  = np.nanstd(feat_matrix,  axis=0)
        col_stds[col_stds == 0] = 1.0
        z_matrix = (feat_matrix - col_means) / col_stds

        fig, ax = plt.subplots(figsize=(max(6, len(features) * 1.1),
                                        max(3, len(cluster_labels) * 0.5)))
        im = ax.imshow(z_matrix, cmap='RdBu_r', aspect='auto',
                       vmin=-2.5, vmax=2.5)
        plt.colorbar(im, ax=ax, shrink=0.8, label='z-score')
        ax.set_xticks(range(len(features)))
        ax.set_xticklabels(features, rotation=40, ha='right', fontsize=8)
        ax.set_yticks(range(len(cluster_labels)))
        ax.set_yticklabels([str(l) for l in cluster_labels], fontsize=8)
        ax.set_title(f'Feature Profile (z-scored) — {bird_name} rank {rank}')
        fig.tight_layout()
        b64 = _fig_to_b64(fig, cfg.dpi)
        sections.append(
            f'<div class="type-section"><h2>Feature Profile Heatmap</h2>'
            f'<p style="color:#aaa;font-size:0.85em;margin-top:0;">'
            f'Each cell shows how a syllable type\'s mean feature value compares to the average '
            f'across all syllable types. <span style="color:#d46060;">Red = higher than average</span>, '
            f'<span style="color:#6090d4;">blue = lower than average</span>. '
            f'|z| &gt; 1.5 indicates a feature that meaningfully distinguishes that syllable type.</p>'
            f'<img src="data:image/png;base64,{b64}"></div>'
        )
    except Exception as e:
        logger.warning(f'Syllable characteristics catalog: feature heatmap failed ({e})', exc_info=True)

    # ------------------------------------------------------------------
    # Between-cluster: pairwise distance matrix
    # ------------------------------------------------------------------
    try:
        valid_rows = ~np.any(np.isnan(feat_matrix), axis=1)
        if valid_rows.sum() >= 2:
            valid_matrix = feat_matrix[valid_rows]
            valid_labels = [cluster_labels[i] for i in range(len(cluster_labels)) if valid_rows[i]]
            n_cl = len(valid_labels)
            dist_mat = np.zeros((n_cl, n_cl))
            for i in range(n_cl):
                for j in range(n_cl):
                    dist_mat[i, j] = np.linalg.norm(valid_matrix[i] - valid_matrix[j])

            fig, ax = plt.subplots(figsize=(max(4, n_cl * 0.6), max(4, n_cl * 0.6)))
            im = ax.imshow(dist_mat, cmap='viridis_r', aspect='equal')
            plt.colorbar(im, ax=ax, shrink=0.8, label='Euclidean dist')
            ax.set_xticks(range(n_cl))
            ax.set_xticklabels([str(l) for l in valid_labels], rotation=45, ha='right', fontsize=8)
            ax.set_yticks(range(n_cl))
            ax.set_yticklabels([str(l) for l in valid_labels], fontsize=8)
            ax.set_title(f'Pairwise Cluster Distance — {bird_name} rank {rank}')
            fig.tight_layout()
            b64 = _fig_to_b64(fig, cfg.dpi)
            sections.append(
                f'<div class="type-section"><h2>Pairwise Cluster Distance</h2>'
                f'<img src="data:image/png;base64,{b64}"></div>'
            )
    except Exception as e:
        logger.warning(f'Syllable characteristics catalog: distance matrix failed ({e})', exc_info=True)

    # ------------------------------------------------------------------
    # Add repeat position column (needed for ramping analysis)
    # ------------------------------------------------------------------
    if 'song_file' in df_all.columns and 'position_in_song' in df_all.columns:
        try:
            df_all = _add_repeat_position(df_all, label_col)
        except Exception as e:
            logger.warning(f'Syllable characteristics catalog: repeat-position annotation failed ({e})')

    # ------------------------------------------------------------------
    # Per-cluster sections
    # ------------------------------------------------------------------
    for lbl in cluster_labels:
        grp = df_all[df_all[label_col] == lbl]
        lbl_specs = label_to_specs.get(lbl, [])
        lbl_color = _css_color(lbl)       # CSS string — use only in HTML
        lbl_mpl   = _label_color(lbl)     # normalized (r,g,b) tuple — use in matplotlib

        identity_tags = []
        if lbl in _pheno_intro_labels:
            identity_tags.append(
                '<span style="background:#5b7fa6;color:#fff;padding:2px 8px;'
                'border-radius:3px;font-size:0.85em;margin-left:8px;">intro note</span>'
            )
        if lbl in _pheno_dyad_syls:
            identity_tags.append(
                '<span style="background:#8a6f3e;color:#fff;padding:2px 8px;'
                'border-radius:3px;font-size:0.85em;margin-left:8px;">dyad</span>'
            )
        elif lbl in _pheno_repeating_syls:
            identity_tags.append(
                '<span style="background:#4a7a4a;color:#fff;padding:2px 8px;'
                'border-radius:3px;font-size:0.85em;margin-left:8px;">repeats</span>'
            )
        identity_str = ''.join(identity_tags)
        header = (
            f'<div class="type-section">'
            f'<h2 style="color:#fff;background:{lbl_color};'
            f'display:inline-block;padding:4px 12px;border-radius:4px;">'
            f'Cluster {lbl} &nbsp; <span style="font-weight:normal;font-size:0.85em;">'
            f'(n={len(grp)})</span></h2>'
            f'{identity_str}'
        )
        body_parts = [header]

        # — CV bar chart —
        try:
            cv_data = {}
            for feat in features:
                vals = grp[feat].dropna()
                if len(vals) > 1 and vals.mean() != 0:
                    cv_data[feat] = vals.std() / abs(vals.mean())
            if cv_data:
                fig, ax = plt.subplots(figsize=(6, 2.5))
                ax.bar(range(len(cv_data)), list(cv_data.values()),
                       color=[_label_color(lbl)] * len(cv_data))
                ax.set_xticks(range(len(cv_data)))
                ax.set_xticklabels(list(cv_data.keys()), rotation=30, ha='right', fontsize=8)
                ax.set_ylabel('CV')
                ax.set_title(f'Feature CV — cluster {lbl}', fontsize=9)
                fig.tight_layout()
                b64 = _fig_to_b64(fig, cfg.dpi)
                body_parts.append(
                    f'<img src="data:image/png;base64,{b64}" '
                    f'style="display:inline-block;vertical-align:top;">'
                )
        except Exception as e:
            logger.warning(f'Syllable characteristics: CV chart failed for {lbl} ({e})')

        # — Feature boxplots —
        try:
            feat_vals = {f: grp[f].dropna().values for f in features if f in grp.columns}
            if feat_vals:
                fig, ax = plt.subplots(figsize=(7, 2.8))
                positions = list(range(len(feat_vals)))
                bp = ax.boxplot(
                    list(feat_vals.values()), positions=positions,
                    vert=True, patch_artist=True, widths=0.6,
                    medianprops={'color': 'white'},
                    boxprops={'facecolor': lbl_mpl, 'alpha': 0.7},
                )
                ax.set_xticks(positions)
                ax.set_xticklabels(list(feat_vals.keys()), rotation=30, ha='right', fontsize=8)
                ax.set_title(f'Feature distributions — cluster {lbl}', fontsize=9)
                fig.tight_layout()
                b64 = _fig_to_b64(fig, cfg.dpi)
                body_parts.append(
                    f'<img src="data:image/png;base64,{b64}" '
                    f'style="display:inline-block;vertical-align:top;">'
                )
        except Exception as e:
            logger.warning(f'Syllable characteristics: boxplots failed for {lbl} ({e})')

        # — Violin plots for most discriminative features —
        try:
            # Identify top-3 features by |z-score| for this cluster in the feature profile
            lbl_idx = cluster_labels.index(lbl)
            if lbl_idx < len(z_matrix):
                z_row = z_matrix[lbl_idx]
                top_feat_idxs = np.argsort(np.abs(z_row))[::-1][:3]
                top_feats = [features[i] for i in top_feat_idxs if i < len(features)]
            else:
                top_feats = features[:3]
            violin_data = {}
            for feat in top_feats:
                all_other = df_all[df_all[label_col] != lbl][feat].dropna().values
                this_vals = grp[feat].dropna().values
                if len(this_vals) >= 3:
                    violin_data[feat] = (this_vals, all_other)
            if violin_data:
                fig, axes = plt.subplots(1, len(violin_data),
                                         figsize=(4 * len(violin_data), 3.2), squeeze=False)
                for ax_v, (feat, (this_v, other_v)) in zip(axes[0], violin_data.items()):
                    parts = ax_v.violinplot(
                        [other_v, this_v] if len(other_v) >= 3 else [this_v],
                        showmedians=True,
                    )
                    colors_v = ['#555', lbl_mpl] if len(other_v) >= 3 else [lbl_mpl]
                    for pc, col in zip(parts['bodies'], colors_v):
                        pc.set_facecolor(col)
                        pc.set_alpha(0.75)
                    xlabels = ['others', 'this'] if len(other_v) >= 3 else ['this']
                    ax_v.set_xticks([1, 2] if len(xlabels) == 2 else [1])
                    ax_v.set_xticklabels(xlabels, fontsize=8)
                    z_val = z_row[features.index(feat)] if feat in features else 0
                    ax_v.set_title(f'{feat}\n(z={z_val:.2f})', fontsize=8)
                    ax_v.set_facecolor('#222')
                fig.suptitle(f'Discriminative features — cluster {lbl}', fontsize=9)
                fig.tight_layout()
                b64 = _fig_to_b64(fig, cfg.dpi)
                body_parts.append(
                    f'<div><h3 style="font-size:0.9em;color:#aaa;">Most Discriminative Features</h3>'
                    f'<img src="data:image/png;base64,{b64}"></div>'
                )
        except Exception as e:
            logger.warning(f'Syllable characteristics: violin plots failed for {lbl} ({e})')

        # — Outgoing transition statistics —
        try:
            if _pheno_transition_matrix is not None and not _pheno_transition_matrix.empty:
                tm = _pheno_transition_matrix
                if lbl in tm.index:
                    # Outgoing: row for this label, normalized across columns
                    out_probs = tm.loc[lbl]
                    # Rename START/END tokens
                    def _fmt_syl(s, idxs):
                        if s == idxs[0]:
                            return 'START'
                        if s == idxs[-1]:
                            return 'END'
                        return str(s)
                    all_idx = list(tm.index)
                    out_labels = [_fmt_syl(s, all_idx) for s in out_probs.index]
                    out_vals = out_probs.values.astype(float)
                    mask = out_vals > 0.01  # hide near-zero transitions
                    if mask.sum() >= 1:
                        fig, ax = plt.subplots(figsize=(max(4, mask.sum() * 0.7 + 1), 2.5))
                        x_pos = np.arange(mask.sum())
                        ax.bar(x_pos, out_vals[mask],
                               color=[lbl_mpl if out_labels[i] not in ('START', 'END')
                                      else '#777' for i in range(len(out_labels)) if mask[i]])
                        ax.set_xticks(x_pos)
                        ax.set_xticklabels([l for l, m in zip(out_labels, mask) if m],
                                           rotation=30, ha='right', fontsize=9)
                        ax.set_ylabel('Transition prob.', fontsize=8)
                        ax.set_title(f'Outgoing transitions — cluster {lbl}', fontsize=9)
                        ax.set_ylim(0, 1.05)
                        ax.set_facecolor('#222')
                        fig.tight_layout()
                        b64 = _fig_to_b64(fig, cfg.dpi)
                        body_parts.append(
                            f'<div><h3 style="font-size:0.9em;color:#aaa;">Outgoing Transitions</h3>'
                            f'<p style="color:#888;font-size:0.8em;margin:0 0 4px;">Where does this syllable '
                            f'tend to go next? (Transitions &lt;1% omitted)</p>'
                            f'<img src="data:image/png;base64,{b64}"></div>'
                        )
        except Exception as e:
            logger.warning(f'Syllable characteristics: transition chart failed for {lbl} ({e})')

        # — Spectrogram thumbnails —
        try:
            n_show = min(cfg.n_per_type, len(lbl_specs))
            if n_show > 0:
                b64 = _render_syllable_grid(
                    specs=lbl_specs[:n_show],
                    label=lbl,
                    label_source='auto',
                    n_cols=cfg.grid_cols,
                    syl_fig_size=cfg.syl_fig_size,
                    dpi=cfg.dpi,
                )
                body_parts.append(
                    f'<div><h3 style="font-size:0.9em;color:#aaa;">Thumbnails ({n_show} shown)</h3>'
                    f'<img src="data:image/png;base64,{b64}"></div>'
                )
        except Exception as e:
            logger.warning(f'Syllable characteristics: thumbnails failed for {lbl} ({e})')

        # — Eigensyllable —
        try:
            if len(lbl_specs) >= 3:
                mean_spec, std_spec = _compute_eigensyllable(lbl_specs)
                b64 = _render_eigensyllable(mean_spec, std_spec, lbl, cfg.dpi)
                body_parts.append(
                    f'<div><h3 style="font-size:0.9em;color:#aaa;">Eigensyllable</h3>'
                    f'<img src="data:image/png;base64,{b64}"></div>'
                )
        except Exception as e:
            logger.warning(f'Syllable characteristics: eigensyllable failed for {lbl} ({e})')

        # — Ramping analysis (repeat clusters only) —
        try:
            if 'position_in_repeat' in df_all.columns and len(grp) > 5:
                rep_df = grp.dropna(subset=['position_in_repeat'])
                max_rep = rep_df['position_in_repeat'].max() if not rep_df.empty else 1
                if max_rep >= 3:
                    ramp_features = [f for f in ('rms_energy_mean', 'spectral_centroid_mean')
                                     if f in rep_df.columns]
                    if ramp_features:
                        fig, axes = plt.subplots(1, len(ramp_features),
                                                 figsize=(4 * len(ramp_features), 3),
                                                 squeeze=False)
                        for ax, rf in zip(axes[0], ramp_features):
                            pos_grp = rep_df.groupby('position_in_repeat')[rf].mean()
                            x = pos_grp.index.values.astype(float)
                            y = pos_grp.values
                            if len(x) >= 3:
                                slope, intercept, r, *_ = _linregress(x, y)
                                ax.scatter(x, y, color=lbl_mpl, s=30, zorder=3)
                                ax.plot(x, slope * x + intercept, 'w--', linewidth=1)
                                ax.set_title(
                                    f'{rf}\nslope={slope:.3g}', fontsize=8
                                )
                            else:
                                ax.scatter(x, y, color=lbl_mpl, s=30)
                                ax.set_title(rf, fontsize=8)
                            ax.set_xlabel('Position in repeat', fontsize=8)
                            ax.set_facecolor('#222')
                        fig.suptitle(f'Ramping — cluster {lbl}', fontsize=9)
                        fig.tight_layout()
                        b64 = _fig_to_b64(fig, cfg.dpi)
                        body_parts.append(
                            f'<div><h3 style="font-size:0.9em;color:#aaa;">Ramping Analysis</h3>'
                            f'<img src="data:image/png;base64,{b64}"></div>'
                        )
        except Exception as e:
            logger.warning(f'Syllable characteristics: ramping failed for {lbl} ({e})')

        body_parts.append('</div>')  # close type-section
        sections.extend(body_parts)

    if not sections:
        logger.warning(f'Syllable characteristics catalog: no sections generated for {bird_name} rank {rank}')
        return ''

    n_clusters = len(cluster_labels)
    html_parts = [
        _HTML_HEAD.format(
            title=f'{bird_name} — Syllable Characteristics (rank {rank})',
            generated=datetime.now().strftime('%Y-%m-%d %H:%M'),
            bird=bird_name,
            params=f'rank={rank}',
            summary=f'{n_clusters} syllable types &nbsp;|&nbsp; features: {", ".join(features)}',
        )
    ]
    html_parts.extend(sections)
    html_parts.append(_HTML_FOOT)

    out_path.write_text(''.join(html_parts), encoding='utf-8')
    logger.info(f'Syllable characteristics catalog written: {out_path}')
    return str(out_path)


def _css_color(label: Any) -> str:
    """Return an inline CSS colour string for a cluster label."""
    rgb = _label_color(label)
    return 'rgba({},{},{},0.85)'.format(
        int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)
    )


# ---------------------------------------------------------------------------
# Convenience: generate both catalogs for a bird
# ---------------------------------------------------------------------------

def generate_all_catalogs(
    bird_path: str,
    rank: int = 0,
    config: Optional[CatalogConfig] = None,
) -> Dict[str, str]:
    """Generate the full suite of HTML catalogs for one bird.

    Convenience wrapper that calls :func:`generate_song_catalog`,
    :func:`generate_syllable_type_catalog` (auto labels), and — when manual
    labels are present — :func:`generate_syllable_type_catalog` (manual
    labels).

    Parameters
    ----------
    bird_path : str
        Path to the bird's root directory.
    rank : int, optional
        Clustering rank passed to each catalog generator.  Default is ``0``.
    config : CatalogConfig or None, optional
        Shared display parameters.  Uses default :class:`CatalogConfig` when
        ``None``.

    Returns
    -------
    dict of str → str
        Maps catalog type to the absolute HTML output path.  Keys may include
        ``'song_catalog'``, ``'syllable_types_auto'``, ``'syllable_types_manual'``,
        ``'sequencing'``, and ``'syllable_characteristics'``.
        Only successfully generated catalogs are included.
    """
    cfg = config or CatalogConfig()
    results = {}

    path = generate_song_catalog(bird_path, rank=rank, config=cfg)
    if path:
        results['song_catalog'] = path

    path = generate_syllable_type_catalog(
        bird_path, rank=rank, label_source='auto', config=cfg
    )
    if path:
        results['syllable_types_auto'] = path

    # Manual labels only if they exist
    from song_phenotyping.tools.pipeline_paths import SPECS_DIR
    specs_dir = Path(bird_path) / SPECS_DIR
    has_manual = False
    for f in specs_dir.glob('syllables_*.h5') if specs_dir.exists() else []:
        try:
            with tables.open_file(str(f), 'r') as hf:
                if hasattr(hf.root, 'manual') and len(hf.root.manual.read()) > 0:
                    has_manual = True
                    break
        except Exception:
            continue

    if has_manual:
        path = generate_syllable_type_catalog(
            bird_path, rank=rank, label_source='manual', config=cfg
        )
        if path:
            results['syllable_types_manual'] = path

    path = generate_sequencing_catalog(bird_path, rank=rank, config=cfg)
    if path:
        results['sequencing'] = path

    path = generate_cluster_quality_catalog(bird_path, rank=rank, config=cfg)
    if path:
        results['syllable_characteristics'] = path

    return results


if __name__ == '__main__':
    import logging as _logging
    _logging.basicConfig(level=_logging.INFO,
                         format='%(levelname)s %(name)s: %(message)s')

    from song_phenotyping.tools.project_config import ProjectConfig
    cfg_proj = ProjectConfig.load()

    # Edit these to point at a real bird directory:
    bird_path = str(cfg_proj.bird_dir('or18or24', experiment='evsong test'))

    results = generate_all_catalogs(bird_path, rank=0)
    for catalog_type, html_path in results.items():
        print(f'{catalog_type}: {html_path}')
