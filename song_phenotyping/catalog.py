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
- :func:`generate_all_catalogs` — convenience wrapper that calls both.
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

    used_manual, used_auto = set(), set()

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
                used_manual.add(str(lbl))
        if auto_labels is not None and i < len(auto_labels):
            lbl = auto_labels[i]
            if int(lbl) >= 0:
                c = _label_color(lbl)
                ax.text(cx, -4, str(lbl), fontsize=8,
                        ha='center', va='top', color='black',
                        bbox=dict(facecolor=c, edgecolor='none', alpha=0.85,
                                  boxstyle='round,pad=0.15'))
                used_auto.add(str(lbl))

    # Legend patches
    legend_handles = []
    if used_manual:
        legend_handles += [
            mpatches.Patch(color=_label_color(l), label=f'M:{l}')
            for l in sorted(used_manual)
        ]
    if used_auto:
        legend_handles += [
            mpatches.Patch(color=_label_color(l), label=f'A:{l}')
            for l in sorted(used_auto, key=lambda x: int(x))
        ]
    if legend_handles:
        ax.legend(handles=legend_handles, loc='upper right',
                  fontsize=7, ncol=min(len(legend_handles), 8),
                  framealpha=0.7, handlelength=0.8)

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
            logger.error(f'Error rendering {syl_file.name}: {e}')
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
            logger.error(f'Error loading {syl_file.name}: {e}')
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
        Maps catalog type (``'song_catalog'``, ``'syllable_types_auto'``,
        ``'syllable_types_manual'``) to the absolute HTML output path.
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
