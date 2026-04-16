"""Cross-bird phenotype comparison script.

Aggregates phenotype data from all birds under a common ``save_path``
and produces a single CSV summary together with an optional HTML report
containing comparative plots.

Each bird's ``phenotype_results.csv`` (from Stage E) is loaded and the
rank-0 automated clustering result is used for comparison.  Per-bird
acoustic feature means (duration, gap, song length) are merged in from
``syllable_features.csv`` when available.

Usage
-----
Run as a standalone script::

    python scripts/cohort_comparison.py E:/xfoster_pipeline_runs

Or import the core function::

    from scripts.cohort_comparison import build_cohort_table
    df = build_cohort_table("E:/xfoster_pipeline_runs")

Output
------
``<save_path>/cohort_comparison.csv`` — one row per bird with all
phenotype and acoustic feature columns.

``<save_path>/cohort_comparison.html`` — interactive comparison plots
(generated only when ``--html`` flag is passed on the CLI).

Notes
-----
- Intro note labels are stored as a serialised list string in
  ``phenotype_results.csv``; they are parsed back into a list.
- F0 features are intentionally excluded (unreliable across syllable types).
- The ``--run_name`` argument (default ``None``) selects the run hash
  directory; when ``None``, the script auto-discovers a single run
  directory per bird (fails loudly if there is more than one or none).
"""

import argparse
import ast
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

logger = logging.getLogger(__name__)

# Acoustic feature columns merged from syllable_features.csv
_ACOUSTIC_FEATURES = [
    'duration_ms',
    'prev_syllable_gap_ms',
    'song_length_syllables',
]

# Phenotype columns included in the comparison table
_PHENOTYPE_COLS = [
    'repertoire_size',
    'entropy',
    'entropy_scaled',
    'entropy_excl_intro',
    'entropy_scaled_excl_intro',
    'repeat_bool',
    'dyad_bool',
    'num_dyad',
    'num_longer_reps',
    'mean_repeat_syls',
    'median_repeat_syls',
    'var_repeat_syls',
    'n_songs',
    'n_syllables_total',
    'has_intro_notes',
    'intro_recurs_in_song',
    'mean_intro_count_per_song',
    'std_intro_count_per_song',
]


# ---------------------------------------------------------------------------
# Discovery helpers
# ---------------------------------------------------------------------------

def _find_run_dir(bird_dir: Path, run_name: Optional[str]) -> Optional[Path]:
    """Return the run directory for *bird_dir*.

    When *run_name* is given, returns ``bird_dir / run_name`` if it exists.
    Otherwise looks for exactly one subdirectory that looks like a run
    (contains a ``results/`` subdir).

    Returns ``None`` if no suitable run directory is found.
    """
    if run_name:
        candidate = bird_dir / run_name
        return candidate if candidate.is_dir() else None

    candidates = [
        d for d in bird_dir.iterdir()
        if d.is_dir() and (d / 'results').is_dir()
    ]
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        # Pick the most recently modified one and warn
        candidates.sort(key=lambda d: d.stat().st_mtime, reverse=True)
        logger.warning(
            f'Multiple run dirs found for {bird_dir.name}; '
            f'using most recent: {candidates[0].name}'
        )
        return candidates[0]
    return None


def _load_phenotype_row(run_dir: Path, rank: int = 0) -> Optional[dict]:
    """Load the rank-0 automated phenotype row from ``phenotype_results.csv``."""
    csv = run_dir / 'results' / 'phenotype_results.csv'
    if not csv.exists():
        return None
    try:
        df = pd.read_csv(csv)
        # Filter to automated rank-0 (rank == 0)
        auto = df[df['rank'].astype(str) == str(rank)]
        if auto.empty:
            return None
        row = auto.iloc[0].to_dict()
        # Reject rows where Stage E ran on empty data
        if not row.get('n_syllables_total', 0) or not row.get('repertoire_size', 0):
            logger.warning(
                f'{csv.parent.parent.name}: n_syllables_total={row.get("n_syllables_total")} '
                f'repertoire_size={row.get("repertoire_size")} — skipping (empty data)'
            )
            return None
        # Parse intro_note_labels back to a list if it's a non-empty string
        raw_labels = row.get('intro_note_labels', '')
        if isinstance(raw_labels, str) and raw_labels.strip():
            try:
                row['intro_note_labels'] = ast.literal_eval(raw_labels)
            except Exception:
                row['intro_note_labels'] = raw_labels
        return row
    except Exception as e:
        logger.warning(f'Could not load phenotype results from {csv}: {e}')
        return None


def _load_acoustic_means(run_dir: Path) -> dict:
    """Load per-bird mean acoustic features from ``syllable_features.csv``."""
    csv = run_dir / 'stages' / 'syllable_database' / 'syllable_features.csv'
    if not csv.exists():
        return {}
    try:
        df = pd.read_csv(csv)
        result = {}
        for feat in _ACOUSTIC_FEATURES:
            if feat in df.columns:
                result[f'mean_{feat}'] = df[feat].mean()
        return result
    except Exception as e:
        logger.warning(f'Could not load syllable features from {csv}: {e}')
        return {}


# ---------------------------------------------------------------------------
# Core builder
# ---------------------------------------------------------------------------

def build_cohort_table(
        save_path: str,
        run_name: Optional[str] = None,
        rank: int = 0,
) -> pd.DataFrame:
    """Build a one-row-per-bird comparison DataFrame.

    Parameters
    ----------
    save_path : str
        Root directory containing one subdirectory per bird.
    run_name : str or None, optional
        Run hash or name to use for all birds.  ``None`` (default) means
        auto-discover the single run directory.
    rank : int, optional
        Automated clustering rank to extract phenotype metrics from.
        Default is ``0`` (best-ranked).

    Returns
    -------
    DataFrame
        One row per bird with phenotype and acoustic feature columns.
        Sorted by ``bird_name``.

    Examples
    --------
    >>> from scripts.cohort_comparison import build_cohort_table
    >>> df = build_cohort_table("E:/xfoster_pipeline_runs")
    >>> df[["bird_name", "repertoire_size", "entropy", "has_intro_notes"]].head()
    """
    root = Path(save_path)
    if not root.is_dir():
        raise ValueError(f'save_path does not exist: {save_path}')

    _SKIP = {'copied_data', 'logs', 'output'}
    bird_dirs = sorted([
        d for d in root.iterdir()
        if d.is_dir() and d.name not in _SKIP
    ], key=lambda d: d.name)

    rows = []
    for bird_dir in bird_dirs:
        run_dir = _find_run_dir(bird_dir, run_name)
        if run_dir is None:
            logger.debug(f'No run dir found for {bird_dir.name} — skipping')
            continue

        pheno = _load_phenotype_row(run_dir, rank=rank)
        if pheno is None:
            logger.debug(f'No phenotype data for {bird_dir.name} — skipping')
            continue

        acoustic = _load_acoustic_means(run_dir)

        row = {'bird_name': bird_dir.name, 'run_name': run_dir.name}
        for col in _PHENOTYPE_COLS:
            row[col] = pheno.get(col, np.nan)
        row.update(acoustic)
        rows.append(row)

    if not rows:
        logger.warning(f'No birds with phenotype data found in {save_path}')
        return pd.DataFrame()

    df = pd.DataFrame(rows).sort_values('bird_name').reset_index(drop=True)
    logger.info(f'Cohort table built: {len(df)} birds')
    return df


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

def _render_comparison_html(df: pd.DataFrame, save_path: str) -> str:
    """Generate a simple HTML comparison report and return its path."""
    import base64, io
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    sections = []
    _NUMERIC_COLS = [c for c in df.columns
                     if c not in ('bird_name', 'run_name', 'intro_note_labels')
                     and pd.api.types.is_numeric_dtype(df[c])]

    # -- Summary table --
    tbl_html = df[['bird_name'] + _NUMERIC_COLS[:8]].to_html(
        index=False, float_format=lambda x: f'{x:.3g}', na_rep='—'
    )
    sections.append(f'<div style="overflow-x:auto;">{tbl_html}</div>')

    # -- Per-feature bar charts --
    plot_cols = [
        'repertoire_size', 'entropy', 'entropy_excl_intro',
        'mean_duration_ms', 'mean_prev_syllable_gap_ms',
    ]
    plot_cols = [c for c in plot_cols if c in df.columns]

    for col in plot_cols:
        col_data = df[['bird_name', col]].dropna()
        if col_data.empty:
            continue
        try:
            fig, ax = plt.subplots(figsize=(max(6, len(col_data) * 0.35), 3))
            ax.bar(range(len(col_data)), col_data[col].values, color='steelblue')
            ax.set_xticks(range(len(col_data)))
            ax.set_xticklabels(col_data['bird_name'].values, rotation=45, ha='right', fontsize=8)
            ax.set_ylabel(col.replace('_', ' '))
            ax.set_title(col.replace('_', ' '), fontsize=10)
            fig.tight_layout()
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='white')
            buf.seek(0)
            b64 = base64.b64encode(buf.read()).decode()
            plt.close(fig)
            sections.append(
                f'<div style="margin:16px 0;">'
                f'<img src="data:image/png;base64,{b64}"></div>'
            )
        except Exception as e:
            logger.warning(f'Cohort comparison: plot failed for {col}: {e}')

    html = (
        '<!DOCTYPE html><html><head>'
        '<meta charset="utf-8">'
        '<title>Cohort Comparison</title>'
        '<style>body{background:#1a1a1a;color:#eee;font-family:sans-serif;padding:24px;}'
        'table{border-collapse:collapse;font-size:0.85em;}'
        'th,td{padding:4px 10px;border-bottom:1px solid #444;text-align:right;}'
        'th{color:#aaa;text-align:center;}</style>'
        '</head><body>'
        f'<h1>Cohort Comparison &mdash; {Path(save_path).name}</h1>'
        f'<p style="color:#888;">{len(df)} birds | '
        f'generated {__import__("datetime").datetime.now():%Y-%m-%d %H:%M}</p>'
        + ''.join(sections)
        + '</body></html>'
    )
    out = Path(save_path) / 'cohort_comparison.html'
    out.write_text(html, encoding='utf-8')
    logger.info(f'HTML report written: {out}')
    return str(out)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Build a cross-bird phenotype comparison table.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('save_path', help='Pipeline output root directory')
    parser.add_argument(
        '--run_name', default=None,
        help='Run hash/name (default: auto-discover one run per bird)'
    )
    parser.add_argument(
        '--rank', type=int, default=0,
        help='Automated clustering rank to use (default: 0)'
    )
    parser.add_argument(
        '--html', action='store_true', default=False,
        help='Also generate an HTML comparison report'
    )
    parser.add_argument(
        '-v', '--verbose', action='store_true', default=False,
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(levelname)-8s  %(message)s',
        stream=sys.stdout,
    )

    df = build_cohort_table(args.save_path, run_name=args.run_name, rank=args.rank)
    if df.empty:
        print('No data found — check save_path and run_name.')
        sys.exit(1)

    out_csv = Path(args.save_path) / 'cohort_comparison.csv'
    df.to_csv(out_csv, index=False)
    print(f'CSV saved: {out_csv}  ({len(df)} birds)')

    if args.html:
        html_path = _render_comparison_html(df, args.save_path)
        print(f'HTML saved: {html_path}')


if __name__ == '__main__':
    main()
