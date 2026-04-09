"""One-time migration utility: rename non-canonical bird output directories.

The song-phenotyping pipeline now normalizes all bird names to a canonical
two-letter color code (e.g. ``r`` -> ``rd``, ``o`` -> ``or``, ``wt`` -> ``wh``).
Existing pipeline output directories created *before* this fix may use
non-canonical names.  This script identifies those directories, renames them
to the canonical form, and merges any content that belongs in the same
canonical directory.

Usage
-----
Dry-run (no changes, just shows what would happen)::

    python scripts/migrate_bird_names.py E:/xfoster_pipeline_runs

Apply changes::

    python scripts/migrate_bird_names.py E:/xfoster_pipeline_runs --apply

Notes
-----
- Run this script **once** before re-running the pipeline on existing data.
- After migration, re-run the pipeline normally; it will find the canonically
  named directories and skip already-completed stages.
- The script never deletes HDF5 files — it only moves directories.  If a
  hash subdirectory exists in both the source and target, contents are merged
  file-by-file (target file wins on conflict).
- Logging is written to ``<save_path>/logs/migrate_bird_names_<timestamp>.log``.
"""

import argparse
import logging
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

# Allow running directly without installing the package
_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from song_phenotyping.tools.bird_name import normalize_bird_name

logger = logging.getLogger(__name__)

# Directories under save_path that are never bird output dirs
_SKIP_DIRS = {'copied_data', 'logs', 'output'}


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def _setup_logging(save_path: str) -> None:
    log_dir = Path(save_path) / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = log_dir / f'migrate_bird_names_{ts}.log'

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s  %(levelname)-8s  %(message)s',
        handlers=[
            logging.FileHandler(str(log_path), encoding='utf-8'),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logger.info(f'Migration log: {log_path}')


def _merge_dir_into(src: Path, dst: Path, dry_run: bool) -> int:
    """Recursively merge *src* into *dst*, skipping files that already exist in *dst*.

    Returns the number of files moved.
    """
    moved = 0
    for item in src.rglob('*'):
        if not item.is_file():
            continue
        rel = item.relative_to(src)
        target = dst / rel
        if target.exists():
            logger.warning(f'  SKIP (already exists in target): {rel}')
            continue
        if not dry_run:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(item), str(target))
        logger.info(f'  {"WOULD MOVE" if dry_run else "MOVED"}: {rel}')
        moved += 1
    return moved


def _remove_empty_tree(path: Path, dry_run: bool) -> None:
    """Remove *path* if it is now empty (or would be after a dry-run merge)."""
    if dry_run:
        logger.info(f'  WOULD REMOVE (now empty): {path.name}')
        return
    try:
        shutil.rmtree(str(path))
        logger.info(f'  REMOVED (now empty): {path.name}')
    except Exception as e:
        logger.warning(f'  Could not remove {path}: {e}')


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def find_migration_plan(save_path: str) -> dict[str, list[str]]:
    """Return a mapping ``{canonical_name: [dir_name, ...]}``.

    Only canonical names that have **more than one** variant directory (or
    exactly one non-canonical directory) are included.
    """
    root = Path(save_path)
    groups: dict[str, list[str]] = {}
    for item in root.iterdir():
        if not item.is_dir() or item.name in _SKIP_DIRS:
            continue
        canonical = normalize_bird_name(item.name)
        groups.setdefault(canonical, []).append(item.name)

    # Keep only groups that need action:
    # - Multiple variants -> need merging
    # - Single non-canonical variant -> need renaming
    plan = {}
    for canonical, variants in groups.items():
        needs_action = len(variants) > 1 or variants[0] != canonical
        if needs_action:
            plan[canonical] = variants

    return plan


# ---------------------------------------------------------------------------
# Migration
# ---------------------------------------------------------------------------

def migrate(save_path: str, dry_run: bool = True) -> None:
    """Migrate all non-canonical bird directories under *save_path*.

    Parameters
    ----------
    save_path : str
        Pipeline output root (same as ``pipeline.save_path`` in config.yaml).
    dry_run : bool
        If ``True`` (default), only print what *would* happen — no files are
        moved or deleted.  Pass ``False`` (or use ``--apply`` on the CLI) to
        make real changes.
    """
    root = Path(save_path)
    if not root.is_dir():
        logger.error(f'save_path does not exist: {save_path}')
        return

    plan = find_migration_plan(save_path)

    if not plan:
        logger.info('No non-canonical directories found — nothing to do.')
        return

    mode = 'DRY RUN' if dry_run else 'APPLYING'
    logger.info(f'\n{"="*60}')
    logger.info(f'  {mode}: {len(plan)} canonical name(s) need attention')
    logger.info(f'  save_path = {save_path}')
    logger.info(f'{"="*60}\n')

    total_moved = 0
    total_renamed = 0

    for canonical, variants in sorted(plan.items()):
        logger.info(f'[{canonical}]  variants: {variants}')
        canonical_dir = root / canonical

        # Separate canonical from non-canonical variants
        non_canonical = [v for v in variants if v != canonical]

        for variant in non_canonical:
            variant_dir = root / variant
            logger.info(f'  Processing non-canonical: {variant} -> {canonical}')

            if not canonical_dir.exists():
                # Simple rename — canonical dir doesn't exist yet
                if not dry_run:
                    variant_dir.rename(canonical_dir)
                logger.info(f'  {"WOULD RENAME" if dry_run else "RENAMED"}: '
                             f'{variant} -> {canonical}')
                total_renamed += 1
            else:
                # Canonical dir already exists — merge content
                logger.info(f'  Canonical dir already exists; merging content…')
                n = _merge_dir_into(variant_dir, canonical_dir, dry_run)
                total_moved += n

                # Merge audio_paths.txt (append unique lines)
                src_apt = variant_dir / 'audio_paths.txt'
                dst_apt = canonical_dir / 'audio_paths.txt'
                if src_apt.exists():
                    _merge_audio_paths_txt(src_apt, dst_apt, dry_run)

                # Remove the (now empty or fully merged) variant dir
                if not dry_run:
                    # Remove any remaining empty dirs
                    for empty_dir in sorted(variant_dir.rglob('*'), reverse=True):
                        if empty_dir.is_dir() and not any(empty_dir.iterdir()):
                            empty_dir.rmdir()
                    if variant_dir.is_dir() and not any(variant_dir.iterdir()):
                        variant_dir.rmdir()
                    elif variant_dir.is_dir():
                        logger.warning(
                            f'  {variant} still has files after merge — NOT removed. '
                            f'Check manually: {variant_dir}'
                        )
                else:
                    logger.info(f'  WOULD REMOVE (after merge): {variant}')

        logger.info('')

    logger.info(f'Summary: {total_renamed} dir(s) renamed, {total_moved} file(s) moved.')
    if dry_run:
        logger.info('This was a DRY RUN — no changes were made.')
        logger.info('Re-run with --apply to apply these changes.')


def _merge_audio_paths_txt(src: Path, dst: Path, dry_run: bool) -> None:
    """Append lines from *src* audio_paths.txt that are not already in *dst*."""
    existing_lines: set[str] = set()
    if dst.exists():
        existing_lines = set(dst.read_text(encoding='utf-8').splitlines())

    new_lines = [
        line for line in src.read_text(encoding='utf-8').splitlines()
        if line and not line.startswith('#') and line not in existing_lines
    ]

    if not new_lines:
        logger.info(f'  audio_paths.txt: no new lines to merge from {src.parent.name}')
        return

    logger.info(f'  {"WOULD APPEND" if dry_run else "APPENDING"} '
                f'{len(new_lines)} line(s) to audio_paths.txt')
    if not dry_run:
        with dst.open('a', encoding='utf-8') as f:
            f.write('\n'.join(new_lines) + '\n')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Rename/merge non-canonical bird output directories.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        'save_path',
        help='Pipeline output root directory (pipeline.save_path in config.yaml)',
    )
    parser.add_argument(
        '--apply',
        action='store_true',
        default=False,
        help='Actually rename/move files (default is dry-run only)',
    )
    args = parser.parse_args()

    _setup_logging(args.save_path)
    migrate(args.save_path, dry_run=not args.apply)


if __name__ == '__main__':
    main()
