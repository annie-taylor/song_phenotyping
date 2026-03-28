#!/usr/bin/env python3
"""Convert selected EvTAF .cbin recordings to .wav for Audacity inspection.

This version takes a *bird* identifier as input and looks up the bird's selected
file paths in `priority_bird_songpaths.json` (the JSON you already generate for
cross-foster birds). It then converts any `.cbin` files it finds to standard
16-bit PCM `.wav` files.

It reuses the existing `tools.evfuncs.load_cbin()` loader, which reads the
matching `.rec` file to get the sample rate and channel count.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np
import soundfile as sf

from tools.evfuncs import load_cbin
from tools.system_utils import replace_macaw_root


def resolve_existing_path(path: Path) -> Path:
    """Try the path directly, then fall back to your Macaw-root remapping."""
    if path.exists():
        return path

    try:
        alt = replace_macaw_root(path)
        if alt.exists():
            return alt
    except Exception:
        pass

    return path


def load_priority_songpaths(json_path: Path, bird: str) -> list[Path]:
    """Load the selected song paths for one bird from the priority JSON."""
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    with json_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    if bird not in data:
        available = ", ".join(sorted(list(data.keys())[:20]))
        if len(data) > 20:
            available += ", ..."
        raise KeyError(f"Bird '{bird}' not found in {json_path}. Available birds: {available}")

    entries = data[bird]
    if not isinstance(entries, list):
        raise ValueError(f"Expected a list of files for bird '{bird}', got {type(entries).__name__}")

    paths: list[Path] = []
    for entry in entries:
        if isinstance(entry, str):
            candidate = entry
        elif isinstance(entry, dict):
            candidate = entry.get("filepath") or entry.get("path") or entry.get("audio_path") or entry.get("server_path")
        else:
            candidate = None

        if not candidate:
            continue

        paths.append(Path(candidate))

    return paths


def iter_cbin_files_from_bird(entry):
    """
    Yield Path objects for .cbin files from a bird's JSON entry.
    Current JSON shape: list[dict] with keys like filepath, filename, size_mb.
    """
    if isinstance(entry, list):
        for item in entry:
            if isinstance(item, dict):
                fp = item.get("filepath")
                if fp and str(fp).lower().endswith(".cbin"):
                    yield Path(fp)
            elif isinstance(item, str) and item.lower().endswith(".cbin"):
                yield Path(item)


def cbin_to_wav(cbin_path: Path, output_dir: Path | None = None, overwrite: bool = False) -> Path:
    cbin_path = resolve_existing_path(cbin_path)
    if not cbin_path.exists():
        raise FileNotFoundError(f".cbin file not found: {cbin_path}")

    audio, sr = load_cbin(str(cbin_path))
    audio = np.asarray(audio)

    # If multi-channel interleaved, keep first channel by default.
    if audio.ndim > 1:
        audio = audio[:, 0]

    # Convert to native float32 for safer export
    audio = audio.astype(np.float32)

    # Optional: normalize only if the input is not already in a sensible range
    max_abs = np.max(np.abs(audio)) if audio.size else 0.0
    if max_abs > 0:
        audio = audio / max_abs

    if output_dir is None:
        out_path = cbin_path.with_suffix(".wav")
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"{cbin_path.stem}.wav"

    if out_path.exists() and not overwrite:
        return out_path

    # Use FLOAT to preserve fidelity for inspection
    sf.write(str(out_path), audio, sr, subtype="FLOAT")
    return out_path


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert cbin files for one bird listed in priority_bird_songpaths.json."
    )
    parser.add_argument("bird", help="Bird identifier to look up in the JSON file")
    parser.add_argument(
        "-j", "--json-path",
        default=None,
        help="Path to priority_bird_songpaths.json. Defaults to a file next to this script.",
    )
    parser.add_argument(
        "-o", "--output-dir",
        default=None,
        help="Optional output directory. If omitted, writes .wav next to each .cbin file.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing .wav files.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    script_dir = Path(__file__).resolve().parent
    json_path = Path(args.json_path).expanduser() if args.json_path else (script_dir / "file_management"
                                                                          / "priority_bird_songpaths.json")
    output_dir = Path(args.output_dir).expanduser() if args.output_dir else None

    try:
        with open(json_path, "r") as f:
            data = json.load(f)
        entry = data.get(args.bird)
        cbin_files = list(iter_cbin_files_from_bird(entry))
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if not cbin_files:
        print(f"No .cbin files found for bird '{args.bird}' in {json_path}", file=sys.stderr)
        return 1

    converted = 0
    skipped = 0
    failed = 0

    for cbin_file in cbin_files:
        try:
            cbin_file = Path(cbin_file)

            if output_dir is None:
                out_path = cbin_file.with_suffix(".wav")
                if out_path.exists() and not args.overwrite:
                    print(f"SKIP  {cbin_file} -> {out_path} (exists)")
                    skipped += 1
                    continue
            else:
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                out_path = output_dir / (cbin_file.stem + ".wav")
                if out_path.exists() and not args.overwrite:
                    print(f"SKIP  {cbin_file} -> {out_path} (exists)")
                    skipped += 1
                    continue

            out_path = cbin_to_wav(cbin_file, output_dir=output_dir, overwrite=args.overwrite)
            print(f"OK    {cbin_file} -> {out_path}")
            converted += 1

        except Exception as exc:
            print(f"FAIL  {cbin_file}: {exc}", file=sys.stderr)
            failed += 1

    print(f"Done. Converted {converted} file(s); skipped {skipped}; failed {failed}.")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
