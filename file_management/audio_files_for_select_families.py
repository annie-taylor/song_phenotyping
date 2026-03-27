import sys
import os
sys.path.append(os.path.dirname(os.getcwd()))
import re
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from comprehensive_metadata_and_filesearch import get_file_locations_for_birds
from tools.dbquery import getUUIDfromBands
from tools.system_utils import check_sys_for_macaw_root


AUDIO_TIMESTAMP_RE = re.compile(r"^.+\.\d{8}\.\d{4}(\d{2})?$")  # .YYYYMMDD.HHMM or .YYYYMMDD.HHMMSS
BIRD_ALIASES = {"ye1tut0": ["ye1tut0", "ye1", "y1"],}


def expand_bird_aliases(bird):
    return BIRD_ALIASES.get(bird, [bird])


def parse_macaw_dirs_file(txt_file_path):
    """Parse MacawAllDirsByBird.txt to get bird -> directories mapping."""
    bird_directories = {}

    try:
        with open(txt_file_path, "r") as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(",")
                if len(parts) >= 4:
                    bird_name = parts[0].strip()
                    dir_string = parts[3].strip("[]")
                    directories = [d.strip() for d in dir_string.split("//") if d.strip()]
                    bird_directories[bird_name] = directories
    except FileNotFoundError:
        print(f"File {txt_file_path} not found!")
        return {}

    return bird_directories


def get_screening_directories(bird_name, root_dir):
    screening_paths = [
        "public/screening",
        "public/adult_screening",
        "public/from_egret/egret/screening",
        "public/from_stork/stork/screening",
    ]

    dirs = []
    for path in screening_paths:
        full = os.path.join(root_dir, path)
        if os.path.exists(full):
            try:
                for sub in os.listdir(full):
                    if bird_name.lower() in sub.lower():
                        dirs.append(os.path.join(full, sub))
            except Exception:
                continue
    return dirs


def is_audio_candidate(filename: str) -> bool:
    name = Path(filename).name.lower()

    if name.endswith((".wav", ".cbin")):
        return True

    # extensionless recording names like bk100bk99.20070925.0900
    return bool(AUDIO_TIMESTAMP_RE.match(name))


def get_audio_files_from_dirs(dirs, min_size_mb=0.1):
    """Scan directories for plausible audio files and return (filepath, size_bytes)."""
    files = []
    for d in dirs:
        if not os.path.isdir(d):
            continue
        try:
            for f in os.listdir(d):
                if not is_audio_candidate(f):
                    continue

                fp = os.path.join(d, f)
                if os.path.isfile(fp):
                    try:
                        size = os.path.getsize(fp)
                        if size >= min_size_mb * 1024 * 1024:
                            files.append((fp, size))
                    except OSError:
                        continue
        except Exception:
            continue
    return files


def process_one_bird(
    bird,
    txt_path,
    db_path,
    root_dir,
    max_files=5,
    min_size_mb=0.1,
    max_dirs_per_bird=5,
):
    """
    Resolve candidate directories for one bird, scan them, and keep the largest audio files.
    Returns (bird, result_dict).
    """
    search_terms = expand_bird_aliases(bird)

    primary_map = get_file_locations_for_birds(
        search_terms,
        txt_file_path=txt_path,
        db_path=db_path,
    )

    dirs = set()
    for alias in search_terms:
        for fp in primary_map.get(alias, []):
            fp = fp.replace("\\", "/")
            full_path = fp if os.path.isabs(fp) else os.path.join(root_dir, fp)
            if os.path.exists(full_path):
                dirs.add(os.path.dirname(full_path))

    dirs = list(dirs)[:max_dirs_per_bird]
    files = get_audio_files_from_dirs(dirs, min_size_mb=min_size_mb)

    files.sort(key=lambda x: x[1], reverse=True)
    top = files[:max_files]

    return bird, {
        "n_dirs_searched": len(dirs),
        "n_files_selected": len(top),
        "files": [
            {
                "filename": os.path.basename(fp),
                "filepath": fp,
                "size_mb": round(size / (1024 * 1024), 2),
            }
            for fp, size in top
        ],
    }


def get_top_songs_parallel(
    birds,
    txt_path,
    db_path,
    root_dir,
    max_files=5,
    min_size_mb=0.1,
    max_dirs_per_bird=5,
    max_workers=None,
):
    """
    Parallel version of the notebook logic.
    Returns a dict: bird -> {"n_dirs_searched", "n_files_selected", "files", ...}
    """
    birds = sorted(set(birds))
    results = {}

    if max_workers is None:
        max_workers = min(32, (os.cpu_count() or 1))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                process_one_bird,
                bird,
                txt_path,
                db_path,
                root_dir,
                max_files,
                min_size_mb,
                max_dirs_per_bird,
            ): bird
            for bird in birds
        }

        for fut in as_completed(futures):
            bird, out = fut.result()
            results[bird] = out
            print(f"{bird}: {out.get('n_files_selected', 0)} files selected")

    return results


def save_audio_lookup_results(results, out_json_path, out_csv_path):
    """Save nested results to JSON and a flat CSV."""
    with open(out_json_path, "w") as f:
        json.dump(results, f, indent=2)

    rows = []
    for bird, info in results.items():
        for item in info.get("files", []):
            rows.append({
                "bird": bird,
                "filename": item["filename"],
                "filepath": item["filepath"],
                "size_mb": item["size_mb"],
                "n_dirs_searched": info.get("n_dirs_searched", 0),
                "n_files_selected": info.get("n_files_selected", 0),
                "error": info.get("error", ""),
            })

    pd.DataFrame(rows).to_csv(out_csv_path, index=False)


def get_birds_from_nest_summary(csv_path, top_n=3):
    df = pd.read_csv(csv_path)
    df = df.sort_values(by="# XF", ascending=False).head(top_n)

    birds = set()

    for _, row in df.iterrows():
        if pd.notna(row.get("HR Birds")) and row["HR Birds"]:
            birds.update([b.strip() for b in row["HR Birds"].split(";") if b.strip()])

        if pd.notna(row.get("XF Birds")) and row["XF Birds"]:
            birds.update([b.strip() for b in row["XF Birds"].split(";") if b.strip()])

        if pd.notna(row.get("Nest Father")) and row["Nest Father"]:
            birds.add(row["Nest Father"].strip())

        if pd.notna(row.get("Genetic Fathers of XF")) and row["Genetic Fathers of XF"]:
            birds.update([b.strip() for b in row["Genetic Fathers of XF"].split(";") if b.strip()])

    return sorted(birds)


if __name__ == "__main__":
    root_dir = check_sys_for_macaw_root()

    txt_path = "../refs/MacawAllDirsByBird.txt"
    db_path = "../refs/2026-01-15-db.sqlite3"

    print("cwd:", os.getcwd())
    print("txt_path:", os.path.abspath(txt_path), os.path.exists(txt_path))
    print("db_path:", os.path.abspath(db_path), os.path.exists(db_path))

    if not os.path.exists(txt_path):
        raise FileNotFoundError(f"Missing txt_path: {txt_path}")
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Missing db_path: {db_path}")

    birds = get_birds_from_nest_summary(
        "/Users/annietaylor/Documents/ucsf/brainard/x-foster/nest_father_offspring_summary.csv")

    results = get_top_songs_parallel(
        birds=birds,
        txt_path=txt_path,
        db_path=db_path,
        root_dir=root_dir,
        max_files=5,
        min_size_mb=0.1,
        max_dirs_per_bird=5,
        max_workers=12,
    )

    # Flatten for downstream compatibility
    song_results = {
        bird: info["files"]
        for bird, info in results.items()
    }

    # Save JSON (used by spectrogram + segmentation)
    json_path = os.path.join(os.getcwd(), "priority_bird_songpaths.json")
    with open(json_path, "w") as f:
        json.dump(song_results, f, indent=2)

    # Also save CSV (nice for inspection)
    csv_path = os.path.join(root_dir, "file_management", "audio_lookup_results.csv")
    save_audio_lookup_results(results, out_json_path=json_path, out_csv_path=csv_path)

    print(f"Saved JSON to: {json_path}")