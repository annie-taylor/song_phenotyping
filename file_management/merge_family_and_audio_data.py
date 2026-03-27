#!/usr/bin/env python3

import os
import pandas as pd
import json

BASE_DIR = "/Users/annietaylor/Documents/ucsf/brainard/globus/song_phenotyping"
OUTPUT_DIR = "/Users/annietaylor/Documents/ucsf/brainard/x-foster"

FAMILY_CSV = os.path.join(OUTPUT_DIR, "nest_gen_pair_offspring_summary.csv")
AUDIO_JSON = os.path.join(BASE_DIR, "file_management", "priority_bird_songpaths.json")
SPEC_CSV = os.path.join(BASE_DIR, "file_management", "xfoster_specs", "spectrogram_manifest.csv")

OUT_FAMILY_CSV = os.path.join(OUTPUT_DIR, "nest_gen_pair_offspring_with_audio.csv")
OUT_FAMILY_SUMMARY_CSV = os.path.join(OUTPUT_DIR, "nest_gen_pair_family_audio_summary.csv")

def split_birds(cell):
    if pd.isna(cell) or not str(cell).strip():
        return []
    return [b.strip() for b in str(cell).split(";") if b.strip()]

def main():
    family_df = pd.read_csv(FAMILY_CSV)

    with open(AUDIO_JSON, "r") as f:
        audio_json = json.load(f)

    # Birds that have at least one audio file discovered
    birds_with_audio = set(audio_json.keys())

    def count_audio_in_cell(cell):
        birds = split_birds(cell)
        return sum(b in birds_with_audio for b in birds)

    family_df["n_hr_with_audio"] = family_df["HR Birds"].apply(count_audio_in_cell)
    family_df["n_xf_with_audio"] = family_df["XF Birds"].apply(count_audio_in_cell)

    family_df["hr_has_audio"] = family_df["n_hr_with_audio"] > 0
    family_df["xf_has_audio"] = family_df["n_xf_with_audio"] > 0
    family_df["family_has_audio"] = (family_df["n_hr_with_audio"] + family_df["n_xf_with_audio"]) > 0

    family_df.to_csv(OUT_FAMILY_CSV, index=False)
    print(f"Saved: {OUT_FAMILY_CSV}")

    family_summary = family_df.groupby(["Nest Father", "Genetic Father"], dropna=False).agg(
        n_families=("Pair", "size"),
        n_hr_birds=("HR Birds", lambda s: sum(len(split_birds(x)) for x in s)),
        n_xf_birds=("XF Birds", lambda s: sum(len(split_birds(x)) for x in s)),
        n_hr_with_audio=("n_hr_with_audio", "sum"),
        n_xf_with_audio=("n_xf_with_audio", "sum"),
        any_audio=("family_has_audio", "any"),
    ).reset_index()

    family_summary.to_csv(OUT_FAMILY_SUMMARY_CSV, index=False)
    print(f"Saved: {OUT_FAMILY_SUMMARY_CSV}")

if __name__ == "__main__":
    main()