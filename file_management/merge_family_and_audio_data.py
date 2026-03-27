#!/usr/bin/env python3

import os
import pandas as pd

BASE_DIR = "/Users/annietaylor/Documents/ucsf/brainard/x-foster"

FAMILY_CSV = os.path.join(BASE_DIR, "nest_gen_pair_offspring_summary.csv")
AUDIO_CSV = os.path.join(BASE_DIR, "file_management", "audio_lookup_results.csv")
SPEC_CSV = os.path.join(BASE_DIR, "file_management", "xfoster_specs", "spectrogram_manifest.csv")

OUT_FAMILY_CSV = os.path.join(BASE_DIR, "nest_gen_pair_offspring_with_audio.csv")
OUT_FAMILY_SUMMARY_CSV = os.path.join(BASE_DIR, "nest_gen_pair_family_audio_summary.csv")


def main():
    family_df = pd.read_csv(FAMILY_CSV)
    audio_df = pd.read_csv(AUDIO_CSV)
    spec_df = pd.read_csv(SPEC_CSV)

    audio_counts = (
        audio_df[audio_df["status"] != "error"]
        .groupby("bird")
        .size()
        .reset_index(name="n_audio_files")
    )

    spec_counts = (
        spec_df[spec_df["status"] == "saved"]
        .groupby("bird")
        .size()
        .reset_index(name="n_spectrograms_saved")
    )

    merged = family_df.merge(
        audio_counts,
        left_on="Bird Name",
        right_on="bird",
        how="left",
    ).drop(columns=["bird"])

    merged = merged.merge(
        spec_counts,
        left_on="Bird Name",
        right_on="bird",
        how="left",
    ).drop(columns=["bird"])

    merged["n_audio_files"] = merged["n_audio_files"].fillna(0).astype(int)
    merged["n_spectrograms_saved"] = merged["n_spectrograms_saved"].fillna(0).astype(int)
    merged["has_audio"] = merged["n_audio_files"] > 0
    merged["has_spectrogram"] = merged["n_spectrograms_saved"] > 0

    merged.to_csv(OUT_FAMILY_CSV, index=False)
    print(f"Saved: {OUT_FAMILY_CSV}")

    family_summary = (
        merged.groupby(["Nest Father", "Genetic Father"], dropna=False)
        .agg(
            n_birds=("Bird Name", "size"),
            n_with_audio=("has_audio", "sum"),
            n_with_spectrogram=("has_spectrogram", "sum"),
        )
        .reset_index()
    )

    family_summary.to_csv(OUT_FAMILY_SUMMARY_CSV, index=False)
    print(f"Saved: {OUT_FAMILY_SUMMARY_CSV}")


if __name__ == "__main__":
    main()