# Compatibility shim — all symbols now live in song_phenotyping.phenotyping
from song_phenotyping.phenotyping import (  # noqa: F401
    PhenotypingConfig,
    phenotype_bird,
    load_bird_syllable_data,
    load_clustering_results,
    load_clustering_labels_for_syllables,
    save_detailed_phenotype_data,
)
