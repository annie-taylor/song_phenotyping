# Compatibility shim — all symbols now live in song_phenotyping.labelling
from song_phenotyping.labelling import (  # noqa: F401
    HDBSCANParams,
    DEFAULT_HDBSCAN_GRID,
    compute_scores,
    cluster_embeddings,
    label_bird,
)
