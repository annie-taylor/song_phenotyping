# Compatibility shim — import from the new package location.
# Will be removed once all pipeline modules are migrated to song_phenotyping/.
from song_phenotyping.embedding import (  # noqa: F401
    UMAPParams,
    explore_embedding_parameters_robust,
    load_flattened_specs,
    save_umap_embeddings,
    save_umap_model,
    inspect_existing_embeddings,
    load_embedding_from_file,
    subsample_data,
    compute_and_save_umap_memory_aware,
    compute_embedding_grid_parallel_robust,
    compare_umap_embeddings_plot,
)
