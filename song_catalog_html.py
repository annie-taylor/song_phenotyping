# Compatibility shim — all symbols now live in song_phenotyping.catalog
from song_phenotyping.catalog import (  # noqa: F401
    CatalogConfig,
    generate_song_catalog,
    generate_syllable_type_catalog,
    generate_all_catalogs,
)
