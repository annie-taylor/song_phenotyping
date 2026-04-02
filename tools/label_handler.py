# Compatibility shim — import from the new package location.
# Will be removed once all pipeline modules are migrated to song_phenotyping/.
from song_phenotyping.tools.label_handler import (  # noqa: F401
    LabelType, LabelHandler, has_manual_labels
)
