"""
Shared label handling for the song phenotyping pipeline.

All pipeline stages that read or write syllable labels should use LabelType and
LabelHandler from here so that label normalization is consistent end-to-end.

Label conventions:
  Manual labels  — single characters ('a'–'z'), start token 's', end token 'z'
  Auto labels    — integers (HDBSCAN output), start token -5, end token -3
"""

from enum import Enum
from typing import Any, Dict, List, Union

import numpy as np


class LabelType(Enum):
    """Source / format of syllable labels."""
    MANUAL = "manual"
    AUTO = "hdbscan"


class LabelHandler:
    """
    Normalize and tokenize syllable labels for a given LabelType.

    Use this wherever labels are read from HDF5 or passed between pipeline
    stages so that manual and auto labels are handled identically upstream.
    """

    def __init__(self, label_type: LabelType):
        self.label_type = label_type

    @property
    def start_token(self) -> Union[str, int]:
        return 's' if self.label_type == LabelType.MANUAL else -5

    @property
    def end_token(self) -> Union[str, int]:
        return 'z' if self.label_type == LabelType.MANUAL else -3

    @property
    def non_syl_tokens(self) -> List[Union[str, int]]:
        """Tokens that mark song boundaries rather than syllable identity."""
        if self.label_type == LabelType.MANUAL:
            return ['s', 'z', '\r']
        else:
            return [-5, -3]

    def normalize_labels(self, raw_labels: List[Any]) -> List[Union[str, int]]:
        """Convert raw labels (possibly bytes) to consistent str or int format."""
        if self.label_type == LabelType.MANUAL:
            return [self._to_string(label) for label in raw_labels]
        else:
            return [self._to_int(label) for label in raw_labels]

    def add_sequence_tokens(self, labels: List[Union[str, int]]) -> List[Union[str, int]]:
        """Wrap a label sequence with start/end song-boundary tokens."""
        return [self.start_token] + labels + [self.end_token]

    # ------------------------------------------------------------------
    # Internal converters
    # ------------------------------------------------------------------

    @staticmethod
    def _to_string(item: Any) -> str:
        if isinstance(item, (bytes, np.bytes_)):
            return item.decode('utf-8')
        return str(item)

    @staticmethod
    def _to_int(item: Any) -> int:
        if isinstance(item, (bytes, np.bytes_)):
            return int(item.decode('utf-8'))
        elif isinstance(item, str):
            return int(item)
        return int(item)


def has_manual_labels(syllable_data: Dict[str, Any]) -> bool:
    """Return True if syllable_data contains non-empty manual labels."""
    return len(syllable_data.get('manual_syllables', [])) > 0
