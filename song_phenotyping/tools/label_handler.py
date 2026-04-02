"""Unified syllable label handling for the song phenotyping pipeline.

All pipeline stages that read or write syllable labels should import
:class:`LabelType` and :class:`LabelHandler` from here so that label
normalisation is consistent end-to-end.

Label conventions
-----------------
Manual labels
    Single characters ``'a'``–``'z'``, with ``'s'`` as the song-start token
    and ``'z'`` as the song-end token.
Auto (HDBSCAN) labels
    Integers produced by HDBSCAN clustering, with ``-5`` as the song-start
    token and ``-3`` as the song-end token.
"""

from enum import Enum
from typing import Any, Dict, List, Union

import numpy as np


class LabelType(Enum):
    """Enumeration of syllable label sources.

    Attributes
    ----------
    MANUAL : str
        Human-annotated labels stored as single characters.
    AUTO : str
        Automated labels produced by HDBSCAN clustering.
    """

    MANUAL = "manual"
    AUTO = "hdbscan"


class LabelHandler:
    """Normalise and tokenise syllable labels for a given :class:`LabelType`.

    Use this wherever labels are read from HDF5 files or passed between
    pipeline stages so that manual and automated labels are handled
    identically.

    Parameters
    ----------
    label_type : LabelType
        Whether labels are human-annotated (``LabelType.MANUAL``) or
        HDBSCAN-generated (``LabelType.AUTO``).

    Examples
    --------
    >>> handler = LabelHandler(LabelType.MANUAL)
    >>> handler.normalize_labels([b'a', b's', b'b'])
    ['a', 's', 'b']
    >>> handler.add_sequence_tokens(['a', 'b'])
    ['s', 'a', 'b', 'z']
    """

    def __init__(self, label_type: LabelType):
        self.label_type = label_type

    @property
    def start_token(self) -> Union[str, int]:
        """Song-start boundary token (``'s'`` for manual, ``-5`` for auto)."""
        return "s" if self.label_type == LabelType.MANUAL else -5

    @property
    def end_token(self) -> Union[str, int]:
        """Song-end boundary token (``'z'`` for manual, ``-3`` for auto)."""
        return "z" if self.label_type == LabelType.MANUAL else -3

    @property
    def non_syl_tokens(self) -> List[Union[str, int]]:
        """Tokens that mark song boundaries rather than syllable identity.

        Returns
        -------
        list of str or int
            ``['s', 'z', '\\r']`` for manual labels;
            ``[-5, -3]`` for auto labels.
        """
        if self.label_type == LabelType.MANUAL:
            return ["s", "z", "\r"]
        return [-5, -3]

    def normalize_labels(self, raw_labels: List[Any]) -> List[Union[str, int]]:
        """Convert raw labels (possibly bytes) to a consistent Python type.

        Parameters
        ----------
        raw_labels : list
            Labels as read from HDF5 — may be ``bytes``, ``numpy.bytes_``,
            ``str``, or ``int``.

        Returns
        -------
        list of str or int
            String labels for :attr:`LabelType.MANUAL`; integer labels for
            :attr:`LabelType.AUTO`.
        """
        if self.label_type == LabelType.MANUAL:
            return [self._to_string(label) for label in raw_labels]
        return [self._to_int(label) for label in raw_labels]

    def add_sequence_tokens(
        self, labels: List[Union[str, int]]
    ) -> List[Union[str, int]]:
        """Wrap a label sequence with song-boundary tokens.

        Parameters
        ----------
        labels : list of str or int
            Syllable labels for a single song, without boundary tokens.

        Returns
        -------
        list of str or int
            ``[start_token, *labels, end_token]``.
        """
        return [self.start_token] + labels + [self.end_token]

    # ------------------------------------------------------------------
    # Internal converters
    # ------------------------------------------------------------------

    @staticmethod
    def _to_string(item: Any) -> str:
        if isinstance(item, (bytes, np.bytes_)):
            return item.decode("utf-8")
        return str(item)

    @staticmethod
    def _to_int(item: Any) -> int:
        if isinstance(item, (bytes, np.bytes_)):
            return int(item.decode("utf-8"))
        if isinstance(item, str):
            return int(item)
        return int(item)


def has_manual_labels(syllable_data: Dict[str, Any]) -> bool:
    """Return ``True`` if *syllable_data* contains non-empty manual labels.

    Parameters
    ----------
    syllable_data : dict
        Dictionary returned by the syllable-data loader, expected to contain
        the key ``'manual_syllables'``.

    Returns
    -------
    bool
        ``True`` when ``syllable_data['manual_syllables']`` is present and
        non-empty; ``False`` otherwise.
    """
    return len(syllable_data.get("manual_syllables", [])) > 0
