"""
song_phenotyping
================

Automated pipeline for phenotyping Bengalese finch song.

Pipeline stages
---------------
A. :mod:`~song_phenotyping.ingestion`
   Extract syllable spectrograms from raw audio + segmentation files.
B. :mod:`~song_phenotyping.flattening`
   Flatten 2-D spectrograms into 1-D feature vectors.
C. :mod:`~song_phenotyping.embedding`
   Reduce dimensionality with UMAP.
D. :mod:`~song_phenotyping.labelling`
   Cluster embeddings with HDBSCAN and evaluate cluster quality.
E. :mod:`~song_phenotyping.phenotyping`
   Compute song-level phenotype metrics from syllable sequences.

Quick start
-----------
>>> from song_phenotyping import ingestion, flattening, embedding, labelling, phenotyping
>>> from song_phenotyping.tools.spectrogram_configs import SpectrogramParams
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("song-phenotyping")
except PackageNotFoundError:
    __version__ = "0.1.0-dev"
