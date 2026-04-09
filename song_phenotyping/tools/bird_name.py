"""Bird name normalization utilities.

Band-based bird names encode two leg-band colors and numbers, e.g. ``bu34or18``
(blue band 34, orange band 18).  Color abbreviations are not standardized across
labs or time periods: ``r``, ``rd``, and (rarely) other forms may all mean red.
This module provides a single normalization function that maps every known variant
to a canonical two-letter code so the pipeline always uses a consistent name.

Canonical color codes
---------------------
=====================  =======
Color                  Code
=====================  =======
Black                  ``bk``
Blue                   ``bu``
Brown                  ``br``
Green                  ``gr``
Orange                 ``or``
Pink                   ``pk``
Purple                 ``pu``
Red                    ``rd``
White                  ``wh``
Yellow                 ``ye``
=====================  =======

Non-standard abbreviations that are normalized
-----------------------------------------------
- ``g``  → ``gr`` (green)
- ``o``  → ``or`` (orange)
- ``r``  → ``rd`` (red)
- ``w``  → ``wh`` (white)
- ``wt`` → ``wh`` (white, variant spelling)
- ``y``  → ``ye`` (yellow)

Names that do not fully parse as ``(letters)(digits)+`` tokens — for example
nest-based IDs like ``n156unb1`` or names with a trailing letter like ``y37s``
— are returned **unchanged**.
"""

import re

# Map from any known abbreviation to the canonical two-letter code.
# Only entries that differ from canonical form are listed here; all others
# (bk, bu, br, gr, or, pk, pu, rd, wh, ye) are already canonical and need
# no entry.
_COLOR_NORMALIZE: dict[str, str] = {
    'g':  'gr',   # green
    'o':  'or',   # orange
    'r':  'rd',   # red
    'w':  'wh',   # white
    'wt': 'wh',   # white (variant)
    'y':  'ye',   # yellow
}

# Matches one (letters)(digits) token pair.
_TOKEN_RE = re.compile(r'([a-zA-Z]+)(\d+)')


def normalize_bird_name(name: str) -> str:
    """Return the canonical form of a band-based bird name.

    Color abbreviations are expanded to their standard two-letter codes (see
    module docstring).  The name is lowercased before processing.

    Names that do not fully decompose into ``(color)(number)+`` tokens are
    returned unchanged (lowercased) — this covers out-of-distribution names
    such as nest IDs (``n156unb1``), single-band birds with a trailing
    qualifier (``y37s``), and any other non-standard identifiers.

    Parameters
    ----------
    name : str
        Raw bird identifier, e.g. ``'r25w57'``, ``'bu34OR18'``, ``'n34unb'``.

    Returns
    -------
    str
        Canonical identifier, e.g. ``'rd25wh57'``, ``'bu34or18'``, ``'n34unb'``.

    Examples
    --------
    >>> normalize_bird_name('r25w57')
    'rd25wh57'
    >>> normalize_bird_name('bu34o18')
    'bu34or18'
    >>> normalize_bird_name('pu91wt67')
    'pu91wh67'
    >>> normalize_bird_name('y25')
    'ye25'
    >>> normalize_bird_name('n34unb')     # out-of-distribution — returned as-is
    'n34unb'
    >>> normalize_bird_name('y37s')       # trailing letter — returned as-is
    'y37s'
    >>> normalize_bird_name('rd25wh57')   # already canonical
    'rd25wh57'
    """
    lower = name.lower()
    tokens = _TOKEN_RE.findall(lower)
    if not tokens:
        return lower

    # Only normalize if the tokens reconstruct the full name exactly.
    # If there are trailing characters (e.g. the 's' in 'y37s'), leave unchanged.
    reconstructed = ''.join(color + num for color, num in tokens)
    if reconstructed != lower:
        return lower

    return ''.join(_COLOR_NORMALIZE.get(color, color) + num for color, num in tokens)
