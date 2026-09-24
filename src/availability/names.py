"""
Player-name normalization for joining injury-report names (free text from PDFs)
to nba_api player names.

Both sources spell names slightly differently ("Jaren Jackson Jr." vs "Jaren
Jackson Jr", accents, spacing artifacts from PDF extraction such as
"Dereck LivelyII" or "Gary TrentJr."). Matching is done on a normalized key,
restricted to the same (team, season) so that a normalized collision across the
league cannot cause a wrong join.
"""

import re
import unicodedata

_SUFFIXES = ("jr", "sr", "iii", "ii", "iv")  # longest first so "iii" wins over "ii"
_GLUED_SUFFIX_RE = re.compile(
    r"^(?P<stem>[a-z]{3,}?)(?:" + "|".join(_SUFFIXES) + r")$"
)  # lazy stem: 'williamsiii' -> 'williams'
_PUNCT_RE = re.compile(r"[^a-z0-9 ]")
_WS_RE = re.compile(r"\s+")


def normalize_name(name: str) -> str:
    """'Jaren Jackson Jr.' -> 'jaren jackson'; 'Nikola Jokić' -> 'nikola jokic';
    'Dereck LivelyII' -> 'dereck lively'."""
    if not name:
        return ""
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii").lower()
    if "," in s:  # "Last, First"
        last, first = s.split(",", 1)
        s = f"{first} {last}"
    s = _PUNCT_RE.sub(" ", s)
    tokens = [t for t in _WS_RE.split(s.strip()) if t and t not in _SUFFIXES]
    if len(tokens) >= 2:
        m = _GLUED_SUFFIX_RE.match(tokens[-1])
        if m:
            tokens[-1] = m.group("stem")
    return " ".join(tokens)


def squash(name_key: str) -> str:
    """Remove spaces from a normalized key: fallback for PDF rows with glued tokens."""
    return name_key.replace(" ", "")
