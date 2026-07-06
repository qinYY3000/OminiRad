"""Anatomy lexicons for lightweight NER from English radiology reports.

These lexicons map free-text location phrases to standardized snake_case anatomy
codes used by the SAR-Loc `<LOC>` token (see `docs/innovation.md` §3.4.2 and
`docs/dataset_plan.md` §5.6).

Each lexicon is a list of (regex_pattern, code) pairs evaluated in order.
The first matching pattern wins.

Reports are assumed to be in English (per dataset_plan.md §9.4).
"""

from .breast import BREAST_ANATOMY_LEXICON
from .thyroid import THYROID_ANATOMY_LEXICON
from .chest import CHEST_ANATOMY_LEXICON

import re


def extract_anatomy(report: str, lexicon: list) -> str | None:
    """Return the first matching standardized anatomy code, or None.

    Parameters
    ----------
    report   : English radiology report text (case-insensitive matching applied).
    lexicon  : list of (regex_pattern, code) tuples.

    Returns
    -------
    A standardized snake_case code (e.g. 'right_breast_upper_outer'),
    or None if no pattern matches.
    """
    text = report.lower()
    for pattern, code in lexicon:
        if re.search(pattern, text, flags=re.IGNORECASE):
            return code
    return None


# Convenience dispatcher: pick the right lexicon by anatomy type.
LEXICONS_BY_ANATOMY = {
    "breast":  BREAST_ANATOMY_LEXICON,
    "thyroid": THYROID_ANATOMY_LEXICON,
    "chest":   CHEST_ANATOMY_LEXICON,
}


def extract_anatomy_auto(report: str, anatomy: str) -> str | None:
    """Auto-select lexicon by anatomy type and run extraction."""
    lex = LEXICONS_BY_ANATOMY.get(anatomy)
    if lex is None:
        return None
    return extract_anatomy(report, lex)


__all__ = [
    "BREAST_ANATOMY_LEXICON",
    "THYROID_ANATOMY_LEXICON",
    "CHEST_ANATOMY_LEXICON",
    "LEXICONS_BY_ANATOMY",
    "extract_anatomy",
    "extract_anatomy_auto",
]
