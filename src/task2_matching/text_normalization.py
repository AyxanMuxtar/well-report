"""
Text normalization for drilling-domain text.

Two transformations, applied to BOTH the corpus (op_text) and the queries
(NDS event text) so they live in the same vocabulary:

1. Abbreviation expansion (bidirectional):
   We replace each abbreviation with the abbreviation followed by its long form
   in parentheses. Both forms are present in the resulting string, so TF-IDF
   and SBERT see consistent tokens regardless of which form the writer used.

   Example:
       'POOH with BHA'  becomes  'POOH (pull out of hole) with BHA (bottom hole assembly)'

2. Query cleanup -- fixes the noise pattern in nds_events.xlsx:
   - runs of double-quotes from Excel quote-escape get collapsed to a single one
   - known typos (ccumulation, ecountered, ...) get fixed
   - duplicate consecutive words (to to 0.16) get deduplicated
   - whitespace is normalized

The corpus is normalized once at index-build time (cached); queries are
normalized on the fly before being passed to TfidfIndex.query / SbertIndex.query.
"""
from __future__ import annotations
import re

# =============================================================================
# 1. Abbreviation dictionary (drilling domain)
# =============================================================================
# Longer abbreviations are tried first (so e.g. 'HWDP' is replaced before 'DP').
# The replacement loop sorts by length below.

ABBREVIATIONS: dict[str, str] = {
    # Tripping / running ops
    "POOH":  "pull out of hole",
    "RIH":   "run in hole",
    "TIH":   "trip in hole",
    "TOH":   "trip out of hole",
    "M/U":   "make up",
    "B/O":   "break out",
    "L/O":   "lay down",
    "P/U":   "pick up",
    "L/D":   "lay down",
    "R/U":   "rig up",
    "R/D":   "rig down",
    "F/":    "from",
    "T/":    "to",

    # Equipment
    "BHA":    "bottom hole assembly",
    "TDS":    "top drive system",
    "BOP":    "blowout preventer",
    "MWD":    "measurement while drilling",
    "LWD":    "logging while drilling",
    "ADN":    "azimuthal density neutron",
    "ARC":    "array resistivity compensated",
    "PRS":    "pipe racking system",
    "CART":   "casing running tool",
    "HPDR":   "high pressure drilling riser",
    "FAC":    "fishing assembly",
    "PDM":    "positive displacement motor",
    "RSS":    "rotary steerable system",
    "HO":     "hole opener",
    "HWDP":   "heavyweight drill pipe",
    "DP":     "drill pipe",
    "DC":     "drill collar",
    "RA":     "radioactive",
    "TBC":    "tie back connector",
    "FLX":    "flex packer",
    "XO":     "crossover",
    "LCM":    "lost circulation material",

    # Mud / fluid
    "OBM":    "oil based mud",
    "WBM":    "water based mud",
    "HPWBM":  "high performance water based mud",
    "SOBM":   "synthetic oil based mud",
    "SBM":    "synthetic based mud",
    "SW":     "sea water",

    # Measurements / parameters
    "WOB":    "weight on bit",
    "ROP":    "rate of penetration",
    "ECD":    "equivalent circulating density",
    "EMW":    "equivalent mud weight",
    "SPP":    "stand pipe pressure",
    "MD":     "measured depth",
    "TVD":    "true vertical depth",
    "TD":     "total depth",
    "RKB":    "rotary kelly bushing",
    "MSL":    "mean sea level",

    # Conditions / flags
    "HPHT":   "high pressure high temperature",
    "FIT":    "formation integrity test",
    "LOT":    "leak off test",
    "DD":     "directional drilling",

    # Units
    "lpm":    "litres per minute",
    "rpm":    "revolutions per minute",
    "kNm":    "kilonewton meter",
    "ppm":    "parts per million",
    "MT":     "metric ton",
    "sg":     "specific gravity",
    "ppg":    "pounds per gallon",
    "klbs":   "kilopounds",
}


# Sorted longest-first so e.g. "HWDP" replaces before "DP".
_ABBREV_PATTERNS = sorted(
    ABBREVIATIONS.items(),
    key=lambda kv: -len(kv[0]),
)


def expand_abbreviations(text: str) -> str:
    """
    Insert the long form after each abbreviation, in parentheses.

    Both the original abbreviation and the expansion remain in the string,
    so the bag-of-words contains both forms.
    """
    if not text:
        return text or ""

    out = text
    for abbr, expansion in _ABBREV_PATTERNS:
        # For purely alphanumeric abbreviations use \b word boundaries.
        # For ones with slashes (L/O, F/, etc.) use a manual non-alnum boundary.
        if abbr.isalnum():
            pattern = rf"\b{re.escape(abbr)}\b"
        else:
            pattern = rf"(?<![A-Za-z0-9]){re.escape(abbr)}(?![A-Za-z0-9])"

        out = re.sub(
            pattern,
            f"{abbr} ({expansion})",
            out,
            flags=re.IGNORECASE,
        )
    return out


# =============================================================================
# 2. Query cleanup (specific to the noise in nds_events.xlsx)
# =============================================================================

# Specific typos seen in nds_events.xlsx. Conservative -- only fix what we KNOW.
_QUERY_TYPO_FIXES: list[tuple[str, str]] = [
    ("ccumulation",     "accumulation"),
    ("ecountered",      "encountered"),
    ("encoutered",      "encountered"),
    ("excesive",        "excessive"),
]

# Run of 2+ double-quotes (Excel escape mess) gets collapsed to a single one.
_EXCEL_QUOTE_RX = re.compile(r'"{2,}')

# Duplicate consecutive words: "to to 0.16" becomes "to 0.16"
_DUP_WORD_RX = re.compile(r"\b(\w+)(\s+\1)+\b", flags=re.IGNORECASE)

_MULTI_WS_RX = re.compile(r"\s+")


def clean_query(text: str) -> str:
    """Clean an NDS event description before matching."""
    if not text:
        return text or ""
    out = str(text)
    out = _EXCEL_QUOTE_RX.sub('"', out)
    for bad, good in _QUERY_TYPO_FIXES:
        out = re.sub(rf"\b{re.escape(bad)}\b", good, out, flags=re.IGNORECASE)
    out = _DUP_WORD_RX.sub(r"\1", out)
    out = _MULTI_WS_RX.sub(" ", out).strip()
    return out


# =============================================================================
# 3. Public combined helpers (used by indexes.py and matcher.py)
# =============================================================================

def normalize_for_indexing(text: str) -> str:
    """Applied to corpus op_text strings before fitting TF-IDF / SBERT."""
    if not text:
        return text or ""
    return expand_abbreviations(text)


def normalize_for_query(text: str) -> str:
    """Applied to NDS event text before passing to a query method."""
    return expand_abbreviations(clean_query(text))
