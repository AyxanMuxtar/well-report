"""
Lightweight entity extractor for the matching reranker.

The reranker (in matcher.py) compares entities between the NDS event text and
each candidate operation. Candidates that share a hard-to-fake entity (a bit
size like 26", a specific depth, a problem keyword) get a score boost.

This is a simpler, narrower version of what Task 3 NER will produce. We extract
just what's useful for ranking, not the full entity zoo.

Extracted entity types:
    bit_sizes    : { '26"', '17.5"', '12.25"', '8.5"', ... }       set[str]
    depths_md    : { 1369, 1002, 958, ... }                          set[int]
    problem_kw   : { 'stuck', 'tight', 'pack-off', ... }              set[str]
    equipment    : { 'BHA', 'BOP', 'TDS', ... }                       set[str]
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field

from src.task2_matching.text_normalization import ABBREVIATIONS


# =============================================================================
# Bit sizes (and other inch-denoted hardware diameters)
# =============================================================================
# Match formats like:  26"   17 1/2"   12 1/4"   8.5"   9-5/8"
# Captures the size in inches as a normalized float string.

_BIT_SIZE_RX = re.compile(
    r"""
    (?<![A-Za-z0-9])                    # left boundary: not in middle of a word/number
    (?P<whole>\d{1,3})                  # whole inches
    (?:                                 # optional fractional part
        \s+(?P<frac_num>\d{1,3})        # space + fraction numerator (e.g. "17 1/2")
        \s*[/-]\s*
        (?P<frac_den>\d{1,3})           # fraction denominator
        |
        [.,](?P<dec>\d{1,3})            # OR decimal: 17.5, 12.25
    )?
    \s*
    (?:                                 # required inch unit
        "                               #   "
        | ''                            #   ''
        | \s*in(?:ch(?:es)?)?\b         #   in / inch / inches  (with word boundary)
    )
    """,
    re.VERBOSE | re.IGNORECASE,
)


def _normalize_inch(whole: str, frac_num: str | None, frac_den: str | None, dec: str | None) -> str:
    """Return the canonical inch string, e.g. '26"', '17.5"', '12.25"'."""
    w = int(whole)
    if frac_num and frac_den and int(frac_den) != 0:
        val = w + int(frac_num) / int(frac_den)
    elif dec:
        val = float(f"{w}.{dec}")
    else:
        val = float(w)
    # Format: integer if no fractional part, else two decimals (trim trailing zeros)
    if val == int(val):
        return f'{int(val)}"'
    return f'{val:g}"'


def extract_bit_sizes(text: str) -> set[str]:
    """Extract canonical bit/hole sizes from text. Filters implausible values."""
    if not text:
        return set()
    out = set()
    for m in _BIT_SIZE_RX.finditer(text):
        size_str = _normalize_inch(
            m.group("whole"),
            m.group("frac_num"),
            m.group("frac_den"),
            m.group("dec"),
        )
        # Plausibility filter: drilling sizes are typically 4 - 36 inches
        try:
            val = float(size_str.rstrip('"'))
            if 4 <= val <= 40:
                out.add(size_str)
        except ValueError:
            pass
    return out


# =============================================================================
# Depths in metres MD
# =============================================================================
# Match patterns like:  958 m   1369m   1369 m MD   2447 mMD   2,591 m
# We're tolerant about the unit but require it to be present (else any number
# would match — which gives noisy results).

_DEPTH_RX = re.compile(
    r"""
    \b(?P<val>\d{2,5}(?:[.,]\d{1,3})?)   # 50 to 99999 with optional decimal
    \s*
    (?:m\s*MD|mMD|m\s+MD|m)              # metres unit
    \b
    """,
    re.VERBOSE | re.IGNORECASE,
)


def extract_depths(text: str) -> set[int]:
    """Extract depth values in metres MD as a set of ints (rounded)."""
    if not text:
        return set()
    out = set()
    for m in _DEPTH_RX.finditer(text):
        v = m.group("val").replace(",", ".")
        try:
            n = int(round(float(v)))
            if 1 <= n <= 10000:   # plausibility
                out.add(n)
        except ValueError:
            pass
    return out


# =============================================================================
# Problem / failure keywords (drilling-specific)
# =============================================================================
# A small curated lexicon. We match whole-word, case-insensitive. These are
# the high-signal terms that should boost a match when shared.

PROBLEM_KEYWORDS = {
    "stuck", "stuck pipe", "differential stuck", "differential",
    "tight hole", "tight", "tight spot",
    "pack off", "pack-off", "packoff", "packed off",
    "kick", "influx",
    "loss", "losses", "lost circulation", "fluid loss", "lost returns",
    "twist off", "twist-off", "twistoff",
    "fishing", "fish",
    "cuttings",
    "swabbing", "swab", "surge",
    "leak", "leaking",
    "stall", "stalled",
    "wash out", "washout", "washed out",
    "key seat", "keyseat",
    "back off", "backed off",
    "ballooning",
    "overpull", "over pull", "over-pull",
    "accumulation", "accumulated",
    "clay", "shale",
    "wellbore instability", "instability",
}

# Pre-compile a single regex that finds any of these
_PROBLEM_RX = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in sorted(PROBLEM_KEYWORDS, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)


def extract_problem_keywords(text: str) -> set[str]:
    """Extract the canonical (lower-cased) problem keywords found in text."""
    if not text:
        return set()
    return {m.group(1).lower() for m in _PROBLEM_RX.finditer(text)}


# =============================================================================
# Equipment names from the abbreviation dictionary
# =============================================================================
# We treat all the alphanumeric-only abbreviations as equipment keywords.
# This catches BHA, BOP, TDS, MWD, ARC, ADN, etc.

_EQUIPMENT_TOKENS = {
    abbr for abbr in ABBREVIATIONS.keys()
    if abbr.isalnum() and len(abbr) >= 2
}

_EQUIPMENT_RX = re.compile(
    r"\b(" + "|".join(re.escape(t) for t in sorted(_EQUIPMENT_TOKENS, key=len, reverse=True)) + r")\b"
)


def extract_equipment(text: str) -> set[str]:
    """Extract equipment abbreviations present in the text (case-sensitive match)."""
    if not text:
        return set()
    return {m.group(1) for m in _EQUIPMENT_RX.finditer(text)}


# =============================================================================
# Combined entity bundle
# =============================================================================

@dataclass
class Entities:
    bit_sizes:  set[str] = field(default_factory=set)
    depths_md:  set[int] = field(default_factory=set)
    problem_kw: set[str] = field(default_factory=set)
    equipment:  set[str] = field(default_factory=set)

    def overlap_score(self, other: "Entities", *,
                      w_bit: float = 0.40,
                      w_depth: float = 0.10,
                      w_problem: float = 0.30,
                      w_equipment: float = 0.20) -> float:
        """
        Compute a weighted Jaccard-style overlap with another Entities object.

        Each component contributes a Jaccard if both sides have any entities of
        that type, otherwise 0. Sum of weights is 1.0.
        """
        def jacc(a: set, b: set) -> float:
            if not a or not b:
                return 0.0
            inter = len(a & b)
            union = len(a | b)
            return inter / union if union else 0.0

        return (
            w_bit       * jacc(self.bit_sizes,  other.bit_sizes) +
            w_depth     * jacc(self.depths_md,  other.depths_md) +
            w_problem   * jacc(self.problem_kw, other.problem_kw) +
            w_equipment * jacc(self.equipment,  other.equipment)
        )

    def is_empty(self) -> bool:
        return not (self.bit_sizes or self.depths_md or self.problem_kw or self.equipment)


def extract_entities(text: str) -> Entities:
    """Extract the full entity bundle from a text snippet."""
    return Entities(
        bit_sizes  = extract_bit_sizes(text),
        depths_md  = extract_depths(text),
        problem_kw = extract_problem_keywords(text),
        equipment  = extract_equipment(text),
    )
