"""
Task 3a — Named Entity Recognition (NER) for drilling reports.

We use a hybrid rule-based approach: regex patterns for measurements/depths/
times, and a curated dictionary for equipment names. This is the right tool
for this domain because:

  - The vocabulary is closed and well-known (a few dozen equipment terms).
  - Numeric entities follow strict patterns (depth = number + 'm MD', RPM = number + 'rpm').
  - We need high precision (a slide-deck table can't have garbage entities).
  - No labeled training data exists for this domain.

Entity types extracted:
    DEPTH         — '2447 m', '1661 m MD', '3520 mTVD'
    EQUIPMENT     — 'BHA', 'BOP', 'TDS', 'FLX packer', 'spear BHA', etc.
    MEASUREMENT   — '4500 lpm', '274 bar', '140 rpm', '20 MT', '25 kNm', '1.5 sg'
    TIME          — '00:00', '03:45', timestamps within remarks
    BIT_SIZE      — '17 1/2"', '26"', '12.25"'

Public entry point:
    extract_entities(text) -> list[Entity]
    extract_entities_for_corpus(con) -> writes to entities table
"""
from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Iterable

# =============================================================================
# Equipment dictionary
# =============================================================================
# Multi-word phrases first so "spear BHA" matches before "BHA" alone.
# Items are stored in canonical form; matching is case-insensitive.

EQUIPMENT_VOCAB: list[str] = [
    # Multi-word / phrasal equipment
    "spear BHA",
    "drillout BHA",
    "drilling BHA",
    "fishing BHA",
    "hole opener BHA",
    "FLX packer",
    "tie back connector",
    "casing running tool",
    "subsea well head",
    "blowout preventer",
    "top drive system",
    "drill pipe",
    "drill collar",
    "heavyweight drill pipe",
    "hole opener",
    "manual slips",
    "hydraulic spider",
    "guide lines",
    "tension ring",
    "centralizer deck",
    "stress jnt",
    "stress joint",
    "RA source",
    "power puls",
    "powerdrive",
    "manriding",
    "trip tank",

    # Single-word abbreviations & technical names
    "BHA", "BOP", "TDS", "PRS", "MWD", "LWD", "ARC", "ADN", "PDM", "RSS",
    "HWDP", "DP", "DC", "XO", "FAC", "TBC", "HPDR", "PRS",
    "LCM", "OBM", "WBM", "HPWBM", "SOBM",
    "CART", "FIT", "LOT", "TDS",
    "shaker", "shakers",
    "mud pump", "mud pumps",
    "manifold",
    "stabilizer",
    "centralizer",
    "scraper",
    "kelly",
    "swivel",
    "tugger",
    "flowline",
    "choke",
    "annular",
    "wellhead",
    "casing",
    "liner",
    "cement",
    "cmt head", "cmt hose",
    "MUD pump",
]

# Build a single regex that matches any equipment term, longest-first.
# Use word boundaries for alphanumeric terms; for multi-word phrases let \b handle it.
def _build_equipment_rx() -> re.Pattern:
    sorted_terms = sorted(EQUIPMENT_VOCAB, key=lambda s: -len(s))
    parts = []
    for term in sorted_terms:
        # Escape regex special chars; phrase terms become space-tolerant
        escaped = re.escape(term)
        parts.append(escaped)
    pattern = r"\b(?:" + "|".join(parts) + r")\b"
    return re.compile(pattern, re.IGNORECASE)


_EQUIPMENT_RX = _build_equipment_rx()


# =============================================================================
# Numeric entity patterns
# =============================================================================

# DEPTH — number + 'm MD' or 'mMD' or 'm TVD' or 'mTVD' or just 'm' after a depth-like number
_DEPTH_RX = re.compile(
    r"""
    (?<![A-Za-z0-9])                  # left boundary
    (?P<value>\d{1,5}(?:[.,]\d{1,3})?)
    \s*
    (?P<unit>m\s*MD|mMD|m\s*TVD|mTVD|m)
    \b
    """,
    re.VERBOSE | re.IGNORECASE,
)

# MEASUREMENT — value + drilling unit. We deliberately exclude 'm' here to
# avoid double-counting depths.
_MEASUREMENT_UNITS = [
    "lpm", "rpm", "bar", "MT", "ton", "tons", "kNm", "kN", "psi", "kg/m3",
    "g/cm3", "sg", "ppg", "klbs", "deg", "degC", "ft", "ft/h", "m/h", "spm",
    "ppm", "%", "GPM",
]
_UNIT_GROUP = "|".join(re.escape(u) for u in sorted(_MEASUREMENT_UNITS, key=len, reverse=True))

_MEASUREMENT_RX = re.compile(
    rf"""
    (?<![A-Za-z0-9.])                 # left boundary
    (?P<value>-?\d{{1,6}}(?:[.,]\d{{1,3}})?)
    \s*
    (?P<unit>{_UNIT_GROUP})
    \b
    """,
    re.VERBOSE | re.IGNORECASE,
)

# TIME — HH:MM (24-hour). Limited to 00-23:00-59 to avoid false positives.
_TIME_RX = re.compile(r"\b(?P<value>(?:[01]\d|2[0-3]):[0-5]\d)\b")

# BIT_SIZE — same as in task2_matching/entities.py but kept here for self-containment
_BIT_SIZE_RX = re.compile(
    r"""
    (?<![A-Za-z0-9])
    (?P<whole>\d{1,3})
    (?:
        \s+(?P<frac_num>\d{1,3})\s*[/-]\s*(?P<frac_den>\d{1,3})
        |
        [.,](?P<dec>\d{1,3})
    )?
    \s*
    (?:"|''|\s*in(?:ch(?:es)?)?\b)
    """,
    re.VERBOSE | re.IGNORECASE,
)


# =============================================================================
# Entity dataclass
# =============================================================================

@dataclass(frozen=True)
class Entity:
    """One extracted entity occurrence."""
    text:        str        # surface form, exactly as in source
    label:       str        # one of: DEPTH, EQUIPMENT, MEASUREMENT, TIME, BIT_SIZE
    value:       str | None # canonical value when available (e.g. 1661.0 for "1661 m MD")
    unit:        str | None # canonical unit when applicable
    char_start:  int
    char_end:    int

    def to_dict(self) -> dict:
        return {
            "text":        self.text,
            "label":       self.label,
            "value":       self.value,
            "unit":        self.unit,
            "char_start":  self.char_start,
            "char_end":    self.char_end,
        }


# =============================================================================
# Entity extractors
# =============================================================================

def _extract_depths(text: str) -> Iterable[Entity]:
    for m in _DEPTH_RX.finditer(text):
        raw_unit = m.group("unit").upper().replace(" ", "")
        # Plausibility filter: depths are 1-9999 m. Skip values < 5 (probably not depths).
        try:
            v = float(m.group("value").replace(",", "."))
        except ValueError:
            continue
        if not (5 <= v <= 9999):
            continue
        canon_unit = "mTVD" if "TVD" in raw_unit else "mMD" if "MD" in raw_unit else "m"
        yield Entity(
            text       = m.group(0),
            label      = "DEPTH",
            value      = f"{v:g}",
            unit       = canon_unit,
            char_start = m.start(),
            char_end   = m.end(),
        )


def _extract_measurements(text: str) -> Iterable[Entity]:
    for m in _MEASUREMENT_RX.finditer(text):
        try:
            v = float(m.group("value").replace(",", "."))
        except ValueError:
            continue
        unit = m.group("unit")
        # Normalise common aliases
        unit_norm = unit
        if unit.lower() in {"ton", "tons"}:
            unit_norm = "MT"
        yield Entity(
            text       = m.group(0),
            label      = "MEASUREMENT",
            value      = f"{v:g}",
            unit       = unit_norm,
            char_start = m.start(),
            char_end   = m.end(),
        )


def _extract_times(text: str) -> Iterable[Entity]:
    for m in _TIME_RX.finditer(text):
        yield Entity(
            text       = m.group(0),
            label      = "TIME",
            value      = m.group("value"),
            unit       = None,
            char_start = m.start(),
            char_end   = m.end(),
        )


def _extract_bit_sizes(text: str) -> Iterable[Entity]:
    for m in _BIT_SIZE_RX.finditer(text):
        whole = int(m.group("whole"))
        if m.group("frac_num") and m.group("frac_den") and int(m.group("frac_den")):
            val = whole + int(m.group("frac_num")) / int(m.group("frac_den"))
        elif m.group("dec"):
            val = float(f"{whole}.{m.group('dec')}")
        else:
            val = float(whole)
        if not (3 <= val <= 40):
            continue   # plausibility for drilling bit sizes
        yield Entity(
            text       = m.group(0),
            label      = "BIT_SIZE",
            value      = f"{val:g}",
            unit       = '"',
            char_start = m.start(),
            char_end   = m.end(),
        )


def _extract_equipment(text: str) -> Iterable[Entity]:
    for m in _EQUIPMENT_RX.finditer(text):
        yield Entity(
            text       = m.group(0),
            label      = "EQUIPMENT",
            value      = m.group(0).upper(),     # canonical form = uppercase
            unit       = None,
            char_start = m.start(),
            char_end   = m.end(),
        )


# =============================================================================
# Public API
# =============================================================================

def extract_entities(text: str | None) -> list[Entity]:
    """
    Extract all entities from a text snippet.

    Returns entities in document order. Overlapping entities (e.g. a DEPTH
    and a MEASUREMENT that both match the same span) are de-duplicated by
    span — DEPTH and BIT_SIZE win over generic MEASUREMENT.
    """
    if not text:
        return []

    all_ents: list[Entity] = []
    all_ents.extend(_extract_depths(text))
    all_ents.extend(_extract_bit_sizes(text))
    all_ents.extend(_extract_measurements(text))
    all_ents.extend(_extract_times(text))
    all_ents.extend(_extract_equipment(text))

    # Deduplicate overlapping spans. Priority order: DEPTH > BIT_SIZE > MEASUREMENT > TIME > EQUIPMENT.
    priority = {"DEPTH": 0, "BIT_SIZE": 1, "MEASUREMENT": 2, "TIME": 3, "EQUIPMENT": 4}
    all_ents.sort(key=lambda e: (e.char_start, priority[e.label], -e.char_end))

    # Walk and drop any entity that overlaps a higher-priority one already kept.
    kept: list[Entity] = []
    for ent in all_ents:
        overlaps = False
        for k in kept:
            if not (ent.char_end <= k.char_start or ent.char_start >= k.char_end):
                # there is an overlap -- if existing is higher priority, skip this one
                if priority[k.label] <= priority[ent.label]:
                    overlaps = True
                    break
        if not overlaps:
            kept.append(ent)

    kept.sort(key=lambda e: e.char_start)
    return kept
