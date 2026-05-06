"""
Helpers for parsing wellbore identifiers from PDF filenames.

PDF filenames follow patterns like:
    15_9_F_15_A_2008_12_16.pdf
    15_9_F_10_2008_11_30.pdf
    15_9_19_ST2_1992_12_19.pdf

We extract:
    - well_prefix: e.g. "15_9_F_15_A" or "15_9_19_ST2"
    - well_family: pooled identifier for matching (sidetracks merged)
    - report_date: YYYY-MM-DD from the trailing date
"""
from __future__ import annotations
import re
from pathlib import Path
from datetime import date
from typing import Optional, Tuple

from src.common.config import WELL_FAMILY_MAP

# Match: <well-prefix>_<YYYY>_<MM>_<DD>.pdf
# well-prefix examples: 15_9_F_15_A   15_9_F_10   15_9_19_ST2
_FILENAME_RX = re.compile(
    r"^(?P<prefix>15_9_(?:F_)?\d+(?:_[A-Z]+\d*)?)_(?P<year>\d{4})_(?P<month>\d{2})_(?P<day>\d{2})\.pdf$",
    re.IGNORECASE,
)


def parse_filename(filename: str | Path) -> Optional[Tuple[str, str, date]]:
    """
    Parse a PDF filename into (well_prefix, well_family, report_date).
    Returns None if the filename does not match the expected pattern.
    """
    name = Path(filename).name
    m = _FILENAME_RX.match(name)
    if not m:
        return None

    prefix = m.group("prefix")
    family = WELL_FAMILY_MAP.get(prefix, prefix)  # fall back to prefix itself

    try:
        report_date = date(
            int(m.group("year")),
            int(m.group("month")),
            int(m.group("day")),
        )
    except ValueError:
        return None

    return prefix, family, report_date


def well_family_from_filename(filename: str | Path) -> Optional[str]:
    """Convenience: just the family, or None on parse failure."""
    parsed = parse_filename(filename)
    return parsed[1] if parsed else None
