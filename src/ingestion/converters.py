"""
Type-conversion helpers used by the section parsers.

The PDFs use:
  - "-999.99" / "-999" / "-999.0" as missing-value sentinels for numerics
  - "Y" / "N" for booleans
  - "YYYY-MM-DD HH:MM" for timestamps, "YYYY-MM-DD" for dates
  - blanks / "None" / "()" for missing strings
"""
from __future__ import annotations
from datetime import datetime, date
from typing import Optional

from src.common.config import MISSING_NUMERIC_SENTINELS

# Strings that mean "missing"
_BLANK_STRINGS = {"", "none", "n/a", "na", "null", "()", "(unknown)", "unknown"}


def to_str(value: str | None) -> Optional[str]:
    if value is None:
        return None
    s = value.strip()
    if not s or s.lower() in _BLANK_STRINGS:
        return None
    return s


def to_float(value: str | None) -> Optional[float]:
    s = to_str(value)
    if s is None:
        return None
    # strip units that sometimes leak in: "1.5 g/cm3" → 1.5
    s = s.replace(",", ".")
    # take first numeric token
    import re
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    if not m:
        return None
    try:
        f = float(m.group(0))
    except ValueError:
        return None
    if f in MISSING_NUMERIC_SENTINELS:
        return None
    return f


def to_int(value: str | None) -> Optional[int]:
    f = to_float(value)
    if f is None:
        return None
    return int(f)


def to_bool_yn(value: str | None) -> Optional[bool]:
    s = to_str(value)
    if s is None:
        return None
    s = s.upper()
    if s.startswith("Y"):
        return True
    if s.startswith("N"):
        return False
    return None


def to_datetime(value: str | None) -> Optional[datetime]:
    s = to_str(value)
    if s is None:
        return None
    import re as _re
    # Collapse any internal whitespace (incl. newlines) to single spaces.
    s = _re.sub(r"\s+", " ", s)
    # Try to extract a datetime pattern from the start, ignoring trailing junk.
    # Order matters: longest pattern first.
    patterns = [
        (r"\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}", "%Y-%m-%d %H:%M:%S"),
        (r"\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}",       "%Y-%m-%d %H:%M"),
        (r"\d{4}-\d{2}-\d{2}",                     "%Y-%m-%d"),
    ]
    for pat, fmt in patterns:
        m = _re.search(pat, s)
        if m:
            try:
                return datetime.strptime(m.group(0), fmt)
            except ValueError:
                continue
    return None


def to_date(value: str | None) -> Optional[date]:
    dt = to_datetime(value)
    return dt.date() if dt else None
