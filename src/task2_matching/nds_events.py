"""
Load and normalize NDS events from nds_events.xlsx.

The spreadsheet has two columns:
    Well   : "15/9-F-10", "15/9-F-11", ...
    Event  : free-text description of the problematic operation

We add:
    well_family    : normalized to the same format as operations.well_family
                     (e.g. "15_9_F_10"), used for strict matching scope.
    event_id       : stable integer ID per row (1-indexed).
    has_pdf_corpus : True if any PDFs exist for that well, False for F-13.
"""
from __future__ import annotations
import re
import pandas as pd

from src.common.config import (
    NDS_EVENTS_XLSX, WELLS_WITH_NO_PDFS, NDS_TARGET_WELLS,
)
from src.common.logging_utils import get_logger

log = get_logger(__name__)


def _normalize_well_to_family(well_str: str) -> str:
    """
    Convert "15/9-F-10" → "15_9_F_10" so it matches operations.well_family.
    """
    s = str(well_str).strip()
    s = s.replace("/", "_").replace("-", "_").replace(" ", "_")
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def load_nds_events() -> pd.DataFrame:
    """
    Read nds_events.xlsx, clean column names, and add normalized well_family
    plus a stable event_id.

    Returns a DataFrame with columns:
        event_id, well_raw, well_family, event_text, has_pdf_corpus
    """
    if not NDS_EVENTS_XLSX.exists():
        raise FileNotFoundError(
            f"NDS events file not found at {NDS_EVENTS_XLSX}. "
            f"Place nds_events.xlsx in data/."
        )

    df = pd.read_excel(NDS_EVENTS_XLSX)

    # Find the well and event columns (defensive — column names may vary)
    cols_lower = {c.lower().strip(): c for c in df.columns}
    well_col  = cols_lower.get("well")
    event_col = cols_lower.get("event")
    if well_col is None or event_col is None:
        raise ValueError(
            f"Expected 'Well' and 'Event' columns in {NDS_EVENTS_XLSX.name}; "
            f"found: {list(df.columns)}"
        )

    df = df.rename(columns={well_col: "well_raw", event_col: "event_text"})
    df = df.dropna(subset=["well_raw", "event_text"]).reset_index(drop=True)
    df["event_id"] = range(1, len(df) + 1)
    df["well_family"] = df["well_raw"].map(_normalize_well_to_family)
    df["has_pdf_corpus"] = ~df["well_family"].isin(WELLS_WITH_NO_PDFS)

    df = df[["event_id", "well_raw", "well_family", "event_text", "has_pdf_corpus"]]

    log.info("Loaded %d NDS events.", len(df))
    log.info("Wells in NDS events: %s", sorted(df["well_family"].unique()))

    unknown_wells = set(df["well_family"]) - NDS_TARGET_WELLS
    if unknown_wells:
        log.warning("NDS events reference wells not in NDS_TARGET_WELLS: %s", unknown_wells)

    return df
