"""
parse_pdf(): the unit of work for one PDF.

Returns a dict with keys:
    report:         dict            (1 row for the reports table)
    operations:     list[dict]
    drilling_fluid: list[dict]
    pore_pressure:  list[dict]
    survey_station: list[dict]
    lithology:      list[dict]
    gas_reading:    list[dict]
"""
from __future__ import annotations
from pathlib import Path

from src.common.wellbore import parse_filename
from src.ingestion.text_extraction import extract_text
from src.ingestion.sections import (
    parse_header,
    parse_summary_24h,
    parse_planned_24h,
    parse_operations,
    parse_drilling_fluid,
    parse_pore_pressure,
    parse_survey_stations,
    parse_lithology,
    parse_gas_readings,
)


def _classify_quality(parsed: dict) -> str:
    """
    Quick heuristic for parse quality:
        full        : has wellbore + at least one operation row extracted
        partial     : has wellbore but no operations (e.g. placeholder reports)
        header_only : missing wellbore (parser failure)
    """
    report = parsed["report"]
    has_wellbore = bool(report.get("wellbore_id"))
    n_ops = len(parsed.get("operations", []))

    if has_wellbore and n_ops > 0:
        return "full"
    if has_wellbore:
        return "partial"
    return "header_only"


def parse_pdf(pdf_path: str | Path) -> dict:
    pdf_path = Path(pdf_path)

    # 1. Extract & normalize text
    text = extract_text(pdf_path)

    # 2. Filename-derived fields
    parsed_name = parse_filename(pdf_path.name)
    if parsed_name:
        well_prefix, well_family, report_date = parsed_name
    else:
        well_prefix, well_family, report_date = (None, None, None)

    # 3. Header
    header = parse_header(text)

    # 4. Free-text summaries
    summary_24h = parse_summary_24h(text)
    planned_24h = parse_planned_24h(text)

    # 5. Tables
    operations  = parse_operations(text)
    fluid       = parse_drilling_fluid(text)
    pore        = parse_pore_pressure(text)
    survey      = parse_survey_stations(text)
    lithology   = parse_lithology(text)
    gas         = parse_gas_readings(text)

    # 6. Assemble report dict
    report: dict = {
        "pdf_path":     str(pdf_path.resolve()),
        "pdf_filename": pdf_path.name,
        "well_prefix":  well_prefix,
        "well_family":  well_family,
        "report_date":  report_date,
        "summary_24h":  summary_24h,
        "planned_24h":  planned_24h,
        **header,
    }

    parsed = {
        "report":         report,
        "operations":     operations,
        "drilling_fluid": fluid,
        "pore_pressure":  pore,
        "survey_station": survey,
        "lithology":      lithology,
        "gas_reading":    gas,
    }
    parsed["report"]["parse_quality"] = _classify_quality(parsed)
    return parsed
