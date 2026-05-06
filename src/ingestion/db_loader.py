"""
Database loader: persist parsed PDF records into DuckDB.

`load_pdf_records` takes the dict produced by parse_pdf() and inserts:
  - 1 row into reports
  - N rows into operations, drilling_fluid, pore_pressure, survey_station,
    lithology, gas_reading

We use auto-incrementing integer ids tracked in Python (DuckDB doesn't have
AUTOINCREMENT in the SQLite sense — sequences exist but it's simpler to
manage ids in-process during a single ingestion run).
"""
from __future__ import annotations
from typing import Any

import duckdb

from src.common.logging_utils import get_logger

log = get_logger(__name__)


# -----------------------------------------------------------------------------
# In-process id counters. Reset by reset_id_counters() at the start of a run.
# -----------------------------------------------------------------------------
_id_counters = {
    "operations": 0,
    "drilling_fluid": 0,
    "pore_pressure": 0,
    "survey_station": 0,
    "lithology": 0,
    "gas_reading": 0,
}


def reset_id_counters() -> None:
    for k in _id_counters:
        _id_counters[k] = 0


def _next_id(table: str) -> int:
    _id_counters[table] += 1
    return _id_counters[table]


# -----------------------------------------------------------------------------
# Column lists (must match schema.sql)
# -----------------------------------------------------------------------------
_REPORT_COLS = [
    "pdf_path", "pdf_filename", "well_prefix", "well_family", "wellbore_id",
    "report_number", "report_date", "period_start", "period_end", "status",
    "report_creation_time", "days_ahead_behind", "operator", "rig_name",
    "drilling_contractor", "spud_date", "wellbore_type", "date_well_complete",
    "elevation_rkb_msl_m", "water_depth_msl_m", "tight_well", "hpht",
    "temperature", "pressure", "dist_drilled_m", "penetration_rate_mph",
    "hole_dia_in", "pressure_test_type", "formation_strength_gcm3",
    "dia_last_casing", "depth_kickoff_md", "depth_kickoff_tvd",
    "depth_md", "depth_tvd", "plug_back_depth_md",
    "depth_formation_strength_md", "depth_formation_strength_tvd",
    "depth_last_casing_md", "depth_last_casing_tvd",
    "summary_24h", "planned_24h", "parse_quality",
]

_OPERATION_COLS = [
    "op_id", "pdf_path", "well_family", "well_prefix", "report_date",
    "op_index", "start_time", "end_time", "end_depth_md", "main_activity",
    "sub_activity", "state", "remark", "op_text",
]

_FLUID_COLS = [
    "fluid_id", "pdf_path", "well_family", "sample_index",
    "sample_time", "sample_point", "sample_depth_md", "fluid_type",
    "fluid_density_gcm3", "funnel_visc_s", "plastic_visc_mpas",
    "yield_point_pa", "test_temp_hpht_degc",
]

_PORE_COLS = [
    "pp_id", "pdf_path", "well_family", "reading_index",
    "sample_time", "depth_md", "depth_tvd", "equ_mud_weight_gcm3",
    "reading_type",
]

_SURVEY_COLS = [
    "survey_id", "pdf_path", "well_family", "station_index",
    "depth_md", "depth_tvd", "inclination_deg", "azimuth_deg", "comment",
]

_LITHO_COLS = [
    "litho_id", "pdf_path", "well_family", "interval_index",
    "start_depth_md", "end_depth_md", "start_depth_tvd", "end_depth_tvd",
    "shows_description", "lithology_description",
]

_GAS_COLS = [
    "gas_id", "pdf_path", "well_family", "reading_index",
    "sample_time", "gas_class", "depth_top_md", "depth_bottom_md",
    "depth_top_tvd", "depth_bottom_tvd",
    "c1_ppm", "c2_ppm", "c3_ppm", "ic4_ppm", "ic5_ppm",
    "highest_gas_pct", "lowest_gas_pct",
]


def _row_tuple(record: dict, cols: list[str]) -> tuple:
    """Build a tuple in the order of `cols`. Missing keys → None."""
    return tuple(record.get(c) for c in cols)


def _placeholders(n: int) -> str:
    return ",".join(["?"] * n)


def load_pdf_records(con: duckdb.DuckDBPyConnection, parsed: dict) -> None:
    """
    Insert a single parsed PDF's records.

    `parsed` is the dict produced by `parse_pdf()`:
        {
            "report":         {...},                # dict (one row)
            "operations":     [{...}, ...],
            "drilling_fluid": [{...}, ...],
            "pore_pressure":  [{...}, ...],
            "survey_station": [{...}, ...],
            "lithology":      [{...}, ...],
            "gas_reading":    [{...}, ...],
        }
    """
    report = parsed["report"]
    pdf_path = report["pdf_path"]
    well_family = report.get("well_family")
    well_prefix = report.get("well_prefix")
    report_date = report.get("report_date")

    # ---- reports ------------------------------------------------------------
    con.execute(
        f"INSERT INTO reports ({','.join(_REPORT_COLS)}) VALUES ({_placeholders(len(_REPORT_COLS))})",
        _row_tuple(report, _REPORT_COLS),
    )

    # ---- operations ---------------------------------------------------------
    for op in parsed.get("operations", []):
        op["op_id"]       = _next_id("operations")
        op["pdf_path"]    = pdf_path
        op["well_family"] = well_family
        op["well_prefix"] = well_prefix
        op["report_date"] = report_date
        con.execute(
            f"INSERT INTO operations ({','.join(_OPERATION_COLS)}) VALUES ({_placeholders(len(_OPERATION_COLS))})",
            _row_tuple(op, _OPERATION_COLS),
        )

    # ---- drilling_fluid -----------------------------------------------------
    for fluid in parsed.get("drilling_fluid", []):
        fluid["fluid_id"]    = _next_id("drilling_fluid")
        fluid["pdf_path"]    = pdf_path
        fluid["well_family"] = well_family
        con.execute(
            f"INSERT INTO drilling_fluid ({','.join(_FLUID_COLS)}) VALUES ({_placeholders(len(_FLUID_COLS))})",
            _row_tuple(fluid, _FLUID_COLS),
        )

    # ---- pore_pressure ------------------------------------------------------
    for pp in parsed.get("pore_pressure", []):
        pp["pp_id"]       = _next_id("pore_pressure")
        pp["pdf_path"]    = pdf_path
        pp["well_family"] = well_family
        con.execute(
            f"INSERT INTO pore_pressure ({','.join(_PORE_COLS)}) VALUES ({_placeholders(len(_PORE_COLS))})",
            _row_tuple(pp, _PORE_COLS),
        )

    # ---- survey_station -----------------------------------------------------
    for s in parsed.get("survey_station", []):
        s["survey_id"]   = _next_id("survey_station")
        s["pdf_path"]    = pdf_path
        s["well_family"] = well_family
        con.execute(
            f"INSERT INTO survey_station ({','.join(_SURVEY_COLS)}) VALUES ({_placeholders(len(_SURVEY_COLS))})",
            _row_tuple(s, _SURVEY_COLS),
        )

    # ---- lithology ----------------------------------------------------------
    for l in parsed.get("lithology", []):
        l["litho_id"]    = _next_id("lithology")
        l["pdf_path"]    = pdf_path
        l["well_family"] = well_family
        con.execute(
            f"INSERT INTO lithology ({','.join(_LITHO_COLS)}) VALUES ({_placeholders(len(_LITHO_COLS))})",
            _row_tuple(l, _LITHO_COLS),
        )

    # ---- gas_reading --------------------------------------------------------
    for g in parsed.get("gas_reading", []):
        g["gas_id"]      = _next_id("gas_reading")
        g["pdf_path"]    = pdf_path
        g["well_family"] = well_family
        con.execute(
            f"INSERT INTO gas_reading ({','.join(_GAS_COLS)}) VALUES ({_placeholders(len(_GAS_COLS))})",
            _row_tuple(g, _GAS_COLS),
        )


def log_parse_error(con: duckdb.DuckDBPyConnection, pdf_path: str, stage: str, err: Exception) -> None:
    con.execute(
        "INSERT INTO parse_errors (pdf_path, stage, error_message) VALUES (?, ?, ?)",
        (pdf_path, stage, f"{type(err).__name__}: {err}"),
    )
