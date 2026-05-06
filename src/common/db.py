"""
DuckDB connection helpers.

Usage:
    from src.common.db import get_connection, init_schema

    init_schema()              # drops & recreates all tables (destructive!)
    with get_connection() as con:
        con.execute("SELECT count(*) FROM operations").fetchone()
"""
from __future__ import annotations
from contextlib import contextmanager
from pathlib import Path

import duckdb

from src.common.config import DB_PATH, SCHEMA_SQL
from src.common.logging_utils import get_logger

log = get_logger(__name__)


@contextmanager
def get_connection(read_only: bool = False):
    """Yield a DuckDB connection. Caller does NOT need to commit (DuckDB autocommits)."""
    con = duckdb.connect(str(DB_PATH), read_only=read_only)
    try:
        yield con
    finally:
        con.close()


def init_schema() -> None:
    """Drop all tables and recreate them from schema.sql. DESTRUCTIVE."""
    log.warning("Initializing schema (this drops existing tables): %s", DB_PATH)
    sql = Path(SCHEMA_SQL).read_text(encoding="utf-8")
    with get_connection() as con:
        con.execute(sql)
    log.info("Schema initialized.")


def table_count(table: str) -> int:
    with get_connection(read_only=True) as con:
        result = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
        return int(result[0]) if result else 0


def db_summary() -> dict[str, int]:
    """Return row counts for every table. Useful for smoke tests."""
    tables = [
        "reports", "operations", "drilling_fluid", "pore_pressure",
        "survey_station", "lithology", "gas_reading", "parse_errors",
    ]
    with get_connection(read_only=True) as con:
        out = {}
        for t in tables:
            try:
                row = con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()
                out[t] = int(row[0]) if row else 0
            except duckdb.CatalogException:
                out[t] = -1   # table missing
        return out
