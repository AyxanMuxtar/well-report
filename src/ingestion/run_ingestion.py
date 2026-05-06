"""
Ingestion orchestrator.

Walks RAW_PDF_DIR, parses every PDF, and loads results into DuckDB.
Per-PDF errors are caught and recorded in the parse_errors table — one bad
PDF must never abort the whole run.

Run:
    python -m src.ingestion.run_ingestion
"""
from __future__ import annotations
import sys
from pathlib import Path

from tqdm import tqdm

from src.common.config import RAW_PDF_DIR
from src.common.db import get_connection, init_schema, db_summary
from src.common.logging_utils import get_logger
from src.ingestion.parse_pdf import parse_pdf
from src.ingestion.db_loader import (
    load_pdf_records, log_parse_error, reset_id_counters,
)

log = get_logger(__name__)


def list_pdfs(root: Path) -> list[Path]:
    pdfs = sorted(p for p in root.glob("*.pdf"))
    return pdfs


def run_ingestion(reset_db: bool = True) -> dict:
    """
    Parse every PDF in RAW_PDF_DIR and load into DuckDB.
    Returns a summary dict of row counts per table.
    """
    log.info("Ingestion starting. PDF source: %s", RAW_PDF_DIR)

    pdfs = list_pdfs(RAW_PDF_DIR)
    log.info("Found %d PDFs.", len(pdfs))
    if not pdfs:
        log.error("No PDFs found in %s. Did you copy them in?", RAW_PDF_DIR)
        return {}

    if reset_db:
        init_schema()
    reset_id_counters()

    n_ok = 0
    n_err = 0
    with get_connection() as con:
        # Wrap inserts in a single transaction for speed
        con.execute("BEGIN TRANSACTION")
        try:
            for pdf in tqdm(pdfs, desc="parsing PDFs", unit="pdf"):
                try:
                    parsed = parse_pdf(pdf)
                except Exception as e:
                    log.exception("Parse failed: %s", pdf.name)
                    log_parse_error(con, str(pdf), "parse", e)
                    n_err += 1
                    continue

                try:
                    load_pdf_records(con, parsed)
                    n_ok += 1
                except Exception as e:
                    log.exception("DB load failed: %s", pdf.name)
                    log_parse_error(con, str(pdf), "load", e)
                    n_err += 1

            con.execute("COMMIT")
        except Exception:
            con.execute("ROLLBACK")
            raise

    summary = db_summary()
    log.info("Ingestion complete. OK=%d, Errors=%d.", n_ok, n_err)
    log.info("Row counts: %s", summary)
    return summary


if __name__ == "__main__":
    run_ingestion(reset_db=True)
