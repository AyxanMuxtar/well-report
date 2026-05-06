"""
Task 3 orchestrator.

End-to-end:
    1. Load operations + reports from DuckDB
    2. Run NER on every operation remark + summaries → entities CSV
    3. Run activity classification on every remark → tags CSV
    4. Compute TF-IDF top-N keywords per report → keywords CSV
    5. Bonus: frequent failure patterns per well, severity scores, transitions

Run:
    python -m src.task3_nlp.run_nlp
or call run_nlp() from a notebook.
"""
from __future__ import annotations
import pandas as pd

from src.common.config import (
    OUTPUTS_DIR,
    TASK3_ENTITIES_CSV, TASK3_ACTIVITY_TAGS_CSV, TASK3_KEYWORDS_CSV,
    FREQUENT_EVENTS_CSV,
)
from src.common.db import get_connection
from src.common.logging_utils import get_logger
from src.task3_nlp.ner import extract_entities
from src.task3_nlp.activities import classify_activity, primary_activity
from src.task3_nlp.keywords import extract_top_keywords, keywords_per_well
from src.task3_nlp.analytics import (
    frequent_failure_patterns, severity_scores, activity_transition_stats,
)

log = get_logger(__name__)


# Output paths for the bonus deliverables (not in config.py)
SEVERITY_CSV    = OUTPUTS_DIR / "task3_severity_scores.csv"
TRANSITIONS_CSV = OUTPUTS_DIR / "task3_activity_transitions.csv"
KEYWORDS_PER_WELL_CSV = OUTPUTS_DIR / "task3_keywords_per_well.csv"


# =============================================================================
# Helpers
# =============================================================================

def _load_operations() -> pd.DataFrame:
    with get_connection(read_only=True) as con:
        return con.execute("""
            SELECT op_id, pdf_path, well_family, well_prefix, report_date,
                   op_index, start_time, end_time, end_depth_md,
                   main_activity, sub_activity, state, remark, op_text
            FROM operations
            ORDER BY pdf_path, op_index
        """).df()


def _load_reports() -> pd.DataFrame:
    with get_connection(read_only=True) as con:
        return con.execute("""
            SELECT pdf_path, pdf_filename, well_family, well_prefix,
                   report_date, summary_24h, planned_24h
            FROM reports
        """).df()


# =============================================================================
# Per-task runners
# =============================================================================

def run_ner(operations_df: pd.DataFrame) -> pd.DataFrame:
    """Extract entities from every operation remark. Long-form output."""
    log.info("Running NER on %d operations.", len(operations_df))
    rows: list[dict] = []
    for _, op in operations_df.iterrows():
        ents = extract_entities(op.get("remark"))
        for e in ents:
            rows.append({
                "op_id":         int(op["op_id"]) if pd.notna(op["op_id"]) else None,
                "pdf_path":      op["pdf_path"],
                "well_family":   op["well_family"],
                "report_date":   op["report_date"],
                "entity_text":   e.text,
                "entity_label":  e.label,
                "entity_value":  e.value,
                "entity_unit":   e.unit,
                "char_start":    e.char_start,
                "char_end":      e.char_end,
            })
    out = pd.DataFrame(rows)
    log.info("Extracted %d entities.", len(out))
    return out


def run_activities(operations_df: pd.DataFrame) -> pd.DataFrame:
    """Tag every operation remark with normalized activity labels."""
    log.info("Tagging activities for %d operations.", len(operations_df))
    rows: list[dict] = []
    for _, op in operations_df.iterrows():
        labels = classify_activity(op.get("remark"), max_labels=3)
        rows.append({
            "op_id":            int(op["op_id"]) if pd.notna(op["op_id"]) else None,
            "pdf_path":         op["pdf_path"],
            "well_family":      op["well_family"],
            "report_date":      op["report_date"],
            "main_activity":    op["main_activity"],
            "sub_activity":     op["sub_activity"],
            "state":            op["state"],
            "primary_label":    labels[0],
            "all_labels":       "|".join(labels),
            "remark_preview":   (str(op.get("remark") or "")[:200]),
        })
    return pd.DataFrame(rows)


# =============================================================================
# Public entry point
# =============================================================================

def run_nlp(top_keywords_n: int = 10,
            severity_threshold: int = 5) -> dict:
    """End-to-end Task 3. Returns a summary dict."""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    log.info("=" * 60)
    log.info("Task 3 — NLP analysis")
    log.info("=" * 60)

    operations_df = _load_operations()
    reports_df    = _load_reports()
    log.info("Loaded %d operations, %d reports.", len(operations_df), len(reports_df))

    # 3a — NER
    entities_df = run_ner(operations_df)
    entities_df.to_csv(TASK3_ENTITIES_CSV, index=False, encoding="utf-8")
    log.info("Wrote entities → %s", TASK3_ENTITIES_CSV)

    # 3b — Activity classification
    activities_df = run_activities(operations_df)
    activities_df.to_csv(TASK3_ACTIVITY_TAGS_CSV, index=False, encoding="utf-8")
    log.info("Wrote activity tags → %s", TASK3_ACTIVITY_TAGS_CSV)

    # 3c — TF-IDF keywords per report
    keywords_df = extract_top_keywords(operations_df, reports_df, top_n=top_keywords_n)
    keywords_df.to_csv(TASK3_KEYWORDS_CSV, index=False, encoding="utf-8")
    log.info("Wrote keywords → %s", TASK3_KEYWORDS_CSV)

    # Bonus 1 — frequent failure patterns per well
    failures_df = frequent_failure_patterns(operations_df)
    failures_df.to_csv(FREQUENT_EVENTS_CSV, index=False, encoding="utf-8")
    log.info("Wrote frequent failure patterns → %s", FREQUENT_EVENTS_CSV)

    # Bonus 2 — severity scores
    severity_df = severity_scores(operations_df, min_score=severity_threshold)
    severity_df.to_csv(SEVERITY_CSV, index=False, encoding="utf-8")
    log.info("Wrote severity scores → %s", SEVERITY_CSV)

    # Bonus 3 — activity transitions
    transitions_df = activity_transition_stats(operations_df)
    transitions_df.to_csv(TRANSITIONS_CSV, index=False, encoding="utf-8")
    log.info("Wrote activity transitions → %s", TRANSITIONS_CSV)

    # Bonus 4 — keywords aggregated per well
    kpw_df = keywords_per_well(keywords_df, top_n=20)
    kpw_df.to_csv(KEYWORDS_PER_WELL_CSV, index=False, encoding="utf-8")
    log.info("Wrote keywords per well → %s", KEYWORDS_PER_WELL_CSV)

    summary = {
        "n_operations":    len(operations_df),
        "n_reports":       len(reports_df),
        "n_entities":      len(entities_df),
        "n_activity_rows": len(activities_df),
        "n_keyword_rows":  len(keywords_df),
        "n_failure_rows":  len(failures_df),
        "n_severity_rows": len(severity_df),
        "n_transitions":   len(transitions_df),
        "outputs": {
            "entities":      str(TASK3_ENTITIES_CSV),
            "activities":    str(TASK3_ACTIVITY_TAGS_CSV),
            "keywords":      str(TASK3_KEYWORDS_CSV),
            "failures":      str(FREQUENT_EVENTS_CSV),
            "severity":      str(SEVERITY_CSV),
            "transitions":   str(TRANSITIONS_CSV),
            "keywords_per_well": str(KEYWORDS_PER_WELL_CSV),
        },
    }
    log.info("Summary: %s", summary)
    return summary


if __name__ == "__main__":
    run_nlp()
