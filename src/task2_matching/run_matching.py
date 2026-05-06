"""
Task 2 orchestrator.

End-to-end:
    1. Load NDS events from xlsx
    2. Load operations corpus from DuckDB
    3. Build (or load cached) TF-IDF + SBERT indexes
    4. Match every event with both methods
    5. Write outputs/task2_matches.csv and outputs/task2_benchmark.csv

Run:
    python -m src.task2_matching.run_matching
or import & call run_matching() from a notebook.
"""
from __future__ import annotations
import pandas as pd

from src.common.config import (
    TASK2_MATCHES_CSV, TASK2_BENCHMARK_CSV, TOP_K_MATCHES, OUTPUTS_DIR,
)
from src.common.logging_utils import get_logger
from src.task2_matching.nds_events import load_nds_events
from src.task2_matching.indexes import get_or_build_indexes
from src.task2_matching.matcher import match_all_events, build_benchmark

log = get_logger(__name__)


def run_matching(force_rebuild_indexes: bool = False, top_k: int = TOP_K_MATCHES) -> dict:
    """Run end-to-end Task 2 and return a dict of summary stats."""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("Task 2 — NDS event matching")
    log.info("=" * 60)

    # 1. Events
    events = load_nds_events()

    # 2. Corpus + indexes
    tfidf, sbert, corpus = get_or_build_indexes(force_rebuild=force_rebuild_indexes)

    # 3. Match
    matches = match_all_events(events, corpus, tfidf, sbert, top_k=top_k)
    matches.to_csv(TASK2_MATCHES_CSV, index=False, encoding="utf-8")
    log.info("Wrote %d match rows → %s", len(matches), TASK2_MATCHES_CSV)

    # 4. Benchmark
    benchmark = build_benchmark(matches)
    benchmark.to_csv(TASK2_BENCHMARK_CSV, index=False, encoding="utf-8")
    log.info("Wrote benchmark (%d events) → %s", len(benchmark), TASK2_BENCHMARK_CSV)

    summary = {
        "n_events":               len(events),
        "n_match_rows":           len(matches),
        "n_strict_events":        int((events["has_pdf_corpus"]).sum()),
        "n_fallback_events":      int((~events["has_pdf_corpus"]).sum()),
        "n_methods":              4,                  # tfidf, sbert, hybrid, reranked
        "top_k":                  top_k,
        "matches_csv":            str(TASK2_MATCHES_CSV),
        "benchmark_csv":          str(TASK2_BENCHMARK_CSV),
        "all_methods_agree":      int(benchmark["all_methods_agree"].sum()) if len(benchmark) else 0,
        "n_benchmark_rows":       len(benchmark),
    }
    log.info("Summary: %s", summary)
    return summary


if __name__ == "__main__":
    run_matching()
