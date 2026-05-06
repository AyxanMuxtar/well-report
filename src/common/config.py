"""
Centralized configuration: paths, constants, well-family mapping.

All other modules import paths and constants from here.
Edit only this file to relocate data, change well groupings, or tune defaults.
"""
from __future__ import annotations
from pathlib import Path

# ---------------------------------------------------------------------------
# Project root (resolved from this file's location, so it's portable)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Data paths
# ---------------------------------------------------------------------------
DATA_DIR = PROJECT_ROOT / "data"
RAW_PDF_DIR = DATA_DIR / "raw_pdfs"
PROCESSED_DIR = DATA_DIR / "processed"
NDS_EVENTS_XLSX = DATA_DIR / "nds_events.xlsx"

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
DB_DIR = PROJECT_ROOT / "db"
DB_PATH = DB_DIR / "drilling.duckdb"
SCHEMA_SQL = PROJECT_ROOT / "src" / "ingestion" / "schema.sql"

# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
TASK2_MATCHES_CSV = OUTPUTS_DIR / "task2_matches.csv"
TASK2_BENCHMARK_CSV = OUTPUTS_DIR / "task2_benchmark.csv"
TASK3_ENTITIES_CSV = OUTPUTS_DIR / "task3_entities.csv"
TASK3_ACTIVITY_TAGS_CSV = OUTPUTS_DIR / "task3_activity_tags.csv"
TASK3_KEYWORDS_CSV = OUTPUTS_DIR / "task3_keywords.csv"
FREQUENT_EVENTS_CSV = OUTPUTS_DIR / "frequent_events_per_well.csv"

# ---------------------------------------------------------------------------
# Cached indexes (rebuilt only when raw data changes)
# ---------------------------------------------------------------------------
TFIDF_INDEX_PKL = PROCESSED_DIR / "tfidf_index.pkl"
SBERT_EMBEDDINGS_NPY = PROCESSED_DIR / "sbert_embeddings.npy"
SBERT_OP_IDS_NPY = PROCESSED_DIR / "sbert_op_ids.npy"

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
SBERT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# ---------------------------------------------------------------------------
# Task 2 — matching parameters
# ---------------------------------------------------------------------------
TOP_K_MATCHES = 5  # top-K matches per NDS event per method

# Well-family map: filename prefix → (display_well_id, family_for_matching)
# F-11 sidetracks (A, B, T2) are pooled together for NDS matching.
# F-15 sidetracks (A) similarly.
# 15/9-19 sidetracks (S, ST2, A, B, BT2) all map to the same exploration well.
WELL_FAMILY_MAP = {
    "15_9_F_10":     "15_9_F_10",
    "15_9_F_11":     "15_9_F_11",
    "15_9_F_11_A":   "15_9_F_11",
    "15_9_F_11_B":   "15_9_F_11",
    "15_9_F_11_T2":  "15_9_F_11",
    "15_9_F_12":     "15_9_F_12",
    "15_9_F_13":     "15_9_F_13",   # no PDFs exist; kept for completeness
    "15_9_F_14":     "15_9_F_14",
    "15_9_F_15":     "15_9_F_15",
    "15_9_F_15_A":   "15_9_F_15",
    "15_9_19_S":     "15_9_19",
    "15_9_19_A":     "15_9_19",
    "15_9_19_B":     "15_9_19",
    "15_9_19_BT2":   "15_9_19",
    "15_9_19_ST2":   "15_9_19",
}

# Wells that have NDS events (must be matched STRICTLY against own reports)
NDS_TARGET_WELLS = {"15_9_F_10", "15_9_F_11", "15_9_F_12", "15_9_F_13"}

# Wells with NO PDFs in the corpus → fall back to aggressive cross-well search
WELLS_WITH_NO_PDFS = {"15_9_F_13"}

# ---------------------------------------------------------------------------
# Numeric placeholders used in the source PDFs to indicate missing values
# ---------------------------------------------------------------------------
MISSING_NUMERIC_SENTINELS = {-999.99, -999.0, -999}

# ---------------------------------------------------------------------------
# Ensure required directories exist on import (idempotent)
# ---------------------------------------------------------------------------
for _d in (DATA_DIR, RAW_PDF_DIR, PROCESSED_DIR, DB_DIR, OUTPUTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)
