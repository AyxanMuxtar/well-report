"""
NDS event -> operations matcher with four ranking methods:

    tfidf     : sklearn TF-IDF cosine similarity
    sbert     : all-MiniLM-L6-v2 cosine similarity
    hybrid    : 0.5 * normalize(tfidf) + 0.5 * normalize(sbert)
    reranked  : 0.7 * hybrid + 0.3 * entity_overlap

For each event in nds_events:
    - If the event's well has PDFs in the corpus (F-10, F-11, F-12):
        match_type = "strict_same_well"
        scope = operations where well_family == event.well_family
    - If the event's well has NO PDFs (F-13):
        match_type = "cross_well_fallback"
        scope = entire corpus

The hybrid blend protects against either base method's blind spots.
The reranker uses extracted entities (bit sizes, equipment, problem keywords,
depths) to boost candidates that share hard-to-fake tokens with the query.
"""
from __future__ import annotations
import math
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.logging_utils import get_logger
from src.task2_matching.indexes import TfidfIndex, SbertIndex
from src.task2_matching.entities import extract_entities, Entities
from src.task2_matching.text_normalization import normalize_for_query

log = get_logger(__name__)


# Output column order for the matches CSV
OUTPUT_COLUMNS = [
    "event_id", "nds_well", "nds_event_text", "match_type", "match_method",
    "rank", "similarity_score",
    "tfidf_score", "sbert_score", "entity_overlap",
    "matched_op_id", "matched_pdf", "matched_well_family", "matched_well_prefix",
    "matched_report_date", "matched_start_time", "matched_end_time", "matched_end_depth_md",
    "matched_main_activity", "matched_sub_activity", "matched_state", "matched_remark",
    "matched_op_text",
]

# Width of the candidate pool we score with all methods. Top_k matches are
# selected from this pool. Wider = better recall but more compute.
_POOL_K = 100

# Blend weights
_HYBRID_TFIDF_W   = 0.5
_HYBRID_SBERT_W   = 0.5
_RERANK_HYBRID_W  = 0.7
_RERANK_ENTITY_W  = 0.3


# =============================================================================
# Helpers
# =============================================================================

def _candidate_op_ids(corpus: pd.DataFrame, well_family: str | None) -> set[int] | None:
    """Strict-scope (specific well) or full-corpus (None) candidate op_ids."""
    if well_family is None:
        return None
    mask = corpus["well_family"] == well_family
    return set(corpus.loc[mask, "op_id"].tolist())


def _minmax(values: dict[int, float]) -> dict[int, float]:
    """Min-max normalize a dict of {op_id: score} into [0, 1]."""
    if not values:
        return {}
    vs = list(values.values())
    lo, hi = min(vs), max(vs)
    if hi - lo < 1e-12:
        # All scores equal -> all become 0.5 (neutral).
        return {k: 0.5 for k in values}
    return {k: (v - lo) / (hi - lo) for k, v in values.items()}


def _topk(scored: dict[int, float], k: int) -> list[tuple[int, float]]:
    """Return the top-k (op_id, score) pairs from a dict, descending by score."""
    return sorted(scored.items(), key=lambda kv: -kv[1])[:k]


# =============================================================================
# Per-event scoring
# =============================================================================

def _score_one_event(
    event: pd.Series,
    corpus: pd.DataFrame,
    corpus_lookup: dict,
    op_id_to_entities: dict[int, Entities],
    tfidf: TfidfIndex,
    sbert: SbertIndex,
    *,
    top_k: int,
) -> dict[str, list[tuple[int, float, dict]]]:
    """
    Run all four scoring methods for a single event.

    Returns:
        {
            "tfidf":    [(op_id, score, components_dict), ...],
            "sbert":    [(op_id, score, components_dict), ...],
            "hybrid":   [(op_id, score, components_dict), ...],
            "reranked": [(op_id, score, components_dict), ...],
        }
        where each list is sorted descending by score and length top_k.
        components_dict has all 3 raw component scores for the row, useful
        in the output CSV: {tfidf, sbert, entity}.
    """
    well_family = event["well_family"]
    has_corpus = bool(event["has_pdf_corpus"])

    if has_corpus:
        cand = _candidate_op_ids(corpus, well_family)
    else:
        cand = None  # full corpus

    # Pool sizes -- pull a wider top-K from each base method so we have a good
    # candidate set to rerank.
    pool_k = max(_POOL_K, top_k)

    tfidf_hits = tfidf.query(event["event_text"], candidate_op_ids=cand, top_k=pool_k)
    sbert_hits = sbert.query(event["event_text"], candidate_op_ids=cand, top_k=pool_k)

    tfidf_dict: dict[int, float] = {oid: s for oid, s in tfidf_hits}
    sbert_dict: dict[int, float] = {oid: s for oid, s in sbert_hits}

    # Union of all candidates seen by either method.
    union_ids = set(tfidf_dict) | set(sbert_dict)
    if not union_ids:
        empty = {"tfidf": [], "sbert": [], "hybrid": [], "reranked": []}
        return empty

    # Min-max normalize each method's scores within this pool so they're on
    # the same [0, 1] scale before blending. Methods may have very different
    # natural score ranges (TF-IDF cosine often ~0.1-0.3, SBERT often 0.3-0.7).
    tfidf_norm = _minmax({oid: tfidf_dict.get(oid, 0.0) for oid in union_ids})
    sbert_norm = _minmax({oid: sbert_dict.get(oid, 0.0) for oid in union_ids})

    # Hybrid score
    hybrid_scores = {
        oid: _HYBRID_TFIDF_W * tfidf_norm[oid] + _HYBRID_SBERT_W * sbert_norm[oid]
        for oid in union_ids
    }

    # Entity-overlap rerank
    event_ents = extract_entities(normalize_for_query(event["event_text"]))
    if event_ents.is_empty():
        entity_overlap = {oid: 0.0 for oid in union_ids}
    else:
        entity_overlap = {
            oid: event_ents.overlap_score(op_id_to_entities.get(oid, Entities()))
            for oid in union_ids
        }

    # Reranked score = hybrid + entity overlap (entity overlap already in [0,1])
    reranked_scores = {
        oid: _RERANK_HYBRID_W * hybrid_scores[oid] + _RERANK_ENTITY_W * entity_overlap[oid]
        for oid in union_ids
    }

    def pack(scored_dict: dict[int, float]) -> list[tuple[int, float, dict]]:
        out = []
        for oid, score in _topk(scored_dict, top_k):
            out.append((
                oid,
                score,
                {
                    "tfidf":  float(tfidf_dict.get(oid, 0.0)),
                    "sbert":  float(sbert_dict.get(oid, 0.0)),
                    "entity": float(entity_overlap.get(oid, 0.0)),
                },
            ))
        return out

    return {
        "tfidf":    pack(tfidf_dict),         # raw scores, ranked by raw tfidf
        "sbert":    pack(sbert_dict),         # raw scores, ranked by raw sbert
        "hybrid":   pack(hybrid_scores),
        "reranked": pack(reranked_scores),
    }


def _enrich_match_rows(
    method_hits: list[tuple[int, float, dict]],
    corpus_lookup: dict,
    *,
    event_row: pd.Series,
    match_type: str,
    match_method: str,
) -> list[dict]:
    """Turn (op_id, score, components) tuples into full output rows."""
    out: list[dict] = []
    for rank, (op_id, score, components) in enumerate(method_hits, start=1):
        op = corpus_lookup.get(op_id)
        if op is None:
            continue
        out.append({
            "event_id":              int(event_row["event_id"]),
            "nds_well":              event_row["well_raw"],
            "nds_event_text":        event_row["event_text"],
            "match_type":            match_type,
            "match_method":          match_method,
            "rank":                  rank,
            "similarity_score":      round(float(score), 6),
            "tfidf_score":           round(components["tfidf"],  6),
            "sbert_score":           round(components["sbert"],  6),
            "entity_overlap":        round(components["entity"], 6),
            "matched_op_id":         int(op_id),
            "matched_pdf":           op["pdf_path"],
            "matched_well_family":   op["well_family"],
            "matched_well_prefix":   op["well_prefix"],
            "matched_report_date":   op["report_date"],
            "matched_start_time":    op["start_time"],
            "matched_end_time":      op["end_time"],
            "matched_end_depth_md":  op["end_depth_md"],
            "matched_main_activity": op["main_activity"],
            "matched_sub_activity":  op["sub_activity"],
            "matched_state":         op["state"],
            "matched_remark":        op["remark"],
            "matched_op_text":       op["op_text"],
        })
    return out


# =============================================================================
# Public entry point
# =============================================================================

def match_all_events(
    events: pd.DataFrame,
    corpus: pd.DataFrame,
    tfidf: TfidfIndex,
    sbert: SbertIndex,
    *,
    top_k: int = 5,
) -> pd.DataFrame:
    """Produce a long-form DataFrame: one row per (event x method x rank)."""
    # Pre-build lookups
    corpus_lookup = corpus.set_index("op_id").to_dict("index")

    # Pre-extract entities for every operation (so the reranker is fast).
    log.info("Extracting entities for %d operations...", len(corpus))
    op_id_to_entities: dict[int, Entities] = {}
    for op_id, op_text in zip(corpus["op_id"], corpus["op_text"].fillna("")):
        op_id_to_entities[int(op_id)] = extract_entities(op_text)
    log.info("Entity extraction done.")

    rows: list[dict] = []

    for _, event in events.iterrows():
        well_family = event["well_family"]
        has_corpus = bool(event["has_pdf_corpus"])

        if has_corpus:
            match_type = "strict_same_well"
            cand = _candidate_op_ids(corpus, well_family)
            scope_size = len(cand) if cand is not None else 0
            if scope_size == 0:
                log.warning(
                    "Event %d (%s): no operations in scope despite has_pdf_corpus=True. "
                    "Falling back to cross-well.",
                    event["event_id"], event["well_raw"],
                )
                match_type = "cross_well_fallback"
                scope_size = len(corpus)
        else:
            match_type = "cross_well_fallback"
            scope_size = len(corpus)

        log.info(
            "Event %d (%s) -> %s, scope=%d ops",
            event["event_id"], event["well_raw"], match_type, scope_size,
        )

        all_hits = _score_one_event(
            event, corpus, corpus_lookup, op_id_to_entities,
            tfidf, sbert, top_k=top_k,
        )

        for method, hits in all_hits.items():
            rows.extend(_enrich_match_rows(
                hits, corpus_lookup,
                event_row=event, match_type=match_type, match_method=method,
            ))

    matches_df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    log.info("Produced %d match rows for %d events.", len(matches_df), len(events))
    return matches_df


# =============================================================================
# Benchmark / agreement analysis
# =============================================================================

def build_benchmark(matches: pd.DataFrame) -> pd.DataFrame:
    """
    Build a method-comparison table:
    one row per NDS event, with the top-1 from each of the 4 methods side-by-side.
    """
    top1 = matches[matches["rank"] == 1].copy()

    rows = []
    for event_id, sub in top1.groupby("event_id"):
        # Pull the top-1 row for each of the 4 methods
        method_top1 = {}
        for method in ("tfidf", "sbert", "hybrid", "reranked"):
            method_rows = sub[sub["match_method"] == method]
            if len(method_rows):
                method_top1[method] = method_rows.iloc[0]

        if not method_top1:
            continue

        # Use any one as the source of event-level metadata
        ref = next(iter(method_top1.values()))
        out = {
            "event_id":          int(event_id),
            "nds_well":          ref["nds_well"],
            "nds_event_text":    ref["nds_event_text"][:120],
            "match_type":        ref["match_type"],
        }

        for method, r in method_top1.items():
            out[f"{method}_score"] = r["similarity_score"]
            out[f"{method}_pdf"]   = _short_pdf(r["matched_pdf"])
            out[f"{method}_well"]  = r["matched_well_family"]

        # Agreement among the 4 methods on which PDF they picked
        unique_pdfs = {r["matched_pdf"] for r in method_top1.values()}
        unique_ops  = {r["matched_op_id"] for r in method_top1.values()}
        out["n_unique_pdfs"] = len(unique_pdfs)
        out["n_unique_ops"]  = len(unique_ops)
        out["all_methods_agree"] = (len(unique_ops) == 1)

        rows.append(out)
    return pd.DataFrame(rows)


def _short_pdf(path: str | None) -> str | None:
    if path is None:
        return None
    return Path(str(path)).name
