"""
Build and cache the two similarity indexes over the operations corpus.

Index 1 — TF-IDF (sklearn): word + bigram, sparse matrix
Index 2 — SBERT (sentence-transformers): all-MiniLM-L6-v2, dense float32

Both indexes are aligned by `op_id`. The matcher (matcher.py) loads them and
queries by either method, optionally restricting to a subset of op_ids.

The corpus text and the query text are both run through
`normalize_for_indexing` / `normalize_for_query` (text_normalization.py) so
abbreviations and noise patterns are aligned across both sides before
similarity is computed.

The indexes are cached to disk (data/processed/) so we don't recompute on every
run. Cache invalidation is by file mtime — if the DB has been updated since
the cache, we rebuild.
"""
from __future__ import annotations
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.common.config import (
    DB_PATH, PROCESSED_DIR,
    TFIDF_INDEX_PKL, SBERT_EMBEDDINGS_NPY, SBERT_OP_IDS_NPY,
    SBERT_MODEL_NAME,
)
from src.common.db import get_connection
from src.common.logging_utils import get_logger
from src.task2_matching.text_normalization import (
    normalize_for_indexing, normalize_for_query,
)

log = get_logger(__name__)


# =============================================================================
# Corpus loader
# =============================================================================

def load_corpus() -> pd.DataFrame:
    """
    Load every operation row from DuckDB along with its text and metadata.
    """
    with get_connection(read_only=True) as con:
        df = con.execute("""
            SELECT op_id, well_family, well_prefix, pdf_path, report_date,
                   start_time, end_time, end_depth_md,
                   main_activity, sub_activity, state, remark, op_text
            FROM operations
            WHERE op_text IS NOT NULL
              AND LENGTH(TRIM(op_text)) > 0
            ORDER BY op_id
        """).df()
    log.info("Loaded corpus: %d operation rows", len(df))
    return df


# =============================================================================
# TF-IDF index
# =============================================================================

@dataclass
class TfidfIndex:
    """A fitted TF-IDF index plus the op_ids it covers."""
    vectorizer: TfidfVectorizer
    matrix: object              # scipy sparse matrix (n_ops × vocab)
    op_ids: np.ndarray          # 1-D int64 array, length n_ops

    def query(self, query_text: str, candidate_op_ids: Optional[set[int]] = None,
              top_k: int = 5) -> list[tuple[int, float]]:
        # Normalize the query the same way the corpus was normalized.
        normalized_query = normalize_for_query(query_text)
        q_vec = self.vectorizer.transform([normalized_query])
        if candidate_op_ids is not None:
            mask = np.isin(self.op_ids, list(candidate_op_ids))
            sub_matrix = self.matrix[mask]
            sub_op_ids = self.op_ids[mask]
        else:
            sub_matrix = self.matrix
            sub_op_ids = self.op_ids

        if sub_matrix.shape[0] == 0:
            return []

        sims = cosine_similarity(q_vec, sub_matrix).flatten()
        if len(sims) <= top_k:
            order = np.argsort(-sims)
        else:
            order = np.argpartition(-sims, top_k)[:top_k]
            order = order[np.argsort(-sims[order])]
        return [(int(sub_op_ids[i]), float(sims[i])) for i in order]


def build_tfidf(corpus: pd.DataFrame) -> TfidfIndex:
    t0 = time.time()
    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        lowercase=True,
        sublinear_tf=True,
    )
    # Normalize each op_text BEFORE fitting so abbreviations are expanded
    # consistently across the whole corpus.
    normalized_texts = corpus["op_text"].fillna("").map(normalize_for_indexing)
    matrix = vectorizer.fit_transform(normalized_texts)
    op_ids = corpus["op_id"].to_numpy(dtype=np.int64)
    log.info("Built TF-IDF: %d docs × %d features in %.1fs",
             matrix.shape[0], matrix.shape[1], time.time() - t0)
    return TfidfIndex(vectorizer=vectorizer, matrix=matrix, op_ids=op_ids)


def save_tfidf(index: TfidfIndex, path: Path = TFIDF_INDEX_PKL) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(index, f, protocol=pickle.HIGHEST_PROTOCOL)
    log.info("Saved TF-IDF index → %s", path)


def load_tfidf(path: Path = TFIDF_INDEX_PKL) -> TfidfIndex:
    with open(path, "rb") as f:
        return pickle.load(f)


# =============================================================================
# SBERT index
# =============================================================================

@dataclass
class SbertIndex:
    """A SBERT embedding matrix plus the op_ids it covers and the model handle."""
    model_name: str
    embeddings: np.ndarray
    op_ids: np.ndarray
    _model: object = None

    def _ensure_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            log.info("Loading SBERT model: %s", self.model_name)
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode_query(self, text: str) -> np.ndarray:
        # Normalize the query so it matches the corpus normalization.
        normalized = normalize_for_query(text)
        model = self._ensure_model()
        v = model.encode([normalized], normalize_embeddings=True, show_progress_bar=False)
        return v.astype(np.float32)[0]

    def query(self, query_text: str, candidate_op_ids: Optional[set[int]] = None,
              top_k: int = 5) -> list[tuple[int, float]]:
        q = self.encode_query(query_text)

        if candidate_op_ids is not None:
            mask = np.isin(self.op_ids, list(candidate_op_ids))
            sub_emb = self.embeddings[mask]
            sub_op_ids = self.op_ids[mask]
        else:
            sub_emb = self.embeddings
            sub_op_ids = self.op_ids

        if sub_emb.shape[0] == 0:
            return []

        sims = sub_emb @ q
        if len(sims) <= top_k:
            order = np.argsort(-sims)
        else:
            order = np.argpartition(-sims, top_k)[:top_k]
            order = order[np.argsort(-sims[order])]
        return [(int(sub_op_ids[i]), float(sims[i])) for i in order]


def build_sbert(corpus: pd.DataFrame, batch_size: int = 256,
                model_name: str = SBERT_MODEL_NAME) -> SbertIndex:
    from sentence_transformers import SentenceTransformer
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("SBERT device: %s", device)

    t0 = time.time()
    model = SentenceTransformer(model_name, device=device)
    # Normalize the same way as the TF-IDF corpus.
    texts = corpus["op_text"].fillna("").map(normalize_for_indexing).tolist()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype(np.float32)
    op_ids = corpus["op_id"].to_numpy(dtype=np.int64)
    log.info("Built SBERT: %d docs × %d dim in %.1fs",
             embeddings.shape[0], embeddings.shape[1], time.time() - t0)

    return SbertIndex(
        model_name=model_name,
        embeddings=embeddings,
        op_ids=op_ids,
        _model=model,
    )


def save_sbert(index: SbertIndex,
               emb_path: Path = SBERT_EMBEDDINGS_NPY,
               ids_path: Path = SBERT_OP_IDS_NPY) -> None:
    emb_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(emb_path, index.embeddings)
    np.save(ids_path, index.op_ids)
    (emb_path.parent / "sbert_model_name.txt").write_text(index.model_name, encoding="utf-8")
    log.info("Saved SBERT index → %s, %s", emb_path, ids_path)


def load_sbert(emb_path: Path = SBERT_EMBEDDINGS_NPY,
               ids_path: Path = SBERT_OP_IDS_NPY) -> SbertIndex:
    embeddings = np.load(emb_path)
    op_ids = np.load(ids_path)
    name_path = emb_path.parent / "sbert_model_name.txt"
    model_name = name_path.read_text(encoding="utf-8").strip() if name_path.exists() else SBERT_MODEL_NAME
    return SbertIndex(model_name=model_name, embeddings=embeddings, op_ids=op_ids)


# =============================================================================
# Cache helpers
# =============================================================================

def _is_cache_fresh(cache_path: Path) -> bool:
    """Cache is fresh iff it exists and is newer than the DB."""
    if not cache_path.exists():
        return False
    if not DB_PATH.exists():
        return False
    return cache_path.stat().st_mtime >= DB_PATH.stat().st_mtime


def get_or_build_indexes(force_rebuild: bool = False) -> tuple[TfidfIndex, SbertIndex, pd.DataFrame]:
    """
    Load both indexes from cache if fresh, otherwise rebuild.
    Returns (tfidf_index, sbert_index, corpus_df).
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    corpus = load_corpus()

    # TF-IDF
    if not force_rebuild and _is_cache_fresh(TFIDF_INDEX_PKL):
        log.info("Loading TF-IDF from cache: %s", TFIDF_INDEX_PKL)
        tfidf = load_tfidf()
    else:
        log.info("Building TF-IDF (no fresh cache)")
        tfidf = build_tfidf(corpus)
        save_tfidf(tfidf)

    # SBERT
    if not force_rebuild and _is_cache_fresh(SBERT_EMBEDDINGS_NPY):
        log.info("Loading SBERT embeddings from cache: %s", SBERT_EMBEDDINGS_NPY)
        sbert = load_sbert()
    else:
        log.info("Building SBERT embeddings (no fresh cache)")
        sbert = build_sbert(corpus)
        save_sbert(sbert)

    return tfidf, sbert, corpus
