"""
Task 3c — TF-IDF keyword extraction per report.

For each PDF (one document = all remarks + summaries from that report),
extract the top-N keywords that distinguish it from the rest of the corpus.

This is what surfaces *what makes each report's language unique* — useful for
spotting unusual events, well-specific issues, and rare equipment usage.

We:
  1. Aggregate per-PDF text by concatenating: summary_24h + planned_24h +
     all operation remarks for that report.
  2. Apply a stop-word list that includes drilling-domain "boilerplate"
     (rpm, lpm, bar, m, MD, well, hole, drilling, etc.) — these dominate
     every report and aren't informative.
  3. Fit TF-IDF with word + bigram features.
  4. For each report, return the top-N tokens by TF-IDF score.
"""
from __future__ import annotations
from typing import Iterable

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from src.common.logging_utils import get_logger

log = get_logger(__name__)


# Domain stop words on top of the english stoplist. These words are so common
# in drilling reports that they wash out every TF-IDF score otherwise.
DOMAIN_STOPWORDS = {
    # generic operation-mode words
    "drilling", "drilled", "operations", "operation", "well", "hole",
    "bha", "ok", "fail",
    # units and measurements
    "rpm", "lpm", "bar", "psi", "mt", "kn", "knm", "ppg", "deg", "degc",
    "kg", "ml", "ft", "spm", "ppm",
    # depth indicators
    "md", "tvd", "mmd", "mtvd", "depth", "depths", "mmd",
    # numeric/positional
    "m", "mm", "cm", "km",
    # generic verbs
    "made", "make", "set", "pull", "pulled", "pulls", "ran", "run", "running",
    "pump", "pumps", "pumped", "pumping", "use", "used", "using",
    "above", "below", "after", "prior", "while", "during",
    # filler
    "performed", "continued", "started", "finished", "complete", "completed",
}


def _aggregate_report_text(operations_df: pd.DataFrame,
                           reports_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build one text document per PDF by concatenating:
        summary_24h
        + planned_24h
        + every operation row's remark (joined with newlines)
    Returns DataFrame with columns: pdf_path, well_family, doc_text, n_ops.
    """
    # Concatenate operation remarks per pdf_path
    op_text_by_pdf = (
        operations_df
        .dropna(subset=["remark"])
        .groupby("pdf_path", as_index=False)
        .agg(op_remarks=("remark", lambda s: "\n".join(map(str, s))),
             n_ops=("remark", "size"))
    )

    docs = reports_df[["pdf_path", "pdf_filename", "well_family",
                       "summary_24h", "planned_24h"]].copy()
    docs = docs.merge(op_text_by_pdf, on="pdf_path", how="left")
    docs["op_remarks"] = docs["op_remarks"].fillna("")
    docs["summary_24h"] = docs["summary_24h"].fillna("")
    docs["planned_24h"] = docs["planned_24h"].fillna("")
    docs["n_ops"] = docs["n_ops"].fillna(0).astype(int)

    docs["doc_text"] = (
        docs["summary_24h"] + "\n"
        + docs["planned_24h"] + "\n"
        + docs["op_remarks"]
    ).str.strip()
    return docs[["pdf_path", "pdf_filename", "well_family", "doc_text", "n_ops"]]


def extract_top_keywords(operations_df: pd.DataFrame,
                         reports_df: pd.DataFrame,
                         *,
                         top_n: int = 10,
                         ngram_range: tuple[int, int] = (1, 2),
                         min_df: int = 3,
                         max_df: float = 0.6) -> pd.DataFrame:
    """
    Compute top-N TF-IDF keywords per report.

    Returns a long-form DataFrame:
        pdf_path | pdf_filename | well_family | rank | keyword | tfidf_score
    """
    docs = _aggregate_report_text(operations_df, reports_df)
    docs = docs[docs["doc_text"].str.len() > 0].reset_index(drop=True)
    log.info("TF-IDF on %d documents (reports with non-empty text).", len(docs))

    # Build the union stoplist: english + domain
    try:
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
        stop_words = list(ENGLISH_STOP_WORDS | DOMAIN_STOPWORDS)
    except ImportError:
        stop_words = list(DOMAIN_STOPWORDS)

    vectorizer = TfidfVectorizer(
        analyzer    = "word",
        ngram_range = ngram_range,
        min_df      = min_df,
        max_df      = max_df,
        lowercase   = True,
        stop_words  = stop_words,
        token_pattern = r"(?u)\b[A-Za-z][A-Za-z\-]{2,}\b",   # alphabetic tokens only, length ≥ 3
        sublinear_tf= True,
    )

    matrix = vectorizer.fit_transform(docs["doc_text"])
    feature_names = vectorizer.get_feature_names_out()
    log.info("TF-IDF matrix: %d × %d, vocab size %d",
             matrix.shape[0], matrix.shape[1], len(feature_names))

    # Per-row top-N. We use the dense slice trick — sparse → dense per row.
    rows: list[dict] = []
    matrix_csr = matrix.tocsr()
    for i in range(matrix_csr.shape[0]):
        row = matrix_csr.getrow(i).toarray().ravel()
        if row.sum() == 0:
            continue
        top_idx = row.argsort()[::-1][:top_n]
        for rank, idx in enumerate(top_idx, start=1):
            score = float(row[idx])
            if score <= 0:
                break
            rows.append({
                "pdf_path":     docs.iloc[i]["pdf_path"],
                "pdf_filename": docs.iloc[i]["pdf_filename"],
                "well_family":  docs.iloc[i]["well_family"],
                "rank":         rank,
                "keyword":      feature_names[idx],
                "tfidf_score":  round(score, 6),
            })

    return pd.DataFrame(rows)


def keywords_per_well(top_keywords_df: pd.DataFrame,
                      *, top_n: int = 20) -> pd.DataFrame:
    """
    Aggregate top-keyword frequency per well_family.
    Useful for the slide deck: 'most distinctive keywords for well 15_9_F_12'.
    """
    counts = (
        top_keywords_df
        .groupby(["well_family", "keyword"], as_index=False)
        .size()
        .rename(columns={"size": "n_reports"})
    )
    counts = counts.sort_values(["well_family", "n_reports"],
                                ascending=[True, False])
    return counts.groupby("well_family").head(top_n).reset_index(drop=True)
