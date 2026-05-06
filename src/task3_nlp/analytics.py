"""
Bonus analytics layered on top of the NER + activity classifier.

Three deliverables:

  1. frequent_failure_patterns()
     For every well, mine recurring problematic events from operation remarks.
     Outputs ranked lists of (failure_type, count, sample_remarks). This is
     the "Layer C" you wanted — generates suggestions for new NDS events.

  2. severity_scores()
     Rank operation remarks by a severity score = weighted sum of:
         - failure activity tags (STUCK_PIPE, EQUIPMENT_FAILURE, WELL_CONTROL...)
         - presence of high-overpull / high-pressure / loss-language
         - state == 'fail'
     This lets reviewers find the "most severe" events per well at a glance.

  3. activity_transition_stats()
     Build a transition matrix: for consecutive operation rows in the same
     report, how often does activity X follow Y? Useful for understanding
     the temporal flow of operations and spotting unusual sequences.
"""
from __future__ import annotations
from collections import Counter, defaultdict
from typing import Iterable

import pandas as pd

from src.common.logging_utils import get_logger
from src.task3_nlp.activities import classify_activity, primary_activity

log = get_logger(__name__)


# =============================================================================
# 1. Frequent failure patterns per well
# =============================================================================

# These activity labels are the "failures" / "incidents" we want to surface.
FAILURE_LABELS = {
    "WELL_CONTROL", "STUCK_PIPE", "TIGHT_HOLE",
    "EQUIPMENT_FAILURE", "FISHING",
}


def frequent_failure_patterns(operations_df: pd.DataFrame,
                              *, top_n: int = 10,
                              min_count: int = 2) -> pd.DataFrame:
    """
    Per well_family, count how often each failure activity appears, and
    sample a few representative remarks.

    Returns long-form DataFrame:
        well_family | failure_label | count | sample_remarks (list[str])
    """
    log.info("Mining frequent failure patterns from %d operation rows.",
             len(operations_df))

    rows: list[dict] = []
    for well, sub in operations_df.groupby("well_family"):
        # Tag every remark
        labels_per_row = sub["remark"].fillna("").map(classify_activity)

        # Count failure-type labels
        label_counts: Counter = Counter()
        sample_remarks: dict[str, list[str]] = defaultdict(list)
        for remark, labels in zip(sub["remark"], labels_per_row):
            for lbl in labels:
                if lbl in FAILURE_LABELS:
                    label_counts[lbl] += 1
                    if remark and len(sample_remarks[lbl]) < 3:
                        sample_remarks[lbl].append(str(remark)[:160])

        for label, count in label_counts.most_common(top_n):
            if count < min_count:
                continue
            rows.append({
                "well_family":     well,
                "failure_label":   label,
                "count":           int(count),
                "sample_remarks":  " || ".join(sample_remarks[label]),
            })

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["well_family", "count"],
                              ascending=[True, False]).reset_index(drop=True)
    return out


# =============================================================================
# 2. Severity scoring
# =============================================================================
# Each ingredient contributes a positive integer; final score is the sum.
# Higher score = more severe.

_SEVERITY_LABEL_WEIGHTS = {
    "WELL_CONTROL":      10,
    "STUCK_PIPE":         8,
    "FISHING":            6,
    "EQUIPMENT_FAILURE":  4,
    "TIGHT_HOLE":         3,
    "REPAIR":             1,
    "WAIT":               1,
}

# Bonus signals from the remark text itself
import re
_SEVERITY_TEXT_SIGNALS = [
    (re.compile(r"\bover\s*pull\b|\boverpull\b",          re.IGNORECASE), 3, "overpull"),
    (re.compile(r"\b(?:loss|lost)\s*(?:of\s*)?circulation",re.IGNORECASE), 4, "lost circulation"),
    (re.compile(r"\bemergency\b",                          re.IGNORECASE), 5, "emergency"),
    (re.compile(r"\btwist[- ]?off\b",                      re.IGNORECASE), 6, "twist-off"),
    (re.compile(r"\bback[- ]?off\b",                       re.IGNORECASE), 5, "back-off"),
    (re.compile(r"\bswab(?:bing)?\b",                      re.IGNORECASE), 2, "swabbing"),
    (re.compile(r"\bfishing\b",                            re.IGNORECASE), 4, "fishing"),
]


def severity_score(remark: str | None, state: str | None = None) -> tuple[int, list[str]]:
    """
    Return (score, contributors) where:
      - score is a non-negative integer
      - contributors lists the factors that contributed (labels + signals)
    """
    if not remark:
        return 0, []

    score = 0
    contribs: list[str] = []

    # Activity-label contributions
    for label in classify_activity(remark, max_labels=5):
        w = _SEVERITY_LABEL_WEIGHTS.get(label, 0)
        if w:
            score += w
            contribs.append(f"{label}(+{w})")

    # Text-signal contributions
    for rx, w, name in _SEVERITY_TEXT_SIGNALS:
        if rx.search(remark):
            score += w
            contribs.append(f"text:{name}(+{w})")

    # State contribution
    if state and str(state).lower() == "fail":
        score += 5
        contribs.append("state:fail(+5)")

    return score, contribs


def severity_scores(operations_df: pd.DataFrame,
                    *, min_score: int = 5) -> pd.DataFrame:
    """
    Score every operation row. Returns rows with score >= min_score, sorted
    descending by score. Useful for slide-worthy "most severe events".
    """
    rows: list[dict] = []
    for _, op in operations_df.iterrows():
        score, contribs = severity_score(op.get("remark"), op.get("state"))
        if score < min_score:
            continue
        rows.append({
            "op_id":          op.get("op_id"),
            "well_family":    op.get("well_family"),
            "pdf_path":       op.get("pdf_path"),
            "report_date":    op.get("report_date"),
            "start_time":     op.get("start_time"),
            "end_depth_md":   op.get("end_depth_md"),
            "main_activity":  op.get("main_activity"),
            "sub_activity":   op.get("sub_activity"),
            "state":          op.get("state"),
            "severity_score": score,
            "severity_contributors": "|".join(contribs),
            "remark":         op.get("remark"),
        })
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    return out.sort_values("severity_score", ascending=False).reset_index(drop=True)


# =============================================================================
# 3. Activity transition statistics
# =============================================================================

def activity_transition_stats(operations_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each PDF, walk operations in op_index order. Build a count table:
        from_activity | to_activity | count
    Aggregated across the entire corpus.

    Useful for spotting:
      - Common operational sequences (DRILL -> CIRCULATE, TRIP_OUT -> LAY_DOWN)
      - Unusual transitions (e.g. DRILL -> STUCK_PIPE indicates an in-drilling issue)
    """
    df = operations_df.dropna(subset=["pdf_path", "remark"]).copy()
    df = df.sort_values(["pdf_path", "op_index"])
    df["primary_label"] = df["remark"].map(primary_activity)

    transitions: Counter = Counter()
    for _, sub in df.groupby("pdf_path"):
        labels = sub["primary_label"].tolist()
        for a, b in zip(labels[:-1], labels[1:]):
            transitions[(a, b)] += 1

    rows = [
        {"from_activity": a, "to_activity": b, "count": n}
        for (a, b), n in transitions.most_common()
    ]
    return pd.DataFrame(rows)
