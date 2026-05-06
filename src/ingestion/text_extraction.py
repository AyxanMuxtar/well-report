"""
PDF text extraction.

We use pdfplumber to extract raw text. The drilling reports were all generated
with the same template (despite spanning 1992–2018), so we don't need OCR or
multiple parser profiles.

Quirks of the source PDFs we handle here:
  - Doubled headers like "Start Start time time" or "Wellbore: Wellbore:"
    These come from how the original tool rendered table column names.
  - "-999.99" sentinel values for missing numerics.
  - Some lines wrap mid-word with newlines inside cells.

This module returns a single normalized text blob per PDF. Section parsers
(see sections.py) work on that blob.
"""
from __future__ import annotations
import re
from pathlib import Path

import pdfplumber

from src.common.logging_utils import get_logger

log = get_logger(__name__)

# Match a word repeated immediately, e.g. "Start Start" or "Wellbore Wellbore".
# Require at least one letter so we don't collapse "00 00" or "1369 1369".
_DOUBLED_WORD_RX = re.compile(r"\b([A-Za-zÀ-ÿ]\w*)\s+\1\b", flags=re.IGNORECASE)

# Match doubled "Word: Word:" headers (the colon-suffixed variant)
_DOUBLED_LABEL_RX = re.compile(r"\b([A-Za-zÀ-ÿ]\w*):\s+\1:", flags=re.IGNORECASE)

# Match a "word" where every character is doubled, e.g. "WWeellllbboorree" or "PPeerriioodd".
# Must be at least 4 characters total (2 distinct), all letters, with consecutive
# pair-doubling throughout. We use a function-based replacement to verify the
# pattern instead of a complex backtracking regex.
_CANDIDATE_DOUBLED_CHARS_RX = re.compile(r"\b[A-Za-zÀ-ÿ]{4,}\b")


def _collapse_doubled_chars_word(word: str) -> str:
    """
    If `word` consists entirely of doubled-character pairs (case-insensitive),
    return the de-duplicated version. Otherwise return the word unchanged.

    Examples:
        "WWeellllbboorree"  -> "Wellbore"
        "PPeerriioodd"      -> "Period"
        "Mississippi"       -> "Mississippi"   (odd length, won't match anyway)
        "letter"            -> "letter"        (no all-pair doubling)
    """
    if len(word) < 4 or len(word) % 2 != 0:
        return word
    halved = []
    for i in range(0, len(word), 2):
        if word[i].lower() != word[i + 1].lower():
            return word  # not consistently doubled -> bail out
        halved.append(word[i])
    return "".join(halved)


def _collapse_doubled_chars(text: str) -> str:
    """Apply the per-word character-doubling collapse across the whole text."""
    return _CANDIDATE_DOUBLED_CHARS_RX.sub(
        lambda m: _collapse_doubled_chars_word(m.group(0)),
        text,
    )


def _collapse_doubled_tokens(text: str) -> str:
    """
    The PDF rendering doubles header tokens. Collapse them.
    "Start Start time time" -> "Start time"
    "Wellbore: Wellbore:"   -> "Wellbore:"
    Applied iteratively because doubling can be triple in extreme cases.
    NOTE: We deliberately do NOT collapse adjacent colons / hyphens, because
    "23:00 00:00" (two HH:MM stamps) would get mangled into "23:00:00".
    """
    prev = None
    cur = text
    while prev != cur:
        prev = cur
        # First collapse "Label: Label:" patterns since the trailing colon
        # blocks the plain word-doubling regex below.
        cur = _DOUBLED_LABEL_RX.sub(r"\1:", cur)
        cur = _DOUBLED_WORD_RX.sub(r"\1", cur)
    return cur


# Common words that get split mid-token by pdfplumber's line wrap inside cells:
#   "circ ulating" -> "circulating"
#   "Sur vey"      -> "Survey"
#   "POO H"        -> "POOH"
# We can't generically un-split (false-positive risk), but we can fix a known
# vocabulary of drilling/operations terms. Conservative list — extend as needed.
_KNOWN_SPLIT_WORDS = [
    ("circ ulating",              "circulating"),
    ("circ ulation",              "circulation"),
    ("Sur vey",                   "Survey"),
    ("Summar y",                  "Summary"),
    ("POO H",                     "POOH"),
    ("RI H",                      "RIH"),
    ("integrity tes t",           "integrity test"),
    ("formation integrity tes t", "formation integrity test"),
    ("pea k",                     "peak"),
    ("Co ntrolled",               "Controlled"),
    ("recipro cating",            "reciprocating"),
    ("rec iprocating",            "reciprocating"),
    ("conditi oning",             "conditioning"),
    ("k Nm",                      "kNm"),
    ("k N",                       "kN"),
    ("PR S",                      "PRS"),
    ("in specting",               "inspecting"),
    ("activitie s",               "activities"),
    ("plan ned",                  "planned"),
    ("oper ations",               "operations"),
]


def _fix_split_words(text: str) -> str:
    """
    Replace each known split form with its joined form. Matches any
    whitespace (incl. newlines) between the split halves, since pdfplumber
    sometimes inserts a newline mid-word.
    """
    for bad, good in _KNOWN_SPLIT_WORDS:
        # Convert literal space in `bad` to regex \s+
        bad_pattern = r"\s+".join(re.escape(part) for part in bad.split(" "))
        text = re.sub(bad_pattern, good, text)
    return text


def normalize_text(raw_text: str) -> str:
    """
    Apply the same normalization pipeline as extract_text(), but on a raw
    string. Useful for testing parsers with fixture text.
    """
    raw = re.sub(r"[ \t]+", " ", raw_text)
    # Char-level doubling first (e.g. WWeellllbboorree -> Wellbore), so the
    # word-level collapsers below can recognize "Wellbore: Wellbore:" patterns.
    cleaned = _collapse_doubled_chars(raw)
    # Collapse "::" -> ":" (artifact from char-doubling on label punctuation).
    # We don't collapse other doubled punct because of e.g. "23:00 00:00" timestamps.
    cleaned = re.sub(r"::+", ":", cleaned)
    cleaned = _collapse_doubled_tokens(cleaned)
    cleaned = _fix_split_words(cleaned)
    return cleaned


def extract_text(pdf_path: str | Path) -> str:
    """
    Extract all text from a PDF as a single string.
    Page boundaries become double newlines.
    Doubled-token artifacts are collapsed.
    """
    pdf_path = Path(pdf_path)
    pages_text: list[str] = []

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            txt = page.extract_text() or ""
            pages_text.append(txt)

    raw = "\n\n".join(pages_text)
    # Normalize whitespace: collapse runs of spaces/tabs but preserve newlines
    raw = re.sub(r"[ \t]+", " ", raw)
    # Char-doubling fix first (e.g. "WWeellllbboorree" -> "Wellbore")
    cleaned = _collapse_doubled_chars(raw)
    # Collapse "::" left over from doubled-label rendering. Don't collapse
    # other doubled punctuation — "23:00 00:00" must stay intact.
    cleaned = re.sub(r"::+", ":", cleaned)
    # Then word-level doubled tokens, then rejoin known split words
    cleaned = _collapse_doubled_tokens(cleaned)
    cleaned = _fix_split_words(cleaned)
    return cleaned
