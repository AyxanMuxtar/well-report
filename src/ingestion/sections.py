"""
Section parsers for the drilling daily report.

Each parser accepts the full normalized text of a PDF and returns either:
  - dict (for header, which has at most one of each field), or
  - list[dict] (for repeated sections like operations, fluid samples, etc.)

The PDFs follow a consistent template across all eras (1992–2018). All section
headings appear verbatim in the text (after our doubled-token collapse).

The strategy for each repeated section is:
    1. Find the section's start by its heading.
    2. Find the section's end by the next known section heading (or EOF).
    3. Slice that block and parse rows from it with section-specific regex.
"""
from __future__ import annotations
import re
from typing import Optional

from src.ingestion.converters import (
    to_str, to_float, to_int, to_bool_yn, to_datetime, to_date,
)

# -----------------------------------------------------------------------------
# Section markers — used to slice the document into sections.
# Order matters: this is the canonical order they appear in the source PDFs.
# -----------------------------------------------------------------------------
SECTION_HEADINGS = [
    "Summary report",
    "Summary of activities (24 Hours)",
    "Summary of planned activities (24 Hours)",
    "Operations",
    "Drilling Fluid",
    "Pore Pressure",
    "Survey Station",
    "Log Information",
    "Lithology Information",
    "Gas Reading Information",
]


def _slice_section(text: str, heading: str) -> Optional[str]:
    """
    Return the substring from `heading` (exclusive) to the next known section
    heading (exclusive), or to end of text. Returns None if the heading is
    not found in the text.
    """
    # Build regex for "this heading" and "any subsequent heading"
    # Use word-boundary anchor on heading start to avoid partial matches
    pattern_start = re.escape(heading)
    m_start = re.search(pattern_start, text)
    if not m_start:
        return None
    body_start = m_start.end()

    # Find the earliest next heading after body_start
    next_pos = len(text)
    for h in SECTION_HEADINGS:
        if h == heading:
            continue
        m = re.search(re.escape(h), text[body_start:])
        if m:
            pos = body_start + m.start()
            if pos < next_pos:
                next_pos = pos
    return text[body_start:next_pos].strip()


# =============================================================================
# 1. Header parser
# =============================================================================
#
# Header fields appear as `Label: value` pairs. Different PDFs have wildly
# different whitespace handling (some put each field on its own line, some
# inline several together). A robust strategy:
#
#   For each field, capture everything between its label and the next *known*
#   label or section heading. That way a field's value can never accidentally
#   absorb subsequent fields, regardless of how pdfplumber laid out the text.
#
# Each entry: (db_column_name, label_in_pdf, converter)

_HEADER_LABELS: list[tuple[str, str, callable]] = [
    ("wellbore_id",                  "Wellbore:",                          to_str),
    ("period",                       "Period:",                            to_str),       # split below
    ("status",                       "Status:",                            to_str),
    ("report_creation_time",         "Report creation time:",              to_datetime),
    ("report_number",                "Report number:",                     to_int),
    ("days_ahead_behind",            "Days Ahead/Behind (+/-):",           to_float),
    ("operator",                     "Operator:",                          to_str),
    ("rig_name",                     "Rig Name:",                          to_str),
    ("drilling_contractor",          "Drilling contractor:",               to_str),
    ("spud_date",                    "Spud Date:",                         to_datetime),
    ("wellbore_type",                "Wellbore type:",                     to_str),
    ("elevation_rkb_msl_m",          "Elevation RKB-MSL (m):",             to_float),
    ("elevation_rkb_msl_m",          "Elevation RKB-MSL ():",              to_float),    # alt unit-less
    ("water_depth_msl_m",            "Water depth MSL (m):",               to_float),
    ("tight_well",                   "Tight well:",                        to_bool_yn),
    ("hpht",                         "HPHT:",                              to_bool_yn),
    ("temperature",                  "Temperature ():",                    to_float),
    ("pressure",                     "Pressure ():",                       to_float),
    ("date_well_complete",           "Date Well Complete:",                to_date),
    ("dist_drilled_m",               "Dist Drilled (m):",                  to_float),
    ("penetration_rate_mph",         "Penetration rate (m/h):",            to_float),
    ("hole_dia_in",                  "Hole Dia (in):",                     to_float),
    ("hole_dia_in",                  "Hole Dia ():",                       to_float),     # alt unit-less
    ("pressure_test_type",           "Pressure Test Type:",                to_str),
    ("formation_strength_gcm3",      "Formation strength (g/cm3):",        to_float),
    ("formation_strength_gcm3",      "Formation strength ():",             to_float),     # alt unit-less
    ("dia_last_casing",              "Dia Last Casing ():",                to_float),
    ("depth_kickoff_md",             "Depth at Kick Off mMD:",             to_float),
    ("depth_kickoff_tvd",            "Depth at Kick Off mTVD:",            to_float),
    ("depth_md",                     "Depth mMd:",                         to_float),
    ("depth_tvd",                    "Depth mTVD:",                        to_float),
    ("plug_back_depth_md",           "Plug Back Depth mMD:",               to_float),
    ("depth_formation_strength_md",  "Depth at formation strength mMD:",   to_float),
    ("depth_formation_strength_tvd", "Depth At Formation Strength mTVD:",  to_float),
    ("depth_last_casing_md",         "Depth At Last Casing mMD:",          to_float),
    ("depth_last_casing_tvd",        "Depth At Last Casing mTVD:",         to_float),
]


def _build_field_stop_pattern() -> str:
    """
    Build an alternation regex of all known header labels and section headings
    so that any header-field value capture stops at the next known label.
    """
    stops = list({label for _, label, _ in _HEADER_LABELS})
    stops += SECTION_HEADINGS
    # Sort by length descending so longer labels match first (e.g. "Depth at
    # Kick Off mTVD:" before "Depth mTVD:").
    stops.sort(key=len, reverse=True)
    return "|".join(re.escape(s) for s in stops)


_FIELD_STOP_RX = _build_field_stop_pattern()


# Pre-built sorted list of (label, columns, converter) — longest label first
# so multi-word labels match before short ones.
_HEADER_LABELS_SORTED = sorted(
    _HEADER_LABELS,
    key=lambda t: -len(t[1]),
)


def _find_all_label_occurrences(line: str) -> list[tuple[int, int, str, str, callable]]:
    """
    Scan `line` for every known header label and return their positions.
    Returns a list of tuples sorted by start_pos:
        (start_pos, end_pos, column_name, label, converter)

    Overlapping matches are filtered: if 'Depth mTVD:' and 'Depth at Kick Off mTVD:'
    both could match at the same position, only the longer one wins.
    """
    candidates: list[tuple[int, int, str, str, callable]] = []
    for col, label, conv in _HEADER_LABELS_SORTED:
        # Find every occurrence of this label
        for m in re.finditer(re.escape(label), line):
            candidates.append((m.start(), m.end(), col, label, conv))

    # Sort by start, then by length descending so longest match at a given
    # position comes first.
    candidates.sort(key=lambda x: (x[0], -(x[1] - x[0])))

    # Filter out any candidate whose span overlaps an earlier (longer) one
    kept: list[tuple[int, int, str, str, callable]] = []
    for cand in candidates:
        s, e, *_ = cand
        if any(not (e <= ks or s >= ke) for ks, ke, *_ in kept):
            continue
        kept.append(cand)

    # Final sort by start position for left-to-right walk
    kept.sort(key=lambda x: x[0])
    return kept


def _strip_dangling_label_prefix(value: str) -> str:
    """
    Some PDFs emit a stray label-prefix word at the end of a value due to
    word-merge issues across visual rows. For example:
        'Mærsk Contractors Depth'  ->  'Mærsk Contractors'
    where 'Depth' is the start of 'Depth At Last Casing mTVD:' from the next
    visual row.

    Strategy: tokenize trailing tokens; if the last 1-3 tokens form a prefix
    of any known label, drop them.
    """
    if not value:
        return value
    tokens = value.split()
    if not tokens:
        return value
    all_labels_lc = [lab.lower() for _, lab, _ in _HEADER_LABELS]
    for n in (3, 2, 1):
        if len(tokens) <= n:
            continue
        tail = " ".join(tokens[-n:]).lower()
        # Match if any label STARTS with this tail (followed by space or end)
        for lab in all_labels_lc:
            if lab.startswith(tail + " ") or lab == tail + ":":
                return " ".join(tokens[:-n]).strip()
    return value


def _parse_header_line(line: str) -> dict:
    """
    Parse one line that may contain multiple "Label: value" pairs side-by-side
    (the PDF's two-column header layout).

    Strategy: locate every known label in the line; the value of label[i] is
    the slice of text between the END of label[i] and the START of label[i+1].
    """
    occurrences = _find_all_label_occurrences(line)
    if not occurrences:
        return {}

    out: dict = {}
    for i, (start, end, col, label, conv) in enumerate(occurrences):
        # Value spans from end of this label to the start of the next label
        value_end = occurrences[i + 1][0] if i + 1 < len(occurrences) else len(line)
        raw = line[end:value_end].strip()
        # If this is the final label on the line, the trailing token may be
        # the start of a label from the next visual row that pdfplumber merged in.
        if i + 1 == len(occurrences):
            raw = _strip_dangling_label_prefix(raw)
        if raw == "":
            value = None
        else:
            try:
                value = conv(raw)
            except Exception:
                value = None
        # First non-None wins (alt-unit variants come after main label)
        if out.get(col) is None and value is not None:
            out[col] = value
    return out


def parse_header(text: str) -> dict:
    """
    Extract header/metadata fields as a dict. Missing fields -> None.

    The PDF header is laid out in two columns. pdfplumber emits each visual
    row as one text line where left-column and right-column fields appear
    side-by-side. We parse each line independently, finding all label
    occurrences and slicing values between them.
    """
    # The header occupies everything before the first "Summary of activities" heading
    summary_match = re.search(r"Summary of activities", text)
    header_block = text[: summary_match.start()] if summary_match else text

    # Initialize all fields to None
    out: dict = {col: None for col, _, _ in _HEADER_LABELS}

    # Walk line by line; a header line may contain multiple labels side by side.
    for line in header_block.splitlines():
        line = line.strip()
        if not line:
            continue
        line_fields = _parse_header_line(line)
        for col, val in line_fields.items():
            if out.get(col) is None and val is not None:
                out[col] = val

    # Period is special: parse "YYYY-MM-DD HH:MM - YYYY-MM-DD HH:MM" into start+end
    period_text = out.pop("period", None)
    out["period_start"] = None
    out["period_end"]   = None
    if period_text:
        m = re.match(
            r"(\d{4}-\d{2}-\d{2}(?:\s+\d{2}:\d{2})?)\s*-\s*(\d{4}-\d{2}-\d{2}(?:\s+\d{2}:\d{2})?)",
            period_text,
        )
        if m:
            out["period_start"] = to_datetime(m.group(1))
            out["period_end"]   = to_datetime(m.group(2))

    return out


# =============================================================================
# 2. Summary parsers (24h activities + planned)
# =============================================================================

def parse_summary_24h(text: str) -> Optional[str]:
    body = _slice_section(text, "Summary of activities (24 Hours)")
    return to_str(body) if body else None


def parse_planned_24h(text: str) -> Optional[str]:
    body = _slice_section(text, "Summary of planned activities (24 Hours)")
    return to_str(body) if body else None


# =============================================================================
# 3. Operations parser
# =============================================================================
#
# After text normalization, the Operations section starts with a column header:
#   "Start time End time End Depth mMD Main - Sub Activity State Remark"
# followed by rows of the form:
#   "00:00 03:45 2447 drilling - drill ok Drilled and oriented..."
#
# Each row begins with two HH:MM stamps and a numeric depth. The body that
# follows contains:
#   <main> "-" <sub> <state> <remark>
# where main/sub may be split across whitespace (e.g. "drilling", "circulating
# conditioning") and state is one of {ok, fail, n/a}.
#
# Strategy:
#   1. Find every row by anchoring on the HH:MM HH:MM <depth> prefix.
#   2. For each row's body, locate the first occurrence of " ok ", " fail ",
#      or " n/a " — that's the state. Everything before it is "<main> - <sub>",
#      everything after is the remark.
#   3. Split main/sub on the dash.

_OP_ROW_RX = re.compile(
    r"""
    (?P<start>\d{2}:\d{2})
    \s+
    (?P<end>\d{2}:\d{2})
    \s+
    (?P<depth>-?\d+(?:\.\d+)?)
    \s+
    (?P<body>.+?)
    (?=
        \s+\d{2}:\d{2}\s+\d{2}:\d{2}\s+-?\d+(?:\.\d+)?\s+
        | $
    )
    """,
    re.VERBOSE | re.DOTALL,
)

# State markers in priority order. Surrounded by whitespace to avoid matching
# "fail" inside a remark word.
_STATE_RX = re.compile(r"\s+(?P<state>ok|fail|n/a)\s+", re.IGNORECASE)


def _parse_op_body(body: str) -> tuple[str | None, str | None, str | None, str]:
    """
    Parse one operation's body string into (main, sub, state, remark).
    If the structure can't be recognized, returns (None, None, None, body).
    """
    body = re.sub(r"\s+", " ", body).strip()

    # Find first state marker
    m = _STATE_RX.search(body)
    if not m:
        return None, None, None, body

    activity_part = body[: m.start()].strip()
    state         = m.group("state").lower()
    remark        = body[m.end():].strip()

    # Split activity into main/sub on the dash separator (either "-" or "--").
    # Activity can be "drilling - drill" or "drilling -- drill" (artifact of the
    # original tool's column rendering) or just "drilling" (no sub).
    main: str | None = None
    sub:  str | None = None
    if re.search(r"\s+-{1,2}\s+", activity_part):
        parts = re.split(r"\s+-{1,2}\s+", activity_part, maxsplit=1)
        main = parts[0].strip() or None
        sub  = parts[1].strip() if len(parts) > 1 else None
        sub  = sub or None
    else:
        main = activity_part or None

    return main, sub, state, remark


def _build_op_text(main: str | None, sub: str | None, state: str | None, remark: str | None) -> str:
    """The string used for similarity matching in Task 2."""
    parts = [p for p in (main, sub, state, remark) if p]
    return " | ".join(parts).strip()


def parse_operations(text: str) -> list[dict]:
    """Parse the Operations section into a list of row dicts."""
    body = _slice_section(text, "Operations")
    if not body:
        return []

    rows: list[dict] = []
    for idx, m in enumerate(_OP_ROW_RX.finditer(body)):
        body_text = m.group("body")
        main, sub, state, remark = _parse_op_body(body_text)

        depth = to_float(m.group("depth"))
        rows.append({
            "op_index":       idx,
            "start_time":     m.group("start"),
            "end_time":       m.group("end"),
            "end_depth_md":   depth,
            "main_activity":  main,
            "sub_activity":   sub,
            "state":          state,
            "remark":         remark or None,
            "op_text":        _build_op_text(main, sub, state, remark),
        })
    return rows


# =============================================================================
# 4. Drilling Fluid parser
# =============================================================================
# The fluid block is column-oriented, e.g.:
#   Sample Time 03:30 11:30 15:30 20:00
#   Sample Point Flowline Flowline Flowline Flowline
#   Sample Depth mMD 2445 2548 2583 2591
#   Fluid Type OBM-Standard OBM-Standard OBM-Standard OBM-Standard
#   Fluid Density (g/cm3) 1.43 1.43 1.43 1.43
#   ...
# We parse it row by row, then transpose into per-sample dicts.

_FLUID_FIELD_PATTERNS: list[tuple[str, str, callable]] = [
    ("sample_time",          r"Sample Time\s+([^\n]+)",                                  None),
    ("sample_point",         r"Sample Point\s+([^\n]+)",                                 None),
    ("sample_depth_md",      r"Sample Depth mMD\s+([^\n]+)",                             None),
    ("fluid_type",           r"Fluid Type\s+([^\n]+)",                                   None),
    ("fluid_density_gcm3",   r"Fluid Density \(g/cm3\)\s+([^\n]+)",                      None),
    ("funnel_visc_s",        r"Funnel Visc \(s\)\s+([^\n]+)",                            None),
    ("plastic_visc_mpas",    r"Plastic visc\. \(mPa\.s\)\s+([^\n]+)",                    None),
    ("yield_point_pa",       r"Yield point \(Pa\)\s+([^\n]+)",                           None),
    ("test_temp_hpht_degc",  r"Test Temp HPHT \(degC\)\s+([^\n]+)",                      None),
]

# Which fields are numeric (need to_float) vs string (raw)
_FLUID_NUMERIC_FIELDS = {
    "sample_depth_md", "fluid_density_gcm3", "funnel_visc_s",
    "plastic_visc_mpas", "yield_point_pa", "test_temp_hpht_degc",
}


def parse_drilling_fluid(text: str) -> list[dict]:
    body = _slice_section(text, "Drilling Fluid")
    if not body:
        return []

    # Extract the value list for each known field
    field_values: dict[str, list[str]] = {}
    for col, pat, _ in _FLUID_FIELD_PATTERNS:
        m = re.search(pat, body)
        if not m:
            continue
        # Split tokens by whitespace; for fluid_type this may merge "OBM-Standard"
        # Sample Time / Sample Point are short tokens. Tokenize by whitespace.
        tokens = m.group(1).strip().split()
        field_values[col] = tokens

    if not field_values:
        return []

    n_samples = max((len(v) for v in field_values.values()), default=0)
    if n_samples == 0:
        return []

    samples: list[dict] = []
    for i in range(n_samples):
        sample: dict = {"sample_index": i}
        for col in (c for c, _, _ in _FLUID_FIELD_PATTERNS):
            tokens = field_values.get(col, [])
            if i < len(tokens):
                raw = tokens[i]
                if col in _FLUID_NUMERIC_FIELDS:
                    sample[col] = to_float(raw)
                else:
                    sample[col] = to_str(raw)
            else:
                sample[col] = None
        samples.append(sample)
    return samples


# =============================================================================
# 5. Pore Pressure parser
# =============================================================================
# Format observed in normalized text:
#   Time Depth mMD Depth TVD Equ Mud Weight (g/cm3) Reading
#   00:00 2591 1 estimated
#
# In practice the row contains: HH:MM <depth_md> <equ_mud_weight> <reading_type>
# (depth_tvd is rarely populated and may be omitted in extracted text).

_PORE_ROW_RX = re.compile(
    r"(?P<time>\d{2}:\d{2})\s+"
    r"(?P<md>-?\d+(?:\.\d+)?)\s+"
    r"(?P<emw>-?\d+(?:\.\d+)?)\s+"
    r"(?P<reading>[a-zA-Z][a-zA-Z _]*)"
)


def parse_pore_pressure(text: str) -> list[dict]:
    body = _slice_section(text, "Pore Pressure")
    if not body:
        return []
    rows = []
    for idx, m in enumerate(_PORE_ROW_RX.finditer(body)):
        rows.append({
            "reading_index":       idx,
            "sample_time":         m.group("time"),
            "depth_md":            to_float(m.group("md")),
            "depth_tvd":           None,
            "equ_mud_weight_gcm3": to_float(m.group("emw")),
            "reading_type":        to_str(m.group("reading")),
        })
    return rows


# =============================================================================
# 6. Survey Station parser
# =============================================================================
# Layout:
#   Depth mMD Depth mTVD Inclination (dega) Azimuth (dega) Comment
#   2431.4 2304.8 27.16 209.96
#   2472.5 2341.2 28.20 217.51
#   ...
#
# Strategy: skip past header words to first numeric, then 4 numerics per row.
# Comment is rarely populated; we leave it None for now.
_SURVEY_ROW_RX = re.compile(
    r"(?P<md>\d+(?:\.\d+)?)\s+"
    r"(?P<tvd>\d+(?:\.\d+)?)\s+"
    r"(?P<incl>\d+(?:\.\d+)?)\s+"
    r"(?P<azi>\d+(?:\.\d+)?)"
)


def _survey_data_start(body: str) -> int:
    m = re.search(
        r"\d+(?:\.\d+)?\s+\d+(?:\.\d+)?\s+\d+(?:\.\d+)?\s+\d+(?:\.\d+)?",
        body,
    )
    return m.start() if m else 0


def parse_survey_stations(text: str) -> list[dict]:
    body = _slice_section(text, "Survey Station")
    if not body:
        return []
    data_body = body[_survey_data_start(body):]
    rows = []
    for idx, m in enumerate(_SURVEY_ROW_RX.finditer(data_body)):
        rows.append({
            "station_index":   idx,
            "depth_md":        to_float(m.group("md")),
            "depth_tvd":       to_float(m.group("tvd")),
            "inclination_deg": to_float(m.group("incl")),
            "azimuth_deg":     to_float(m.group("azi")),
            "comment":         None,
        })
    return rows


# =============================================================================
# 7. Lithology parser
# =============================================================================
# Format observed:
#   Start Depth mMD End Depth mMD Start Depth mTVD End Depth mTVD Shows Description Lithology Description
#   2474 2531 2345 2391 Shale, tuff and minor dolomite
#   2531 2591 2391 2441 Shale, minor limestone and dolomite, silty base
#
# Strategy: skip past the column-header words to the first numeric token,
# then iterate: 4 numbers + description-up-to-(next 4 numbers OR end-of-section).

_LITHO_ROW_RX = re.compile(
    r"(?P<smd>\d+(?:\.\d+)?)\s+"
    r"(?P<emd>\d+(?:\.\d+)?)\s+"
    r"(?P<stvd>\d+(?:\.\d+)?)\s+"
    r"(?P<etvd>\d+(?:\.\d+)?)\s+"
    r"(?P<desc>.+?)"
    r"(?=\s+\d+(?:\.\d+)?\s+\d+(?:\.\d+)?\s+\d+(?:\.\d+)?\s+\d+(?:\.\d+)?|\Z)",
    re.DOTALL,
)


def _lithology_data_start(body: str) -> int:
    """
    Skip past the column-header text. Returns the index where the first row
    of numeric data begins, or 0 if not found.
    """
    # Look for the first 4 consecutive numbers
    m = re.search(
        r"\d+(?:\.\d+)?\s+\d+(?:\.\d+)?\s+\d+(?:\.\d+)?\s+\d+(?:\.\d+)?",
        body,
    )
    return m.start() if m else 0


def parse_lithology(text: str) -> list[dict]:
    body = _slice_section(text, "Lithology Information")
    if not body:
        return []
    data_body = body[_lithology_data_start(body):]
    rows = []
    for idx, m in enumerate(_LITHO_ROW_RX.finditer(data_body)):
        desc = re.sub(r"\s+", " ", m.group("desc")).strip()
        rows.append({
            "interval_index":        idx,
            "start_depth_md":        to_float(m.group("smd")),
            "end_depth_md":          to_float(m.group("emd")),
            "start_depth_tvd":       to_float(m.group("stvd")),
            "end_depth_tvd":         to_float(m.group("etvd")),
            "shows_description":     None,
            "lithology_description": desc or None,
        })
    return rows


# =============================================================================
# 8. Gas Reading parser
# =============================================================================
# The Gas Reading layout is messy — column headers wrap around the data rows
# and exact column ordering can vary. We extract the reliably-identifiable
# parts (time and gas class) and capture the rest of the row as raw values.
#
# Many reports have no gas readings at all; that's normal.

_GAS_HEAD_RX = re.compile(
    r"(?P<time>\d{2}:\d{2})\s+(?P<gclass>(?:drilling\s+gas|connection\s+gas|trip\s+gas|background\s+gas)[a-z\s]*)",
    re.IGNORECASE,
)


def parse_gas_readings(text: str) -> list[dict]:
    body = _slice_section(text, "Gas Reading Information")
    if not body:
        return []
    rows = []
    for idx, m in enumerate(_GAS_HEAD_RX.finditer(body)):
        rows.append({
            "reading_index":     idx,
            "sample_time":       m.group("time"),
            "gas_class":         to_str(m.group("gclass")),
            "depth_top_md":      None,
            "depth_bottom_md":   None,
            "depth_top_tvd":     None,
            "depth_bottom_tvd":  None,
            "c1_ppm": None, "c2_ppm": None, "c3_ppm": None,
            "ic4_ppm": None, "ic5_ppm": None,
            "highest_gas_pct":   None,
            "lowest_gas_pct":    None,
        })
    return rows
