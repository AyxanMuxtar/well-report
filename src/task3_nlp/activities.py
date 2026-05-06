"""
Task 3b — Activity classification.

Tag each operation remark with one (or more) normalized activity labels.

Approach: rule-based with priority-ordered keyword patterns. We chose this
over training a model for three reasons:
  1. No labeled data exists for this domain.
  2. Drilling activity vocabulary is highly stylized and small.
  3. Rules are interpretable and trivial to extend.

Each rule is a (label, regex_pattern, priority) triple. Higher priority
matches win when multiple patterns fire. A remark can carry multiple labels
(e.g. an EQUIPMENT_FAILURE that is also a REPAIR).

Labels (from the brief, plus a few additions):
    TRIP_IN, TRIP_OUT, CIRCULATE, DRILL, CASING, CEMENT,
    PRESSURE_TEST, EQUIPMENT_FAILURE, REPAIR, WAIT, FISHING,
    SURVEY, BOP_OPERATION, RIG_UP, RIG_DOWN, MAKE_UP, LAY_DOWN,
    TIGHT_HOLE, STUCK_PIPE, WELL_CONTROL, OTHER
"""
from __future__ import annotations
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class _Rule:
    label: str
    pattern: re.Pattern
    priority: int = 0


def _R(label: str, pattern: str, priority: int = 0) -> _Rule:
    return _Rule(label=label, pattern=re.compile(pattern, re.IGNORECASE), priority=priority)


# Priority guide:
#   90 — well-control / safety-critical incidents (override everything)
#   70 — equipment failures and stuck-pipe events
#   50 — major operational categories (drill, trip, circ, casing, cement)
#   30 — secondary ops (rig up/down, make up, survey, BOP test, pressure test)
#   10 — fillers (wait, repair)
#    0 — fallback OTHER

_RULES: list[_Rule] = [
    # ----- 90: well-control / critical -----
    _R("WELL_CONTROL",      r"\b(?:kick|influx|well\s*control|shut\s*in|kill\s*operation|gas\s*alarm)\b", 90),

    # ----- 70: failures and stuck pipe -----
    _R("STUCK_PIPE",        r"\b(?:stuck\s+pipe|differential\s+stuck|pack\s*off|packed\s*off|pack[- ]off)\b", 70),
    _R("TIGHT_HOLE",        r"\btight(?:\s+hole|\s+spot)?\b", 70),
    _R("EQUIPMENT_FAILURE", r"\b(?:fail(?:ure|ed)?|out\s+of\s+service|broken|broke|leak(?!\s*off)(?:age|ing)?|malfunction|stalled|tripped\s+(?:offline|out))\b", 70),
    _R("FISHING",           r"\b(?:fish(?:ing)?|spear|overshot|wash[- ]over|junk\s+sub|impression\s+block)\b", 70),

    # ----- 50: major operational categories -----
    _R("TRIP_OUT",          r"\b(?:POOH|pull\s+out\s+of\s+hole|tripping\s+out|trip\s+out|TOH)\b", 50),
    _R("TRIP_IN",           r"\b(?:RIH|run\s+in\s+hole|tripping\s+in|trip\s+in|TIH)\b", 50),
    _R("CIRCULATE",         r"\b(?:circulat(?:e|ed|es|ing|ion)|bottoms[- ]up)\b", 50),
    _R("DRILL",             r"\b(?:drill(?:ing|ed)?|drilled|new\s+formation|reaming|reamed|orient(?:ed|ing)?)\b", 50),
    _R("CASING",            r"\b(?:casing|liner|run\s+(?:13\s*3/8|9\s*5/8|7|20|26|30)\s*(?:\"|in)?\s*casing|landing\s+string)\b", 50),
    _R("CEMENT",            r"\b(?:cement(?:ing|ed)?|cmt(?:\s|$)|cement\s+head|spacer|squeeze)\b", 50),

    # ----- 30: secondary operations -----
    _R("PRESSURE_TEST",     r"\b(?:pressure\s+test|leak[- ]off\s+test|formation\s+integrity\s+test|FIT|LOT\b)\b", 30),
    _R("BOP_OPERATION",     r"\b(?:BOP|blowout\s+preventer|annular|wellhead|riser|HPDR)\b", 30),
    _R("SURVEY",            r"\b(?:survey|inclination|azimuth|MWD\s+survey|gyro)\b", 30),
    _R("RIG_UP",            r"\b(?:R/U\b|rig(?:ged|ging)?\s+up)\b", 30),
    _R("RIG_DOWN",          r"\b(?:R/D\b|rig(?:ged|ging)?\s+down)\b", 30),
    _R("MAKE_UP",           r"\b(?:M/U\b|making?\s+up|made\s+up)\b", 30),
    _R("LAY_DOWN",          r"\b(?:L/O\b|L/D\b|lay(?:ing|ed)?\s+down|laid\s+out)\b", 30),
    _R("CUT",               r"\b(?:cut\s+(?:string|drill\s+pipe|pipe)|free\s+point|back[- ]off)\b", 30),

    # ----- 10: fillers -----
    _R("REPAIR",            r"\b(?:repair(?:ed|ing)?|replace(?:d|ment)?|maint(?:enance|ained?)|service\b|change(?:d)?\s+(?:bit|tool|seal))\b", 10),
    _R("WAIT",              r"\b(?:wait(?:ing)?|stand[- ]?by|hold\s+up|delay(?:ed)?|WOO|WOC)\b", 10),
    _R("FLOW_CHECK",        r"\b(?:flow[- ]?check(?:ed|ing)?|trip\s+tank)\b", 10),
    _R("TOOLBOX_TALK",      r"\b(?:tool\s*box\s+(?:talk|meeting)|safety\s+meeting|pre[- ]?job\s+(?:safety\s+)?meeting|TBT)\b", 10),
]


def classify_activity(remark: str | None,
                      *, max_labels: int = 3) -> list[str]:
    """
    Classify a remark into 0..max_labels normalized activity tags.

    Tags are returned highest-priority first. If no rule fires, returns
    ['OTHER'].

    Examples:
        classify_activity('Continued POOH with 17 1/2" BHA from 1798 m')
            -> ['TRIP_OUT']

        classify_activity('Mud pumps #2 and #3 out of service. Repaired pumps.')
            -> ['EQUIPMENT_FAILURE', 'REPAIR']

        classify_activity('Tight hole at 1335 m MD when stabilizer entering shoetrack.')
            -> ['TIGHT_HOLE']

        classify_activity('Held tool box meeting prior to L/O 26" BHA.')
            -> ['TOOLBOX_TALK', 'LAY_DOWN']
    """
    if not remark:
        return ["OTHER"]

    hits: list[tuple[int, str]] = []   # (priority, label)
    for rule in _RULES:
        if rule.pattern.search(remark):
            hits.append((rule.priority, rule.label))

    if not hits:
        return ["OTHER"]

    # Sort by priority desc, dedupe labels (preserve first occurrence order)
    hits.sort(key=lambda x: -x[0])
    seen: set[str] = set()
    out: list[str] = []
    for _prio, label in hits:
        if label not in seen:
            seen.add(label)
            out.append(label)
        if len(out) >= max_labels:
            break
    return out


def primary_activity(remark: str | None) -> str:
    """Return just the highest-priority label (or OTHER)."""
    return classify_activity(remark, max_labels=1)[0]


# =============================================================================
# Aggregate stats over a corpus
# =============================================================================

def label_distribution(remarks: list[str | None]) -> dict[str, int]:
    """Count how often each label appears as the primary activity."""
    from collections import Counter
    return dict(Counter(primary_activity(r) for r in remarks))
