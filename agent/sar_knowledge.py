"""
SAR domain knowledge base.

When the clue identifies one or more lost-person categories, the
retrieval function returns per-category baseline walking speeds and
typical day-hike reach ranges, plus two always-on heuristics about
water features and unreachable terrain.
"""

import re


# ══════════════════════════════════════════════════════════════════════
#  Per-person capabilities (category = individual, not sub-unit)
#
#  A sub-unit's effective pace and reach are bounded by its slowest /
#  weakest member; the LLM derives that from the sub-unit composition
#  given in the clue.
# ══════════════════════════════════════════════════════════════════════

CATEGORY_SPEEDS = {
    "child": {
        "on_trail":  (0.4, 0.7),
        "off_trail": (0.2, 0.4),
    },
    "average_adult": {
        "on_trail":  (0.7, 1.0),
        "off_trail": (0.4, 0.6),
    },
    "experienced_hiker": {
        "on_trail":  (0.9, 1.3),
        "off_trail": (0.6, 0.9),
    },
}

# Typical single-trip distance a person in this category can cover
# on foot without extended rest, in km.
CATEGORY_STAMINA_KM = {
    "child":             (4, 5),
    "average_adult":     (6, 9),
    "experienced_hiker": (15, 30),
}

# Clue cues that indicate a category is present in the scenario.
CATEGORY_TRIGGERS = {
    "child": [
        "child", "children", "kid", "minor", "toddler",
        "adolescent", "7-year", "age 7",
    ],
    "average_adult": [
        "couple", "married", "average adult", "moderately fit",
    ],
    "experienced_hiker": [
        "experienced", "fit hiker",
    ],
}

# Always-on SAR heuristics (returned whenever KB is retrieved).
GENERAL_HEURISTICS = [
    "Some lost persons may move toward water features (streams, ponds, "
    "lakeshores) for rest, drinking, or orientation.",
    "Generally, lost persons do not occupy regions they cannot physically "
    "reach.",
]


# ══════════════════════════════════════════════════════════════════════
#  Retrieval
# ══════════════════════════════════════════════════════════════════════

def _has_cue(text: str, cues) -> bool:
    """Case-insensitive cue match. Single words match as whole words;
    phrases containing a space or digit match as literal substrings."""
    for cue in cues:
        cue_l = cue.lower()
        if any(ch.isdigit() or ch in " -" for ch in cue_l):
            if cue_l in text:
                return True
        else:
            if re.search(rf"(?<!\w){re.escape(cue_l)}(?!\w)", text):
                return True
    return False


def _format_speed_table(categories) -> str:
    header = (
        "| category          | on-trail (m/s) | off-trail (m/s) |\n"
        "|-------------------|----------------|-----------------|"
    )
    rows = []
    for cat in categories:
        spec = CATEGORY_SPEEDS[cat]
        on_lo, on_hi = spec["on_trail"]
        off_lo, off_hi = spec["off_trail"]
        rows.append(
            f"| {cat:<17} | {on_lo:.1f}-{on_hi:.1f}        "
            f"| {off_lo:.1f}-{off_hi:.1f}         |"
        )
    return header + "\n" + "\n".join(rows)


def _format_stamina_table(categories) -> str:
    lines = ["  category           range (km)"]
    for cat in categories:
        lo, hi = CATEGORY_STAMINA_KM[cat]
        lines.append(f"  {cat:<17}  ~{lo}-{hi}")
    return "\n".join(lines)


def retrieve_knowledge(clue_text: str) -> str:
    """Return SAR domain knowledge relevant to the clue.

    * If the clue identifies one or more person categories, prepend
      per-category speed and stamina tables.
    * Always append the general SAR heuristics at the end.
    * If no category matches and no heuristics exist, return an empty
      string.
    """
    text = clue_text.lower()
    matched = [
        cat for cat, cues in CATEGORY_TRIGGERS.items()
        if _has_cue(text, cues)
    ]

    parts = []
    if matched:
        parts.append(
            "Baseline walking speeds for lost-person categories:\n\n"
            + _format_speed_table(matched)
        )
        parts.append(
            "Typical single-trip reach without extended rest:\n\n"
            + _format_stamina_table(matched)
        )
    if GENERAL_HEURISTICS:
        parts.append("\n".join(GENERAL_HEURISTICS))

    return "\n\n".join(parts) if parts else ""
