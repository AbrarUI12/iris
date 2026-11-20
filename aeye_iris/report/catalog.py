"""Approved, DESCRIPTIVE statement templates (no LLM). Every sentence in a generated
report comes from here (or is practitioner-authored text that passed `safety.check`).

Nothing here states or implies organ function, disease, toxins or treatment.
"""
from __future__ import annotations

MARKING_DESC = {
    "dark_pigmentation": "localised darker (pigment-like) region",
    "white_marking": "localised lighter region",
    "lacuna": "gap-like area in the iris fibre pattern (lacuna candidate)",
    "contraction_furrow": "curved concentric line (contraction-furrow candidate)",
    "collarette": "collarette (inner ring)",
    "fibre_type": "fibre pattern",
}

COLOR_DESC = {"blue": "blue", "brown": "brown", "green_mixed": "green / mixed", "uncertain": "not clearly classifiable"}

STATEMENTS = {
    "color": "The visible iris colour of the {eye} eye was classified as {color} from the photograph.",
    "color_uncertain": "The iris colour of the {eye} eye could not be classified confidently from the photograph.",
    "marking_in_zone": "A {marking} was observed in sector {sector} ({ring} ring) of the {eye} iris photograph.",
    "marking_none": "No {marking} was detected in the assessable areas of the {eye} iris photograph.",
    "unassessable": "Fine iris markings could not be assessed reliably in the {eye} photograph (image detail insufficient).",
    "collarette": "The collarette of the {eye} iris lies at about {pct}% of the distance from pupil to iris edge{flags}.",
    "fibre": "The fibre pattern of the {eye} iris appears predominantly {label}.",
    "retake": "The {eye} eye photograph needs to be retaken: {reason}.",
    "referral": "Your practitioner recommends an eye examination by an optometrist or ophthalmologist.",
    "limitations": ("These are descriptive observations of photographs. They are not a medical examination "
                    "and do not diagnose, treat or rule out any condition."),
}

PRACTITIONER_LABEL_PREFIX = "Practitioner framework note"


def render(statement_id: str, **kw) -> str:
    return STATEMENTS[statement_id].format(**kw)
