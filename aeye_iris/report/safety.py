"""Output safety validation for ALL released text: catalog statements, practitioner notes
and practitioner edits.

Policy (agreed: descriptive output + practitioner labels):
* BLOCK diagnosis / disease / treatment / prescription / dosage language, named eye
  diseases, and "organ + state" claims ("liver load", "kidney weakness", ...).
* General wellness suggestions written by the practitioner are allowed. They are always
  shown as the practitioner's own note, never as a system finding.
A word list cannot make an unsupported inference valid. It is the last line of defence on
top of template-only system text and mandatory practitioner approval.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

BLOCK_TERMS = [
    r"diagnos\w*", r"disease\w*", r"disorder\w*", r"syndrome\w*", r"cancer\w*", r"tumou?r\w*", r"malignan\w*",
    r"diabet\w*", r"hypertension", r"infection\w*", r"infected", r"inflammat\w*", r"toxin\w*", r"toxic\w*",
    r"detox\w*", r"cure[sd]?", r"curing", r"heal(?:s|ed|ing)?", r"treat(?:s|ed|ing|ment|ments)?", r"therap\w*",
    r"prescri\w*", r"dosage", r"dose[sd]?", r"\d+\s?(?:mg|mcg|ml|iu)\b", r"medicat\w*", r"symptom\w*",
    r"glaucoma", r"cataract\w*", r"uveitis", r"melanoma", r"keratitis", r"iritis", r"patholog\w*",
    r"illness\w*", r"you (?:have|suffer)", r"indicates? (?:a |an )?(?:problem|weakness|dysfunction|condition)",
    r"dysfunction\w*", r"organ failure", r"deficien\w*", r"congest\w*", r"stasis", r"acidosis",
]
ORGANS = (r"(?:liver|kidney|kidneys|heart|lung|lungs|stomach|bowel|colon|intestin\w*|pancrea\w*|thyroid|"
          r"spleen|gall\s?bladder|bladder|lymph\w*|hepat\w*|adrenal\w*|brain|uter\w*|ovar\w*|prostat\w*|"
          r"digestive system|nervous system|immune system|blood)")
STATE = (r"(?:load|stress|weak\w*|burden|imbalance|problem\w*|damage\w*|overload\w*|sluggish\w*|"
         r"insufficien\w*|impair\w*|disturb\w*|tension|toxicity)")
ORGAN_CLAIM = re.compile(
    rf"\b{ORGANS}\W+(?:\w+\W+){{0,3}}?{STATE}\b|\b{STATE}\W+(?:\w+\W+){{0,3}}?{ORGANS}\b", re.I)
BLOCK_RE = re.compile(r"\b(?:" + "|".join(BLOCK_TERMS) + r")\b", re.I)


@dataclass
class SafetyResult:
    ok: bool
    violations: list[str] = field(default_factory=list)


def check(text: str) -> SafetyResult:
    t = text or ""
    v = [m.group(0) for m in BLOCK_RE.finditer(t)] + [m.group(0) for m in ORGAN_CLAIM.finditer(t)]
    return SafetyResult(not v, sorted({x.strip() for x in v}))


def check_all(texts: list[str]) -> SafetyResult:
    viol = []
    for t in texts:
        r = check(t)
        viol += [f"{x!r} in: {t[:90]}" for x in r.violations]
    return SafetyResult(not viol, viol)
