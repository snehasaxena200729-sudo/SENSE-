"""
interactive/parser.py - what kind of question is this, and about what?

Technique: SLOT FILLING with a small rule grammar (the same family of method
as the existing answers.classify_intent). Step 1 picks the question TYPE
(colour, count, side ...) from key phrases; step 2 fills the SLOTS: which
class ("chairs" -> chair), which named thing ("my bottle", "Sneha"), which
side ("on my right"), which colour ("is it blue?").

Rules are checked in a fixed order and the parser DECLINES (returns None)
for everything the existing SENSE commands own - so those keep running
exactly the old code. That guard is the most important part of this file.
"""
import re
from dataclasses import dataclass
from typing import Optional

from . import text as T

# Existing intents (answers.INTENT_PATTERNS) that Path B must never take.
GUARD_INTENTS = {"FORGET", "RENAME", "STOP_FIND", "FIND", "LEARN", "QUIET", "VERBOSE",
                 "NORMAL", "SLOWER", "FASTER", "CHANGES", "WHO", "RECENT"}

PRONOUNS = {"it", "that", "this", "them", "they", "one", "those", "these"}


@dataclass
class Query:
    kind: str                     # TIME DATE READ COLOUR COUNT SIDE DESCRIBE PAST REWRITE
    cls: Optional[str] = None     # detector class: "chair"
    name: Optional[str] = None    # a specific remembered/known name: "My Bottle", "Sneha"
    side: Optional[str] = None    # "left" | "right" | "ahead" | "behind"
    colour: Optional[str] = None  # colour the user asked to confirm ("is it blue?")
    pronoun: bool = False         # "it" / "that": use the follow-up subject
    safety: bool = False          # "is it safe to walk?": never answer "yes, safe"
    handheld: bool = False        # "what am I holding?": small objects only
    undetectable: Optional[str] = None   # "keys": neither model can see them
    rewrite: Optional[str] = None # REWRITE: the text to hand back to existing SENSE


_TIME = re.compile(r"\bwhat time is it\b|\bwhat is the time\b|\btell me the time\b|"
                   r"\bcurrent time\b|\bwhat time now\b|\bthe time now\b|\bwhat is the time now\b|"
                   r"\btime is it\b")
_DATE = re.compile(r"(?<!up )\bdate\b|\bwhat day\b|\bwhich day\b|\bwhat month\b|\bwhat year\b|"
                   r"\bday is (it|today)\b|\bday of the week\b|\bwhat is today\b")
_READ = re.compile(r"\bwhat does\b.*\bsay\b|\bwhat do(es)? (it|this|that|they) say\b|"
                   r"\bwhat is (written|printed)\b|\bread (out )?(the|this|that|my|me|what)\b|"
                   r"\bread\b.*\b(label|packet|box|sign|text|letter|note|page|paper|screen|"
                   r"medicine|wrapper|card|menu|book|bill|receipt)\b|\bwhat (is|does) the (label|text|sign)\b")
_SAFETY = re.compile(r"\bsafe to (walk|go|move|step)\b|\bcan i (walk|go|move|step) (forward|ahead|straight|on)\b|"
                     r"\bis (the|my) (way|path|road) clear\b|\bpath (is )?clear\b|\bway (is )?clear\b|"
                     r"\banything in (my|the) way\b|\bany obstacles?\b|\bobstacles? (ahead|in front)\b|"
                     r"\bwill i (hit|bump|walk into)\b")
_COLOUR = re.compile(r"\bcolour(s|ed)?\b|\bwhat shade\b")
_IS_COLOUR = re.compile(r"^(is|are) (it|this|that|the|my|they|those|these)\b")
_COUNT = re.compile(r"\bhow many\b|\bnumber of\b|\bcount\b|\bhow much people\b")
_PAST = re.compile(r"\bpreviously\b|\bbefore\b|\bearlier\b|\bused to be\b|\bwas there\b|"
                   r"\bwere there\b|\bdid you see\b|\bhave you seen\b|\bwas (in front|on my|behind|to my)\b|"
                   r"\bwhat was\b|\bwhat were\b|\bwho was\b")
_SIDE_ASK = re.compile(r"\b(anything|something|anyone|anybody|someone|somebody|what|who|objects?|"
                       r"things?|stuff|items?|is there|are there|is there a|any)\b")
_DESCRIBE = re.compile(r"\bdescribe\b|\bwhat am i (holding|looking at|pointing at)\b|"
                       r"\bwhat is (this|that)\b(?! (bottle|cup|phone))|\bwhat (is )?in my hand\b|"
                       r"\btell me about\b|\bwhat (kind|type|sort) of\b|\bwhat is it\b|\bwhat object\b")

_SIDES = [
    ("behind", re.compile(r"\bbehind( me| you)?\b|\bat my back\b")),
    ("left", re.compile(r"\b(on|to|at) (my|the|your) left\b|\bleft side\b|\bmy left\b|\bon the left\b")),
    ("right", re.compile(r"\b(on|to|at) (my|the|your) right\b|\bright side\b|\bmy right\b|\bon the right\b")),
    ("ahead", re.compile(r"\bin front( of (me|you|us))?\b|\bahead( of me)?\b|\bstraight ahead\b|"
                         r"\bfacing me\b|\bbefore me\b")),
]


def side_in(t):
    for side, pat in _SIDES:
        if pat.search(t):
            return side
    return None


def _has_pronoun(t):
    return any(w in PRONOUNS for w in t.split())


def parse(raw_text, existing_intent=None, ocr_owns=False, find_target=None):
    """Return a Query, or None when existing SENSE should answer (unchanged).

    existing_intent: answers.classify_intent(text) for the same text.
    ocr_owns: the OCR controller recognises it ("read this", "stop reading").
    find_target(text): sense_master.find_target - a remembered/known name or None.
    """
    t = T.normalise(raw_text)
    if not t or ocr_owns or existing_intent in GUARD_INTENTS:
        return None
    if existing_intent == "WHERE" and t.startswith("where") and not _has_pronoun(t):
        return None       # "where is my phone / the door / Sneha": existing code, unchanged
    if re.search(r"\bhow far\b", t) and not _has_pronoun(t):
        return None       # distance questions: existing WHERE + door_ranger answer them

    if _TIME.search(t):
        return Query("TIME")
    if _DATE.search(t) and "when" not in t.split():
        return Query("DATE")
    if _READ.search(t):
        return Query("READ")

    cls = T.class_in(t)
    side = side_in(t)
    name = find_target(raw_text) if find_target else None
    if name and T.CLASS_WORDS.get(name.lower()) == cls:
        name = None                       # "bottle" is a class, not a specific thing
    pron = _has_pronoun(t) and cls is None and name is None
    undetectable = T.undetectable_in(t) if (cls is None and name is None) else None

    if _SAFETY.search(t):
        return Query("SIDE", side="ahead", safety=True)
    if _COLOUR.search(t) or (_IS_COLOUR.search(t) and T.colour_in(t) and
                             T.colour_in(t) not in (name or "").lower()):
        asked = None if _COLOUR.search(t) else T.colour_in(t)     # "is the bottle blue?"
        return Query("COLOUR", cls=cls, name=name, side=side, colour=asked,
                     pronoun=pron or (cls is None and name is None), undetectable=undetectable)
    if _COUNT.search(t):
        generic = any(w in T.GENERIC_THING for w in t.split())
        return Query("COUNT", cls=cls, name=name, side=side,
                     pronoun=pron and not generic, undetectable=undetectable)
    if re.search(r"\bdescribe\b|\bwhat am i (holding|looking at|pointing at)\b|\bin my hand\b", t) \
            and existing_intent != "OVERVIEW":
        return Query("DESCRIBE", cls=cls, name=name, side=side,
                     pronoun=pron and not re.search(r"\b(this|that)\b", t),
                     handheld=bool(re.search(r"\bholding\b|\bin my hand\b", t)))
    if side == "behind" or (side and _PAST.search(t)):
        if name is None:
            return Query("PAST", cls=cls, side=side, undetectable=undetectable)
        return None       # "was my phone on my left before?": existing PREVIOUS / WHERE
    if side and (_SIDE_ASK.search(t) or cls) and not t.startswith("where") and name is None:
        return Query("SIDE", cls=cls, side=side, undetectable=undetectable)
    if _DESCRIBE.search(t) and existing_intent != "OVERVIEW":   # "describe the room": existing overview
        return Query("DESCRIBE", cls=cls, name=name, side=side,
                     pronoun=pron and not re.search(r"\b(this|that)\b", t))
    return None


def rewrite_follow_up(raw_text, subject_phrase, existing_intent):
    """'where is it' + subject 'the bottle' -> 'where is the bottle' for the EXISTING
    WHERE / WHEN / PREVIOUS code. None when this is not such a follow-up."""
    if existing_intent not in ("WHERE", "WHEN", "PREVIOUS") or not subject_phrase:
        return None
    t = T.normalise(raw_text)
    if not _has_pronoun(t) or T.class_in(t):
        return None
    out, done = [], False
    for w in t.split():
        if not done and w in PRONOUNS:
            out.append(subject_phrase)
            done = True
        else:
            out.append(w)
    return " ".join(out) if done else None
