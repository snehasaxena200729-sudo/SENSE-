"""
phrasing.py - from a stored anchor to words.

    relation_text(anchor)  -> "on the table, to the left of the laptop"
    describe(...)          -> "Your phone was on the table, to the left of the
                               laptop, 10 minutes ago. It is probably still
                               there, on your left. To face it, turn about 60
                               degrees to your left."

The confidence score, its tiers and the "it may have been moved" wording come
from answers.py (Roadmap E5), so SENSE stays exactly as honest as before; this
module only changes WHERE the place words go in the sentence.
"""

import re

import answers

SIDE_WORDS = {"left of": "to the left of", "right of": "to the right of", "next to": "next to"}


def thing_name(name):
    """'my bottle' -> 'your bottle'; 'white cup' -> 'the white cup';
    'yellow dog 2' -> 'the yellow dog' (identity.py's number is not spoken)."""
    n = re.sub(r"\s+\d+$", "", (name or "").lower().strip())
    if n.startswith("my "):
        return "your " + n[3:]
    return "the " + n


def relation_text(anchor):
    """Place words for an anchor, or None if it holds nothing useful."""
    if not anchor:
        return None
    parts = []
    if anchor.get("surface"):
        parts.append(f"on the {anchor['surface']}")
    if anchor.get("landmark") and anchor.get("relation") in SIDE_WORDS:
        parts.append(f"{SIDE_WORDS[anchor['relation']]} {thing_name(anchor['landmark'])}")
    return ", ".join(parts) or None


def _reason(status, same_session):
    """The same honesty wording as answers.describe_memory."""
    status = (status or "").upper()
    if status in ("GONE", "MISSING"):
        return "but it disappeared while I was watching, so it may have been moved"
    if status == "OUT_OF_VIEW":
        return "but I have not looked there since, so I am not sure it is still there"
    if not same_session:
        return "but that was before I was restarted, so please check"
    return "but that was a while ago, so it may have moved"


def describe(name, place, age_s, status, where, same_session, turn_hint=None):
    """name: spoken name ('your phone'); place: relation_text(); where:
    direction words now ('on your left'), only used within this run.
    Returns (sentence, level, score) like answers.describe_memory."""
    score = answers.confidence(age_s, status, same_session)
    level = answers.tier(score)
    when = answers.format_time_ago(age_s)
    subject = name[0].upper() + name[1:]
    sentence = f"{subject} was {place}, {when}."
    if level in ("confirmed", "likely"):
        sentence += " It is probably still there"
        sentence += f", {where}." if same_session and where else "."
    else:
        sentence = f"{subject} was {place}, {when}, {_reason(status, same_session)}."
        if same_session and where:
            sentence += f" That place is {where}."
    if turn_hint:
        sentence += f" To face it, {turn_hint}."
    return sentence, level, score


def earlier_note(place, anchor_age_s):
    """Appended when the anchor is older than the last sighting (stale)."""
    return f" Earlier, {answers.format_time_ago(anchor_age_s)}, it was {place}, but it may have moved since."
