"""
answers.py - Understanding questions and answering honestly (Roadmap E5, E6)

1) INTENT CLASSIFICATION (E6) - rule based, on purpose.
   We match key phrases in a fixed priority order. A local LLM could accept
   freer wording, but it needs GBs of RAM, takes seconds on a laptop CPU and
   can invent facts that are not in memory. Rules are fast, offline and
   predictable, which matters more for an assistive device.

2) FUZZY NAME MATCHING (E6) - difflib.get_close_matches (standard library,
   Ratcliff/Obershelp similarity) corrects small speech-recognition errors
   ("sneh" -> "sneha") against names that are actually in memory.

3) CONFIDENCE (E5) - a score in [0, 1] that combines:
       recency       exp(-age / HALF_LIFE)  (old sightings are less reliable)
       last status   GONE lowers it a lot, OUT_OF_VIEW a little
       session       a sighting from a previous run gets a penalty
   and is mapped to three tiers: confirmed / likely / uncertain.
   Each tier has its own wording, so the user knows when to double-check.
"""

import difflib
import math

# ----------------------------------------------------------------------
# Intents (checked in this order: first match wins)
# ----------------------------------------------------------------------
INTENT_PATTERNS = [
    ("FORGET",   ["forget"]),
    ("LEARN",    ["this is", "my name is", "remember as", "call this", "call it"]),
    ("QUIET",    ["be quiet", "quiet mode", "stop talking", "silent mode", "fewer alerts"]),
    ("VERBOSE",  ["talk more", "verbose", "tell me everything", "more alerts"]),
    ("NORMAL",   ["normal mode", "normal alerts"]),
    ("SLOWER",   ["speak slower", "talk slower", "slow down"]),
    ("FASTER",   ["speak faster", "talk faster", "speed up"]),
    ("CHANGES",  ["what changed", "what has changed", "what's changed", "anything changed",
                  "what is different", "what's different", "anything different", "what happened"]),
    ("WHO",      ["who was here", "who came", "who is here", "who's here", "who did you see",
                  "anyone here", "anybody here"]),
    ("WHEN",     ["when did", "when was", "what time"]),
    ("PREVIOUS", ["where was", "used to be", "before"]),
    ("RECENT",   ["what did you see", "what have you seen", "seen recently", "recently"]),
    ("OVERVIEW", ["what do you see", "room", "everything", "around me", "what's around", "what is around"]),
]


def classify_intent(text):
    t = text.lower()
    for intent, phrases in INTENT_PATTERNS:
        if any(p in t for p in phrases):
            return intent
    return "WHERE"


STOP_WORDS = {"where", "is", "my", "the", "a", "an", "was", "did", "you", "last", "see",
              "when", "it", "this", "that", "me", "please", "now", "here", "there", "before",
              "of", "to", "what", "who", "forget", "about", "have", "has", "been"}


def fuzzy_find(text, names, cutoff=0.78):
    """Return the known name best matching the spoken text, or None.

    First tries exact substring (multi-word names like 'white cup'), then
    word-by-word fuzzy matching for recognition errors.
    """
    t = text.lower()
    by_length = sorted(names, key=len, reverse=True)
    for n in by_length:
        if n.lower() in t:
            return n
    lower_map = {}
    for n in names:
        for part in n.lower().split():
            if part not in STOP_WORDS:
                lower_map.setdefault(part, n)
    for word in t.replace("?", " ").replace(".", " ").replace(",", " ").split():
        if word in STOP_WORDS or len(word) < 3:
            continue
        match = difflib.get_close_matches(word, list(lower_map), n=1, cutoff=cutoff)
        if match:
            return lower_map[match[0]]
    return None


# ----------------------------------------------------------------------
# Confidence (E5)
# ----------------------------------------------------------------------
LIVE_WINDOW_S = 2.5
HALF_LIFE_S = 600.0          # after ~10 minutes confidence has halved
STATUS_FACTOR = {"VISIBLE": 1.0, "MOVED": 1.0, "RETURNED": 1.0,
                 "OUT_OF_VIEW": 0.8, "MISSING": 0.3, "GONE": 0.3}
OLD_SESSION_FACTOR = 0.6


def confidence(age_s, status, same_session):
    if age_s <= LIVE_WINDOW_S:
        return 1.0
    score = math.exp(-age_s * math.log(2) / HALF_LIFE_S)
    score *= STATUS_FACTOR.get((status or "VISIBLE").upper(), 0.8)
    if not same_session:
        score *= OLD_SESSION_FACTOR
    return round(score, 3)


def tier(score):
    if score >= 0.999:
        return "confirmed"
    if score >= 0.6:
        return "likely"
    return "uncertain"


def format_time_ago(seconds):
    seconds = int(max(0, seconds))
    if seconds < 60:
        return f"{seconds} seconds ago"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
    hours = minutes // 60
    if hours < 48:
        return f"{hours} hour{'s' if hours != 1 else ''} ago"
    days = hours // 24
    return f"{days} days ago"


def describe_memory(name, age_s, status, where, same_session, turn_hint=None, relation=None):
    """Build an uncertainty-aware sentence about a remembered item.

    where: direction words relative to the user now (or the old zone)
    """
    score = confidence(age_s, status, same_session)
    level = tier(score)
    place = where + (f", {relation}" if relation else "")
    when = format_time_ago(age_s)
    status = (status or "").upper()
    if level == "confirmed":
        sentence = f"I can see {name} {place}."
    elif level == "likely":
        sentence = f"I saw {name} {place} {when}. It is probably still there."
    else:
        if status in ("GONE", "MISSING"):
            reason = "but it disappeared while I was watching, so it may have been moved"
        elif status == "OUT_OF_VIEW":
            reason = "but I have not looked there since, so I am not sure it is still there"
        elif not same_session:
            reason = "but that was before I was restarted, so please check"
        else:
            reason = "but that was a while ago, so it may have moved"
        sentence = f"I last saw {name} {place} {when}, {reason}."
    if turn_hint and level != "confirmed":
        sentence += f" To face it, {turn_hint}."
    return sentence, level, score


EVENT_WORDS = {
    "MOVED": "was moved",
    "GONE": "disappeared",
    "RETURNED": "came back",
    "APPEARED": "appeared",
}


def summarise_changes(events, max_items=4):
    """'Two things changed: the white cup was moved ..., my bottle disappeared.'

    events: list of dicts with name, type, (optional) zone. Only the LAST event
    per item is reported - that is its current situation.
    """
    latest = {}
    for ev in events:
        if ev["type"] in EVENT_WORDS:
            latest[ev["name"]] = ev
    if not latest:
        return "Nothing has changed that I noticed."
    items = sorted(latest.values(), key=lambda e: e["ts"], reverse=True)
    parts = []
    for ev in items[:max_items]:
        words = EVENT_WORDS[ev["type"]]
        where = ""
        if ev.get("zone") and ev["type"] == "MOVED":
            where = f", now {ev['zone']}"
        elif ev.get("zone") and ev["type"] == "APPEARED":
            where = f" {ev['zone']}"
        parts.append(f"{ev['name']} {words}{where}")
    count = len(items)
    head = "One thing changed" if count == 1 else f"{count} things changed"
    extra = f", and {count - max_items} more" if count > max_items else ""
    return f"{head}: " + "; ".join(parts) + extra + "."
