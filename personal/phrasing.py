"""
phrasing.py - understanding "my cup" in a question, and the honest extra sentence.

Rules (no model): a question names a personal item if it contains the item's
exact name ("my water bottle"), or "my <kind>" where exactly one personal item
is of that kind. Two of that kind -> ask which one. None -> say SENSE doesn't
know which one is yours, instead of answering about any cup it happens to see.
"""

import re

import spatial

IRREGULAR_PLURALS = {"mouse": "mice", "bench": "benches", "couch": "couches", "glass": "glasses"}


def plural(word):
    return IRREGULAR_PLURALS.get(word, word + "s")


def spoken(name):
    """'my cup' -> 'your cup'."""
    return "your " + name[3:] if name.startswith("my ") else "the " + name


def number(n):
    return {2: "two", 3: "three", 4: "four", 5: "five"}.get(n, str(n))


def category_in(text, known_categories, synonyms):
    """'where is my water bottle' -> 'bottle' (the kind of thing after 'my')."""
    m = re.search(r"\bmy\s+(.*)", text)
    if not m:
        return None
    words = re.sub(r"[^a-z ]", " ", m.group(1)).split()[:3]
    for w in words:
        if w in known_categories:
            return w
        if w in synonyms and synonyms[w] in known_categories:
            return synonyms[w]
    for k in range(len(words) - 1):               # "cell phone", "tv remote"
        two = f"{words[k]} {words[k + 1]}"
        if two in synonyms and synonyms[two] in known_categories:
            return synonyms[two]
        if two == "cell phone":
            return "phone"
    return None


def resolve(text, items, known_categories, synonyms):
    """('target', name) | ('say', sentence) | None (not about a personal item)."""
    t = " " + re.sub(r"[^a-z ]", " ", text.lower()) + " "
    t = re.sub(r"\s+", " ", t)
    for name in sorted(items, key=len, reverse=True):
        if f" {name} " in t:
            return "target", name
    category = category_in(t, known_categories, synonyms)
    if category is None:
        return None
    mine = sorted(n for n, it in items.items() if it.category == category)
    if len(mine) == 1:
        return "target", mine[0]
    if mine:
        listed = ", ".join(spoken(n) for n in mine[:-1]) + " or " + spoken(mine[-1])
        return "say", (f"You have {number(len(mine))} {plural(category)} I know. "
                       f"Do you mean {listed}?")
    return "say", (f"I don't know which {category} is yours yet. Point the camera at it and say: "
                   f"this is my {category}. To hear about any {category}, ask: where is the {category}?")


def direction(angle, heading):
    if angle is None:
        return "nearby"
    return spatial.direction_phrase(spatial.relative_angle(angle, heading))


def note(status, now, heading, visible_window_s=2.5):
    """The sentence added after the memory answer, or ''.

    Never says another object IS yours; says what else is in view so the user
    knows SENSE did not silently swap it."""
    if status is None or status["visible"]:
        return ""
    item = status["item"]
    cat = item.category
    parts = []
    if item.last_seen is None or now - item.last_seen > visible_window_s:
        parts.append("I can't see it right now.")
    looks = status["lookalikes"]
    if status["ambiguous"] or len(looks) >= 2:
        n = max(2, len(looks))
        parts.append(f"I can see {number(n)} {plural(cat)} that look like yours, "
                     f"so I can't tell which one is yours. Hold yours up and say: this is {item.name}.")
    elif looks:
        parts.append(f"There is a {cat} {direction(looks[0].world_angle, heading)} that looks like yours, "
                     f"but I can't be sure it is yours. If it is, say: this is {item.name}.")
    others = status["others"]
    named = [ts for ts in others if ts.locked]
    unnamed = [ts for ts in others if not ts.locked]
    for ts in named[:2]:
        parts.append(f"I can see {spoken(ts.locked)} {direction(ts.world_angle, heading)}.")
    if len(unnamed) == 1:
        parts.append(f"I can see another {cat} {direction(unnamed[0].world_angle, heading)}, "
                     f"but it doesn't look like yours.")
    elif unnamed:
        parts.append(f"I can see {number(len(unnamed))} other {plural(cat)}, but they don't look like yours.")
    return (" " + " ".join(parts)) if parts else ""
