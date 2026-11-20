"""
phrases.py - which spoken sentences are home-map commands. Pure functions.

    parse(text, known_rooms=(), known_names=()) -> Command or None

None means "not a room command": SENSE's existing voice code handles the
sentence exactly as before. When in doubt we return None. Phase 0 owns only:

    TEACH       "this room is the kitchen", "call this room the kitchen"
    WHERE_AM_I  "where am I", "which room am I in", "what room is this"
    GO          "take me to the bedroom", "how do I get to the kitchen"
                (only if the place is a room word or a registered room)
    LIST        "which rooms do you know", "list the rooms"

Clash rules, each checked by the stress test:
  * "this is Sneha" / "this is my bottle" / "this is the kitchen" are NOT
    room commands. The naming command keeps "this is ...". Teaching a room
    needs the word "room": "this room is the kitchen".
  * "take me to my phone", "take me to the door", "take me to Sneha" stay
    with the existing find command: a place that is an object word, or the
    name of a person or object SENSE knows, is never treated as a room.
  * "what's in this room" / "describe the room" stay with the overview.
"""

import difflib
import re
from dataclasses import dataclass
from typing import Optional

from .config import MappingConfig

TEACH, WHERE_AM_I, GO, LIST = "TEACH", "WHERE_AM_I", "GO", "LIST"
KINDS = (TEACH, WHERE_AM_I, GO, LIST)


@dataclass
class Command:
    kind: str
    room: Optional[str] = None   # room name (lower case), or None if not caught
    heard: str = ""              # the words after the command, as heard


_DEFAULT = MappingConfig()
_LEAD = re.compile(r"^(?:(?:hey|ok|okay|so|um|uh|please|sense|and|well|now|can you|could you|"
                   r"would you|just)\s+)+")
_TRAIL = re.compile(r"(?:\s+(?:please|right now|now|sense|thanks|thank you|okay|ok))+$")
_ARTICLE = re.compile(r"^(?:the|a|an|our)\s+")

_WHERE_AM_I = re.compile(
    r"^(?:where am i|where are we|(?:which|what) room (?:am i|are we)(?: in)?(?: right now| now)?|"
    r"(?:which|what) room is (?:this|it|here)|(?:tell me )?(?:which|what) room i am in|"
    r"where am i (?:right )?now|where am i in the house)$")
_LIST = re.compile(
    r"^(?:(?:tell me )?(?:which|what) rooms (?:do you know|have you learned|have you learnt|"
    r"do you have|are there|did i teach you|have i taught you|do you remember)|"
    r"list (?:the |all |my |all the )?rooms|(?:how many|name the|name all the|tell me the) rooms"
    r"(?: do you know| you know)?)$")
_GO = re.compile(
    r"^(?:take me|guide me|lead me|bring me|walk me|get me|navigate me|"
    r"how do i get|how do i go|how can i get|how can i go|help me get|show me the way)"
    r"\s+(?:back\s+)?(?:to|into)\s+(?P<n>.+)$")
_TEACH = re.compile(
    r"^(?:this room is(?: called)?|this room's name is|the name of this room is|"
    r"(?:call|name) this room|remember this room as|save this room as)(?:\s+(?P<n>.+))?$")


def normalise(text, cfg=_DEFAULT):
    t = (text or "").lower().replace("’", "'")
    t = re.sub(r"[^a-z0-9' \-]", " ", t)
    t = t.replace("-", " ")
    t = re.sub(r"\s+", " ", t).strip(" '")
    for a, b in sorted(cfg.spoken_fixes.items(), key=lambda kv: -len(kv[0])):
        t = re.sub(r"\b%s\b" % re.escape(a), b, t)
    t = _LEAD.sub("", t)
    t = _TRAIL.sub("", t)
    return t.strip()


def _clean(raw):
    return re.sub(r"\s+", " ", _TRAIL.sub("", (raw or "").strip(" '"))).strip()


def room_name(raw, known_rooms=(), known_names=(), cfg=_DEFAULT, explicit=False):
    """The room a spoken place refers to, or None.

    Order: a registered room, a room word, "<someone>'s room" / "my room",
    a close spelling of a registered room or room word. explicit=True (after
    "this room is") also accepts any other short name, e.g. "the den".
    A person or object SENSE knows by that name is never a room.
    """
    full = _clean(raw)
    stripped = _ARTICLE.sub("", full)
    if not stripped:
        return None
    known = {k.lower() for k in known_rooms}
    taken = {n.lower() for n in known_names}
    words = stripped.split()
    if len(words) > cfg.max_name_words:
        return None
    bare = re.sub(r"^my\s+", "", stripped)
    if stripped in taken or bare in taken or bare in cfg.not_rooms or stripped in cfg.not_rooms:
        return None
    if stripped in known:
        return stripped
    if stripped in cfg.room_words:
        return stripped
    if bare in cfg.room_words and stripped != bare:
        return bare                           # "my kitchen" -> "kitchen"
    if len(words) >= 2 and words[-1] == "room" and not set(words[:-1]) & cfg.not_rooms:
        return stripped                       # "sneha's room", "my room", "tv room"
    pool = sorted(known) + [w for w in cfg.room_words if w not in known]
    close = difflib.get_close_matches(bare, pool, n=1, cutoff=cfg.fuzzy_cutoff)
    if close:
        return close[0]
    if explicit and all(re.fullmatch(r"[a-z][a-z']*", w) for w in words):
        return stripped
    return None


def parse(text, known_rooms=(), known_names=(), cfg=_DEFAULT):
    """Command for a home-map sentence, or None (= not ours, leave it alone)."""
    t = normalise(text, cfg)
    if not t:
        return None
    if _LIST.match(t):
        return Command(LIST)
    if _WHERE_AM_I.match(t):
        return Command(WHERE_AM_I)
    m = _TEACH.match(t)
    if m:
        heard = _clean(m.group("n"))
        return Command(TEACH, room_name(heard, known_rooms, known_names, cfg, explicit=True), heard)
    m = _GO.match(t)
    if m:
        heard = _clean(m.group("n"))
        room = room_name(heard, known_rooms, known_names, cfg)
        return Command(GO, room, heard) if room else None
    return None
