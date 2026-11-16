"""
interactive/context.py - remembering what "it" means.

A tiny dialogue state: the last subject the user asked about (a class such
as "bottle", a name such as "my bottle" or "Sneha", and - when exactly one
was visible - its ByteTrack id). It expires after SUBJECT_TTL_S, so an "it"
asked ten minutes later is not silently bound to an old topic.
"""
from dataclasses import dataclass
from typing import Optional

from . import config as C
from .text import CLASS_WORDS


@dataclass
class Subject:
    cls: Optional[str] = None
    name: Optional[str] = None      # lower case: "my bottle", "white cup", "sneha"
    tid: Optional[int] = None
    time: float = 0.0


class DialogueContext:
    def __init__(self, ttl=C.SUBJECT_TTL_S):
        self.ttl = ttl
        self.subject = None

    def set(self, now, cls=None, name=None, tid=None):
        if cls is None and name is None:
            return
        self.subject = Subject(cls, name.lower() if name else None, tid, now)

    def get(self, now):
        s = self.subject
        if s is None or now - s.time > self.ttl:
            return None
        return s

    def phrase(self, now):
        """Words to put in place of "it": 'my bottle', 'sneha', 'the bottle'."""
        s = self.get(now)
        if s is None:
            return None
        if s.name:
            last = s.name.split()[-1]
            if s.name.startswith("my ") or last not in CLASS_WORDS:
                return s.name                 # "my bottle", "sneha"
            return "the " + s.name            # "the white bottle"
        return f"the {s.cls}" if s.cls else None

    def clear(self):
        self.subject = None
