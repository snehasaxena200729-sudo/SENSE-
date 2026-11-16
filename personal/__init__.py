"""
personal - "my cup" is ONE physical cup, never every cup.

When you say "this is my cup", SENSE binds the name to the cup instance on
camera (its look, where it is, and which other cups are around it). Later a
cup is called "your cup" only when SENSE is confident it is that same cup;
otherwise it stays "a cup", and answers say so honestly:

    "I can't see your cup right now. I last saw it in front of you, on the
     table, 2 minutes ago. There is a cup on your right that looks like yours,
     but I can't be sure it is yours."

    config.py    every threshold, with the reason for its value
    matcher.py   PersonalMatcher: the gated association (pure logic)
    store.py     the personal_instances SQLite table (additive migration)
    phrasing.py  "my <kind>" in questions, and the extra honest sentence
    manager.py   PersonalItems: the only class sense_master.py talks to

Like whereabouts/ and ocr/: no camera, no thread, no network. It only reads
what sense_master.py already computed.
"""

from .config import PersonalConfig
from .manager import PersonalItems, create
from .matcher import Item, PersonalMatcher

__all__ = ["PersonalConfig", "PersonalItems", "PersonalMatcher", "Item", "create"]
