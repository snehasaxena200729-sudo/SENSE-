"""
mapping/ - SENSE learns the rooms of your home (a room-level map).

Phase 0 (this version) is groundwork only. It adds four database tables and
four voice commands; nothing in SENSE's camera, tracking, memory, face, audio
or question-answering code is changed.

    "This room is the kitchen."   -> registers the room (saves its name)
    "Where am I?"                 -> the room you last told SENSE about, honestly
    "Take me to the bedroom."     -> says whether it knows that room (no guidance yet)
    "Which rooms do you know?"    -> lists the registered rooms

Files (one responsibility each):
    config.py      settings and room word lists
    schema.py      the four tables; additive, safe to run any number of times
    rooms.py       room management                    (table: rooms)
    photos.py      room photo records                 (table: room_photos)
    doors.py       door connections between rooms     (table: door_links)
    objects.py     which room an object was seen in   (table: object_rooms)
    phrases.py     which spoken sentences are room commands (pure functions)
    replies.py     the spoken answers
    manager.py     HomeMap: the only class sense_master.py talks to

Rules: no thread, no camera, no network, no new model. If anything here fails,
the room commands are switched off and SENSE runs exactly as before.
"""

from .config import MappingConfig
from .manager import HomeMap, create
from .phrases import parse

__all__ = ["MappingConfig", "HomeMap", "create", "parse"]
