"""
manager.py - HomeMap, the only class sense_master.py talks to.

    handle(text, now, known_names) -> sentence to speak, or None
        None = not a room command; SENSE's existing voice code carries on
        with the sentence exactly as before.

Every public method is wrapped: an error is printed once and the method
returns None, so a bug here can never stop SENSE or swallow a command.
"""

import functools
import time

from . import phrases as P
from . import replies as R
from .config import MappingConfig
from .doors import DoorLinks
from .objects import ObjectRooms
from .photos import RoomPhotos
from .rooms import Rooms
from .schema import migrate


def _safe(fn):
    @functools.wraps(fn)
    def inner(self, *a, **kw):
        if not self.enabled:
            return None
        try:
            return fn(self, *a, **kw)
        except Exception as e:          # never let the home map break SENSE
            if fn.__name__ not in self._errors:
                self._errors.add(fn.__name__)
                print(f"🏠 Home map {fn.__name__} error (ignored): {e!r}")
            return None
    return inner


class HomeMap:
    def __init__(self, conn, session_id, cfg=None, clock=time.time):
        self.cfg = cfg or MappingConfig()
        self.session_id = session_id
        self.clock = clock
        self.enabled = True
        self._errors = set()
        migrate(conn)                   # raises SchemaConflict -> create() switches us off
        self.rooms = Rooms(conn, clock)
        self.photos = RoomPhotos(conn, clock)
        self.doors = DoorLinks(conn, clock)
        self.objects = ObjectRooms(conn, clock)

    @_safe
    def handle(self, text, now=None, known_names=()):
        now = self.clock() if now is None else now
        cmd = P.parse(text, self.rooms.names(), known_names, self.cfg)
        if cmd is None:
            return None
        print(f"🏠 Room command: {cmd.kind}" + (f" ({cmd.room})" if cmd.room else ""))
        if cmd.kind == P.LIST:
            return R.room_list(self.rooms.names())
        if cmd.kind == P.TEACH:
            if not cmd.room:
                return R.teach_missed(cmd.heard)
            room, is_new = self.rooms.register(cmd.room, self.session_id, now)
            return R.taught(room["name"], is_new, len(self.rooms.names()))
        current = self.current_room(now)
        if cmd.kind == P.WHERE_AM_I:
            if current is None:
                return R.where_am_i(None, 0, bool(self.rooms.names()))
            return R.where_am_i(current["name"], now - current["last_declared_at"], True)
        if cmd.kind == P.GO:
            known = self.rooms.get(cmd.room) is not None
            return R.go(cmd.room, known, current["name"] if current else None)
        return None

    def current_room(self, now):
        """The room the user named in THIS run, if recent enough; else None.
        (Phase 0 has no recognition, so this is only what the user told us.)"""
        room = self.rooms.last_declared(self.session_id)
        if room is None or now - room["last_declared_at"] > self.cfg.declared_room_valid_s:
            return None
        return room


class _Disabled:
    """Stands in when the home map can't start: every call does nothing."""
    enabled = False

    def handle(self, *a, **kw):
        return None


def create(session_id, conn=None, cfg=None):
    """Make the home map on memory_db's connection; never raises."""
    try:
        if conn is None:
            import memory_db
            conn = memory_db._get_conn()
        home = HomeMap(conn, session_id, cfg)
        n = len(home.rooms.names())
        print(f"🏠 Home map on ({n} room{'s' if n != 1 else ''} registered).")
        return home
    except Exception as e:
        print(f"🏠 Home map off ({e!r})")
        return _Disabled()
