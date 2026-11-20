"""rooms.py - room management (table: rooms)."""

import time


class Rooms:
    COLS = ("id", "name", "created_at", "updated_at", "last_declared_at", "last_declared_session")

    def __init__(self, conn, clock=time.time):
        self.conn = conn
        self.clock = clock

    def _row(self, r):
        return dict(zip(self.COLS, r)) if r else None

    def get(self, name):
        r = self.conn.execute(f"SELECT {', '.join(self.COLS)} FROM rooms WHERE name = ? COLLATE NOCASE",
                              (name.strip().lower(),)).fetchone()
        return self._row(r)

    def get_by_id(self, room_id):
        r = self.conn.execute(f"SELECT {', '.join(self.COLS)} FROM rooms WHERE id = ?",
                              (room_id,)).fetchone()
        return self._row(r)

    def all(self):
        """Rooms in the order they were registered."""
        return [self._row(r) for r in self.conn.execute(
            f"SELECT {', '.join(self.COLS)} FROM rooms ORDER BY created_at, id")]

    def names(self):
        return [r["name"] for r in self.all()]

    def register(self, name, session_id=None, now=None):
        """Add the room if new, and note that the user just said they are in it.

        Returns (room, is_new)."""
        now = self.clock() if now is None else now
        name = name.strip().lower()
        room = self.get(name)
        if room is None:
            self.conn.execute(
                "INSERT INTO rooms (name, created_at, updated_at, last_declared_at, last_declared_session) "
                "VALUES (?, ?, ?, ?, ?)", (name, now, now, now, session_id))
            is_new = True
        else:
            self.conn.execute(
                "UPDATE rooms SET updated_at = ?, last_declared_at = ?, last_declared_session = ? "
                "WHERE id = ?", (now, now, session_id, room["id"]))
            is_new = False
        self.conn.commit()
        return self.get(name), is_new

    def last_declared(self, session_id):
        """The room most recently named in this SENSE run, or None."""
        r = self.conn.execute(
            f"SELECT {', '.join(self.COLS)} FROM rooms WHERE last_declared_session = ? "
            f"ORDER BY last_declared_at DESC LIMIT 1", (session_id,)).fetchone()
        return self._row(r)

    def delete(self, name):
        """Remove a room and every row that points at it (foreign keys are not
        switched on for the shared connection, so we cascade by hand)."""
        room = self.get(name)
        if room is None:
            return False
        rid = room["id"]
        self.conn.execute("DELETE FROM room_photos WHERE room_id = ?", (rid,))
        self.conn.execute("DELETE FROM door_links WHERE room_a = ? OR room_b = ?", (rid, rid))
        self.conn.execute("DELETE FROM object_rooms WHERE room_id = ?", (rid,))
        self.conn.execute("DELETE FROM rooms WHERE id = ?", (rid,))
        self.conn.commit()
        return True
