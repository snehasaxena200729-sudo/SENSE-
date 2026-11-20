"""objects.py - which room an object was last seen in (table: object_rooms).

Keyed by the same lower-case SENSE name as environmental_memory ("my bottle"),
but kept in its own table: nothing in environmental_memory is read or changed."""

import time


class ObjectRooms:
    def __init__(self, conn, clock=time.time):
        self.conn = conn
        self.clock = clock

    def set(self, object_name, room_id, session_id=None, now=None):
        now = self.clock() if now is None else now
        self.conn.execute(
            "INSERT INTO object_rooms (object_name, room_id, seen_at, session_id) VALUES (?, ?, ?, ?) "
            "ON CONFLICT(object_name) DO UPDATE SET room_id = excluded.room_id, "
            "seen_at = excluded.seen_at, session_id = excluded.session_id",
            (object_name.strip().lower(), room_id, now, session_id))
        self.conn.commit()

    def get(self, object_name):
        r = self.conn.execute(
            "SELECT o.object_name, o.room_id, r.name, o.seen_at, o.session_id FROM object_rooms o "
            "JOIN rooms r ON r.id = o.room_id WHERE o.object_name = ?",
            (object_name.strip().lower(),)).fetchone()
        if not r:
            return None
        return dict(zip(("object_name", "room_id", "room_name", "seen_at", "session_id"), r))

    def in_room(self, room_id):
        return [r[0] for r in self.conn.execute(
            "SELECT object_name FROM object_rooms WHERE room_id = ? ORDER BY seen_at DESC", (room_id,))]

    def forget(self, object_name):
        self.conn.execute("DELETE FROM object_rooms WHERE object_name = ?", (object_name.strip().lower(),))
        self.conn.commit()

    def rename(self, old, new):
        o, n = old.strip().lower(), new.strip().lower()
        if o == n:
            return
        self.conn.execute("DELETE FROM object_rooms WHERE object_name = ?", (n,))
        self.conn.execute("UPDATE object_rooms SET object_name = ? WHERE object_name = ?", (n, o))
        self.conn.commit()
