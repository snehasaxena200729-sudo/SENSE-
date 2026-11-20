"""doors.py - door connections between rooms (table: door_links).

A link is undirected (kitchen <-> hall), stored once with the smaller room id
first. Walking through the same door again increases `walks`, so later phases
can ignore a link that was only seen once by mistake."""

import time


class DoorLinks:
    COLS = ("id", "room_a", "room_b", "door_heading_a", "door_heading_b", "walks",
            "created_at", "last_walked_at")

    def __init__(self, conn, clock=time.time):
        self.conn = conn
        self.clock = clock

    def link(self, room_x, room_y, heading_from_x=None, heading_from_y=None, now=None):
        """Record a door between two room ids. Returns the link row."""
        if room_x == room_y:
            raise ValueError("a door links two different rooms")
        now = self.clock() if now is None else now
        a, b, ha, hb = (room_x, room_y, heading_from_x, heading_from_y) if room_x < room_y \
            else (room_y, room_x, heading_from_y, heading_from_x)
        self.conn.execute(
            "INSERT INTO door_links (room_a, room_b, door_heading_a, door_heading_b, walks, "
            "created_at, last_walked_at) VALUES (?, ?, ?, ?, 1, ?, ?) "
            "ON CONFLICT(room_a, room_b) DO UPDATE SET walks = walks + 1, "
            "last_walked_at = excluded.last_walked_at, "
            "door_heading_a = COALESCE(excluded.door_heading_a, door_heading_a), "
            "door_heading_b = COALESCE(excluded.door_heading_b, door_heading_b)",
            (a, b, ha, hb, now, now))
        self.conn.commit()
        return self.get(room_x, room_y)

    def get(self, room_x, room_y):
        a, b = sorted((room_x, room_y))
        r = self.conn.execute(f"SELECT {', '.join(self.COLS)} FROM door_links "
                              f"WHERE room_a = ? AND room_b = ?", (a, b)).fetchone()
        return dict(zip(self.COLS, r)) if r else None

    def neighbours(self, room_id, min_walks=1):
        """[(other room id, door heading seen from room_id, walks)]"""
        out = []
        for r in self.conn.execute(
                "SELECT room_a, room_b, door_heading_a, door_heading_b, walks FROM door_links "
                "WHERE (room_a = ? OR room_b = ?) AND walks >= ?", (room_id, room_id, min_walks)):
            a, b, ha, hb, w = r
            out.append((b, ha, w) if a == room_id else (a, hb, w))
        return out
