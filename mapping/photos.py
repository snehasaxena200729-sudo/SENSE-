"""photos.py - room photo records (table: room_photos).

Phase 0 only stores and reads records; capturing views while you turn round
comes in Phase 1."""

import time


class RoomPhotos:
    COLS = ("id", "room_id", "captured_at", "session_id", "heading_deg", "image_path",
            "fingerprint", "landmarks")

    def __init__(self, conn, clock=time.time):
        self.conn = conn
        self.clock = clock

    def add(self, room_id, heading_deg=None, image_path=None, fingerprint=None,
            landmarks=(), session_id=None, now=None):
        now = self.clock() if now is None else now
        lm = ",".join(sorted({str(x).lower() for x in landmarks})) if landmarks else None
        cur = self.conn.execute(
            "INSERT INTO room_photos (room_id, captured_at, session_id, heading_deg, image_path, "
            "fingerprint, landmarks) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (room_id, now, session_id, heading_deg, image_path,
             None if fingerprint is None else bytes(fingerprint), lm))
        self.conn.commit()
        return cur.lastrowid

    def for_room(self, room_id):
        rows = self.conn.execute(
            f"SELECT {', '.join(self.COLS)} FROM room_photos WHERE room_id = ? ORDER BY captured_at, id",
            (room_id,)).fetchall()
        out = []
        for r in rows:
            d = dict(zip(self.COLS, r))
            d["landmarks"] = d["landmarks"].split(",") if d["landmarks"] else []
            out.append(d)
        return out

    def count(self, room_id):
        return self.conn.execute("SELECT COUNT(*) FROM room_photos WHERE room_id = ?",
                                 (room_id,)).fetchone()[0]

    def clear(self, room_id):
        self.conn.execute("DELETE FROM room_photos WHERE room_id = ?", (room_id,))
        self.conn.commit()
