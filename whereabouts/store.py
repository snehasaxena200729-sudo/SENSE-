"""
store.py - the item_anchors table.

One row per item (the latest VOTED anchor). It lives in the same SQLite file as
the rest of SENSE's memory (sense_memory.db), through memory_db's connection,
so there is still one database and one connection on the main thread.

Migration: CREATE TABLE IF NOT EXISTS + add-missing-columns. It never drops,
resets or rewrites anything that already exists, so running a new version on an
old database (or an old version on a new one) is safe. Other tables are not
touched. The tracker never stores ByteTrack ids, which restart from 1 every run;
the key is the item's SENSE name ("my bottle", "white cup"), the same key as
environmental_memory.
"""

import sqlite3

COLUMNS = [
    ("item_name", "TEXT PRIMARY KEY"),
    ("item_category", "TEXT"),
    ("surface", "TEXT"),             # "table" / "desk" / "bed" / "couch" or NULL
    ("surface_votes", "INTEGER"),
    ("landmark", "TEXT"),            # SENSE name of the landmark ("laptop", "my bottle") or NULL
    ("landmark_category", "TEXT"),
    ("relation", "TEXT"),            # "left of" / "right of" / "next to" or NULL
    ("landmark_votes", "INTEGER"),
    ("zone", "TEXT"),                # existing zone words at that moment
    ("world_angle", "REAL"),         # only meaningful inside session_id
    ("session_id", "TEXT"),
    ("seen_at", "REAL"),             # last time this anchor was confirmed
    ("first_seen_at", "REAL"),       # when this anchor (this placement) was first confirmed
]
NAMES = [c for c, _ in COLUMNS]


class AnchorStore:
    def __init__(self, conn):
        self.conn = conn
        self.migrate()

    def migrate(self):
        cur = self.conn.cursor()
        cur.execute("CREATE TABLE IF NOT EXISTS item_anchors (%s)" %
                    ", ".join(f"{c} {d}" for c, d in COLUMNS))
        have = {row[1] for row in cur.execute("PRAGMA table_info(item_anchors)")}
        for col, decl in COLUMNS:
            if col not in have and "PRIMARY KEY" not in decl:
                cur.execute(f"ALTER TABLE item_anchors ADD COLUMN {col} {decl}")
        self.conn.commit()

    def get(self, name):
        try:
            row = self.conn.execute(
                f"SELECT {', '.join(NAMES)} FROM item_anchors WHERE item_name = ?",
                (name.lower(),)).fetchone()
        except sqlite3.Error:
            return None
        return dict(zip(NAMES, row)) if row else None

    def save(self, anchor):
        row = {c: anchor.get(c) for c in NAMES}
        row["item_name"] = row["item_name"].lower()
        self.conn.execute(
            f"INSERT INTO item_anchors ({', '.join(NAMES)}) VALUES ({', '.join('?' * len(NAMES))}) "
            f"ON CONFLICT(item_name) DO UPDATE SET " +
            ", ".join(f"{c} = excluded.{c}" for c in NAMES if c != "item_name"),
            [row[c] for c in NAMES])
        self.conn.commit()

    def forget(self, name):
        """Privacy: remove the item's anchor and its use as anyone's landmark."""
        n = name.lower()
        self.conn.execute("DELETE FROM item_anchors WHERE item_name = ?", (n,))
        self.conn.execute("UPDATE item_anchors SET landmark = NULL, landmark_category = NULL, "
                          "relation = NULL WHERE landmark = ?", (n,))
        self.conn.commit()

    def rename(self, old, new):
        o, n = old.lower(), new.lower()
        if o == n:
            return
        self.conn.execute("DELETE FROM item_anchors WHERE item_name = ?", (n,))
        self.conn.execute("UPDATE item_anchors SET item_name = ? WHERE item_name = ?", (n, o))
        self.conn.execute("UPDATE item_anchors SET landmark = ? WHERE landmark = ?", (n, o))
        self.conn.commit()
