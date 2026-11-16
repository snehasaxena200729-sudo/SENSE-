"""
store.py - the personal_instances table (one row per personal object instance).

It lives in sense_memory.db through memory_db's connection (one database, one
connection, main thread only), like whereabouts/store.py.

Migration is additive: CREATE TABLE IF NOT EXISTS + add missing columns. No
existing table is changed. custom_object_profiles (the old "my cup" colour
fingerprint) is still written by sense_master exactly as before; this table
adds what that one lacks: a stable instance id, the item's look-alikes and its
last position.
"""

import json

import numpy as np

from .matcher import Item, Lookalike

SIG_SHAPE = (16, 20)
SIG_SIZE = SIG_SHAPE[0] * SIG_SHAPE[1]

COLUMNS = [
    ("name", "TEXT PRIMARY KEY"),     # "my cup" (same key as environmental_memory)
    ("category", "TEXT"),             # "cup"
    ("instance_id", "TEXT"),          # "cup-1731772800123", never reused
    ("taught_sig", "BLOB"),
    ("adapted_sig", "BLOB"),
    ("lookalikes", "BLOB"),           # k signatures, concatenated
    ("lookalike_places", "TEXT"),     # JSON [[world_angle, session_id], ...], same order
    ("world_angle", "REAL"),          # only meaningful inside session_id
    ("session_id", "TEXT"),
    ("last_seen", "REAL"),
    ("created_at", "REAL"),
]
NAMES = [c for c, _ in COLUMNS]


def _blob(sig):
    return None if sig is None else np.asarray(sig, np.float32).tobytes()


def _sig(blob):
    if not blob:
        return None
    return np.frombuffer(blob, np.float32).reshape(SIG_SHAPE).copy()


def _sigs(blob):
    if not blob:
        return []
    flat = np.frombuffer(blob, np.float32)
    return [flat[i:i + SIG_SIZE].reshape(SIG_SHAPE).copy() for i in range(0, len(flat) - SIG_SIZE + 1, SIG_SIZE)]


class PersonalStore:
    def __init__(self, conn):
        self.conn = conn
        self.migrate()

    def migrate(self):
        cur = self.conn.cursor()
        cur.execute("CREATE TABLE IF NOT EXISTS personal_instances (%s)" %
                    ", ".join(f"{c} {d}" for c, d in COLUMNS))
        have = {row[1] for row in cur.execute("PRAGMA table_info(personal_instances)")}
        for col, decl in COLUMNS:
            if col not in have and "PRIMARY KEY" not in decl:
                cur.execute(f"ALTER TABLE personal_instances ADD COLUMN {col} {decl}")
        self.conn.commit()

    def load(self):
        rows = self.conn.execute(f"SELECT {', '.join(NAMES)} FROM personal_instances").fetchall()
        items = []
        for row in rows:
            r = dict(zip(NAMES, row))
            taught = _sig(r["taught_sig"])
            if taught is None:
                continue
            try:
                places = json.loads(r["lookalike_places"] or "[]")
            except ValueError:
                places = []
            sigs = _sigs(r["lookalikes"])
            places = (places + [[None, None]] * len(sigs))[:len(sigs)]
            looks = [Lookalike(s, p[0], p[1]) for s, p in zip(sigs, places)]
            items.append(Item(r["name"], r["category"], taught, r["instance_id"] or f"{r['category']}-0",
                              r["created_at"] or 0.0, adapted_sig=_sig(r["adapted_sig"]),
                              lookalikes=looks, world_angle=r["world_angle"],
                              session_id=r["session_id"], last_seen=r["last_seen"]))
        return items

    def save(self, item):
        lookalikes = b"".join(_blob(l.sig) for l in item.lookalikes) or None
        places = json.dumps([[None if l.world_angle is None else float(l.world_angle), l.session_id]
                             for l in item.lookalikes])
        self.conn.execute(f"""
            INSERT INTO personal_instances ({', '.join(NAMES)})
            VALUES ({', '.join('?' * len(NAMES))})
            ON CONFLICT(name) DO UPDATE SET
                category = excluded.category, instance_id = excluded.instance_id,
                taught_sig = excluded.taught_sig, adapted_sig = excluded.adapted_sig,
                lookalikes = excluded.lookalikes, lookalike_places = excluded.lookalike_places,
                world_angle = excluded.world_angle,
                session_id = excluded.session_id, last_seen = excluded.last_seen,
                created_at = excluded.created_at
        """, (item.name, item.category, item.instance_id, _blob(item.taught_sig), _blob(item.adapted_sig),
              lookalikes, places, None if item.world_angle is None else float(item.world_angle),
              item.session_id, None if item.last_seen is None else float(item.last_seen),
              float(item.created_at or 0.0)))
        self.conn.commit()

    def forget(self, name):
        self.conn.execute("DELETE FROM personal_instances WHERE LOWER(name) = ?", (name.lower(),))
        self.conn.commit()

    def rename(self, old, new):
        with self.conn:
            self.conn.execute("DELETE FROM personal_instances WHERE LOWER(name) = ?", (new.lower(),))
            self.conn.execute("UPDATE personal_instances SET name = ? WHERE LOWER(name) = ?",
                              (new.lower(), old.lower()))
