"""
schema.py - the four home-map tables. Additive only.

    rooms         one row per room you named ("kitchen")
    room_photos   saved views of a room (filled from Phase 1; Phase 0 can store records)
    door_links    which two rooms a door connects (filled from Phase 3)
    object_rooms  the room an object was last seen in (filled from Phase 5)

Safety rules (same as whereabouts/ and personal/):
  * CREATE TABLE / INDEX IF NOT EXISTS only. Nothing is dropped, renamed,
    altered or rewritten, and no existing table is read or written.
  * Safe to run any number of times, on a new or an old database.
  * If a table with one of these names already exists WITHOUT the columns we
    need (someone else's table), we raise SchemaConflict and change nothing.
    The manager then switches the room commands off instead of touching it.

Foreign keys: room_photos, door_links and object_rooms point at rooms(id).
SQLite only enforces them when "PRAGMA foreign_keys = ON" is set on the
connection. That connection is shared with the rest of SENSE, so we do NOT
change that setting; rooms.delete() removes child rows itself instead.
object_rooms.object_name is plain text, deliberately NOT a foreign key to
environmental_memory, so nothing in the existing memory can ever be blocked or
deleted because of this package.
"""

import sqlite3

TABLES = {
    "rooms": """
        CREATE TABLE IF NOT EXISTS rooms (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE COLLATE NOCASE,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            last_declared_at REAL,          -- last "this room is ..." for this room
            last_declared_session TEXT      -- the SENSE run it was said in
        )""",
    "room_photos": """
        CREATE TABLE IF NOT EXISTS room_photos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            room_id INTEGER NOT NULL REFERENCES rooms(id) ON DELETE CASCADE,
            captured_at REAL NOT NULL,
            session_id TEXT,
            heading_deg REAL,               -- which way the camera faced
            image_path TEXT,                -- optional file on disk
            fingerprint BLOB,               -- image descriptor (Phase 1)
            landmarks TEXT                  -- e.g. "refrigerator,oven,sink"
        )""",
    "door_links": """
        CREATE TABLE IF NOT EXISTS door_links (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            room_a INTEGER NOT NULL REFERENCES rooms(id) ON DELETE CASCADE,
            room_b INTEGER NOT NULL REFERENCES rooms(id) ON DELETE CASCADE,
            door_heading_a REAL,            -- direction of the door seen from room_a
            door_heading_b REAL,            -- ... and from room_b
            walks INTEGER NOT NULL DEFAULT 1,
            created_at REAL NOT NULL,
            last_walked_at REAL NOT NULL,
            CHECK (room_a < room_b),        -- one row per pair, stored in id order
            UNIQUE (room_a, room_b)
        )""",
    "object_rooms": """
        CREATE TABLE IF NOT EXISTS object_rooms (
            object_name TEXT PRIMARY KEY,   -- same key as environmental_memory
            room_id INTEGER NOT NULL REFERENCES rooms(id) ON DELETE CASCADE,
            seen_at REAL NOT NULL,
            session_id TEXT
        )""",
}

INDEXES = (
    "CREATE INDEX IF NOT EXISTS mapping_room_photos_room ON room_photos(room_id)",
    "CREATE INDEX IF NOT EXISTS mapping_door_links_b ON door_links(room_b)",
    "CREATE INDEX IF NOT EXISTS mapping_object_rooms_room ON object_rooms(room_id)",
)

REQUIRED = {
    "rooms": {"id", "name", "created_at", "updated_at", "last_declared_at", "last_declared_session"},
    "room_photos": {"id", "room_id", "captured_at", "session_id", "heading_deg", "image_path",
                    "fingerprint", "landmarks"},
    "door_links": {"id", "room_a", "room_b", "door_heading_a", "door_heading_b", "walks",
                   "created_at", "last_walked_at"},
    "object_rooms": {"object_name", "room_id", "seen_at", "session_id"},
}


class SchemaConflict(RuntimeError):
    """A table with our name exists but is not ours. Nothing was changed."""


def columns(conn, table):
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}


def check_conflicts(conn):
    """Raise SchemaConflict if an existing table with our name lacks our columns."""
    existing = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    for table, need in REQUIRED.items():
        if table in existing:
            missing = need - columns(conn, table)
            if missing:
                raise SchemaConflict(f"table {table} already exists without {sorted(missing)}; "
                                     f"left untouched")


def migrate(conn):
    """Create the four tables and their indexes if missing. Idempotent."""
    check_conflicts(conn)            # before writing anything
    try:
        for ddl in TABLES.values():
            conn.execute(ddl)
        for ddl in INDEXES:
            conn.execute(ddl)
        conn.commit()
    except sqlite3.Error:
        conn.rollback()
        raise
