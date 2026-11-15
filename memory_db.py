import sqlite3
import time
import os
import numpy as np

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sense_memory.db")

# One connection reused for the whole run (opening/closing SQLite on every
# frame was the main efficiency cost). All DB calls happen on the MAIN thread
# since Phase 0, so sharing it is safe.
_conn = None


def _get_conn():
    global _conn
    if _conn is None:
        _conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        _conn.execute("PRAGMA journal_mode=WAL")      # faster, crash-safe writes
        _conn.execute("PRAGMA synchronous=NORMAL")
    return _conn


def close():
    global _conn
    if _conn is not None:
        _conn.commit()
        _conn.close()
        _conn = None


def _num(v):
    """numpy scalars -> plain float (SQLite would store numpy values as BLOBs)."""
    return None if v is None else float(v)


def _add_column_if_missing(cursor, table, column, decl):
    cols = [r[1] for r in cursor.execute(f"PRAGMA table_info({table})")]
    if column not in cols:
        cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")

def init_database():
    """Initializes tables and safely cleans any old ghost names."""
    conn = _get_conn()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS environmental_memory (
            object_name TEXT PRIMARY KEY,
            last_zone TEXT,
            last_x INTEGER,
            last_y INTEGER,
            last_seen REAL,
            status TEXT
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS people_profiles (
            name TEXT PRIMARY KEY,
            face_hist BLOB,
            cloth_hist BLOB,
            created_at REAL
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS custom_object_profiles (
            name TEXT PRIMARY KEY,
            base_category TEXT,
            signature BLOB,
            created_at REAL
        )
    """)
    # --- Roadmap E1: stable entities (one row per physical object) ---
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS entities (
            entity_id TEXT PRIMARY KEY,
            name TEXT,
            category TEXT,
            color TEXT,
            signature BLOB,
            last_seen REAL
        )
    """)
    # --- Roadmap E2: append-only event history ---
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts REAL,
            name TEXT,
            event TEXT,
            zone TEXT,
            world_angle REAL,
            session_id TEXT,
            dist_feet REAL,
            detail TEXT
        )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_name_ts ON events(name, ts)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts)")
    # --- Roadmap O2: user preferences ---
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    """)
    # --- Roadmap E4: world-anchored position for the latest-state table ---
    _add_column_if_missing(cursor, "environmental_memory", "world_angle", "REAL")
    _add_column_if_missing(cursor, "environmental_memory", "session_id", "TEXT")
    _add_column_if_missing(cursor, "environmental_memory", "dist_feet", "REAL")

    # -------------------------------------------------------------
    # SAFELY PURGE GHOSTS (Zero constraint conflicts!)
    # -------------------------------------------------------------
    cursor.execute("DELETE FROM environmental_memory WHERE object_name IN ('neha', 'snail', 'now')")
    cursor.execute("DELETE FROM people_profiles WHERE name IN ('Neha', 'Snail', 'Now')")

    conn.commit()
    print("💾 SENSE Database initialized cleanly (Ghosts purged).")

def save_object_observation(name, zone, x, y, status="VISIBLE", world_angle=None,
                            session_id=None, dist_feet=None, seen_at=None):
    conn = _get_conn()
    cursor = conn.cursor()
    current_time = seen_at if seen_at is not None else time.time()
    world_angle = _num(world_angle)
    dist_feet = _num(dist_feet)
    x, y = int(x), int(y)

    cursor.execute("""
        INSERT INTO environmental_memory (object_name, last_zone, last_x, last_y, last_seen, status,
                                          world_angle, session_id, dist_feet)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(object_name) DO UPDATE SET
            last_zone = excluded.last_zone,
            last_x = excluded.last_x,
            last_y = excluded.last_y,
            last_seen = excluded.last_seen,
            status = excluded.status,
            world_angle = excluded.world_angle,
            session_id = excluded.session_id,
            dist_feet = excluded.dist_feet
    """, (name.lower(), zone, x, y, current_time, status, world_angle, session_id, dist_feet))

    conn.commit()


def set_object_status(name, status):
    """Change only the status (e.g. GONE) without touching the last position/time."""
    conn = _get_conn()
    conn.execute("UPDATE environmental_memory SET status = ? WHERE object_name = ?",
                 (status, name.lower()))
    conn.commit()

def get_object_memory(name):
    conn = _get_conn()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT last_zone, last_seen, status 
        FROM environmental_memory 
        WHERE LOWER(object_name) = ?
    """, (name.lower(),))
    
    row = cursor.fetchone()

    if row:
        return {
            "last_zone": row[0],
            "last_seen": row[1],
            "status": row[2]
        }
    return None

def get_all_remembered_items():
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute("SELECT object_name, last_zone, last_seen, status FROM environmental_memory")
    rows = cursor.fetchall()
    return rows

def save_person_visual_profile(name, face_hist, cloth_hist):
    """Permanently saves a person's visual fingerprint into SQLite!"""
    conn = _get_conn()
    cursor = conn.cursor()
    current_time = time.time()

    face_bytes = face_hist.tobytes()
    cloth_bytes = cloth_hist.tobytes()

    cursor.execute("""
        INSERT INTO people_profiles (name, face_hist, cloth_hist, created_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(name) DO UPDATE SET
            face_hist = excluded.face_hist,
            cloth_hist = excluded.cloth_hist,
            created_at = excluded.created_at
    """, (name.capitalize(), face_bytes, cloth_bytes, current_time))

    conn.commit()
    print(f"💾 Permanently saved visual profile for {name} in SQLite database!")

def load_all_people_visual_profiles():
    """Loads all known people and their visual signatures from SQLite on startup!"""
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute("SELECT name, face_hist, cloth_hist FROM people_profiles")
    rows = cursor.fetchall()

    people_dict = {}
    for name, f_bytes, c_bytes in rows:
        face_hist = np.frombuffer(f_bytes, dtype=np.float32).reshape((16, 20))
        cloth_hist = np.frombuffer(c_bytes, dtype=np.float32).reshape((16, 20))
        people_dict[name] = (face_hist, cloth_hist)

    return people_dict
def save_custom_object_profile(name, base_category, signature):
    """Saves a personal item's visual fingerprint into SQLite."""
    conn = _get_conn()
    cursor = conn.cursor()
    sig_bytes = signature.tobytes()

    cursor.execute("""
        INSERT INTO custom_object_profiles (name, base_category, signature, created_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(name) DO UPDATE SET
            base_category = excluded.base_category,
            signature = excluded.signature,
            created_at = excluded.created_at
    """, (name.lower(), base_category.lower(), sig_bytes, time.time()))

    conn.commit()
    print(f"💾 Permanently memorized custom object: '{name}' in SQLite!")

def load_all_custom_object_profiles():
    """Loads all personal items on startup."""
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute("SELECT name, base_category, signature FROM custom_object_profiles")
    rows = cursor.fetchall()

    objects_dict = {}
    for name, cat, s_bytes in rows:
        sig = np.frombuffer(s_bytes, dtype=np.float32).reshape((16, 20))
        objects_dict[name] = {"category": cat, "sig": sig}
    return objects_dict


# =====================================================================
# Roadmap additions
# =====================================================================
_MEMORY_COLS = "object_name, last_zone, last_x, last_y, last_seen, status, world_angle, session_id, dist_feet"


def _memory_row_to_dict(row):
    keys = ["name", "last_zone", "last_x", "last_y", "last_seen", "status",
            "world_angle", "session_id", "dist_feet"]
    return dict(zip(keys, row))


def find_latest_memory(target):
    """Most recently seen memory whose name CONTAINS target ('bottle' ->
    'white bottle', 'my bottle', ...). Fixes answers that used to read a stale
    exact-name row."""
    cur = _get_conn().execute(
        f"SELECT {_MEMORY_COLS} FROM environmental_memory WHERE object_name LIKE ? "
        "ORDER BY last_seen DESC LIMIT 1", (f"%{target.lower()}%",))
    row = cur.fetchone()
    return _memory_row_to_dict(row) if row else None


def get_memories_seen_since(ts):
    cur = _get_conn().execute(
        f"SELECT {_MEMORY_COLS} FROM environmental_memory WHERE last_seen >= ? "
        "ORDER BY last_seen DESC", (ts,))
    return [_memory_row_to_dict(r) for r in cur.fetchall()]


def get_all_memory_names():
    return [r[0] for r in _get_conn().execute("SELECT object_name FROM environmental_memory")]


# ---- Entities (E1) ----------------------------------------------------
def save_entity(entity_id, name, category, color, signature, last_seen):
    conn = _get_conn()
    conn.execute("""
        INSERT INTO entities (entity_id, name, category, color, signature, last_seen)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(entity_id) DO UPDATE SET
            name = excluded.name, category = excluded.category, color = excluded.color,
            signature = excluded.signature, last_seen = excluded.last_seen
    """, (entity_id, name, category, color,
          signature.astype(np.float32).tobytes() if signature is not None else None, _num(last_seen)))
    conn.commit()


def load_entities():
    out = {}
    for eid, name, cat, color, sig, last_seen in _get_conn().execute(
            "SELECT entity_id, name, category, color, signature, last_seen FROM entities"):
        out[eid] = {
            "name": name, "category": cat, "color": color,
            "sig": np.frombuffer(sig, dtype=np.float32).reshape((16, 20)).copy() if sig else None,
            "last_x": None, "last_y": None, "last_seen": last_seen,
        }
    return out


# ---- Events (E2) -------------------------------------------------------
def log_event(ts, name, event, zone=None, world_angle=None, session_id=None,
              dist_feet=None, detail=None):
    conn = _get_conn()
    conn.execute("""
        INSERT INTO events (ts, name, event, zone, world_angle, session_id, dist_feet, detail)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (float(ts), name.lower(), event, zone, _num(world_angle), session_id, _num(dist_feet), detail))
    conn.commit()


def _event_rows(sql, args):
    keys = ["ts", "name", "type", "zone", "world_angle", "session_id", "dist_feet", "detail"]
    cur = _get_conn().execute(sql, args)
    return [dict(zip(keys, r)) for r in cur.fetchall()]


def get_events_since(ts, names=None):
    sql = ("SELECT ts, name, event, zone, world_angle, session_id, dist_feet, detail "
           "FROM events WHERE ts >= ?")
    args = [ts]
    if names:
        sql += " AND name IN (%s)" % ",".join("?" * len(names))
        args += [n.lower() for n in names]
    return _event_rows(sql + " ORDER BY ts", args)


def get_item_history(name, limit=10):
    """Newest-first events for one item (matching by substring)."""
    return _event_rows(
        "SELECT ts, name, event, zone, world_angle, session_id, dist_feet, detail FROM events "
        "WHERE name LIKE ? ORDER BY ts DESC LIMIT ?", (f"%{name.lower()}%", limit))


def get_last_event_time(event_types, session_id=None):
    sql = "SELECT MAX(ts) FROM events WHERE event IN (%s)" % ",".join("?" * len(event_types))
    args = list(event_types)
    if session_id:
        sql += " AND session_id = ?"
        args.append(session_id)
    row = _get_conn().execute(sql, args).fetchone()
    return row[0] if row and row[0] else None


# ---- Settings (O2) -----------------------------------------------------
def get_setting(key, default=None):
    row = _get_conn().execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
    return row[0] if row else default


def set_setting(key, value):
    conn = _get_conn()
    conn.execute("INSERT INTO settings (key, value) VALUES (?, ?) "
                 "ON CONFLICT(key) DO UPDATE SET value = excluded.value", (key, str(value)))
    conn.commit()


# ---- Privacy: forget someone or something ---------------------------------
def forget(name):
    """Delete a person or item everywhere: profile, memory, entity and history."""
    conn = _get_conn()
    n = name.lower()
    removed = 0
    removed += conn.execute("DELETE FROM people_profiles WHERE LOWER(name) = ?", (n,)).rowcount
    removed += conn.execute("DELETE FROM custom_object_profiles WHERE LOWER(name) = ?", (n,)).rowcount
    removed += conn.execute("DELETE FROM environmental_memory WHERE object_name = ?", (n,)).rowcount
    removed += conn.execute("DELETE FROM entities WHERE LOWER(name) = ?", (n,)).rowcount
    removed += conn.execute("DELETE FROM events WHERE name = ?", (n,)).rowcount
    conn.commit()
    return removed
