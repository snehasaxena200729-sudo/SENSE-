import sqlite3
import time
import os
import numpy as np

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sense_memory.db")

def init_database():
    """Initializes tables and safely cleans any old ghost names."""
    conn = sqlite3.connect(DB_PATH)
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
    # -------------------------------------------------------------
    # SAFELY PURGE GHOSTS (Zero constraint conflicts!)
    # -------------------------------------------------------------
    cursor.execute("DELETE FROM environmental_memory WHERE object_name IN ('neha', 'snail', 'now')")
    cursor.execute("DELETE FROM people_profiles WHERE name IN ('Neha', 'Snail', 'Now')")

    conn.commit()
    conn.close()
    print("💾 SENSE Database initialized cleanly (Ghosts purged).")

def save_object_observation(name, zone, x, y, status="VISIBLE"):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    current_time = time.time()

    cursor.execute("""
        INSERT INTO environmental_memory (object_name, last_zone, last_x, last_y, last_seen, status)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(object_name) DO UPDATE SET
            last_zone = excluded.last_zone,
            last_x = excluded.last_x,
            last_y = excluded.last_y,
            last_seen = excluded.last_seen,
            status = excluded.status
    """, (name.lower(), zone, x, y, current_time, status))

    conn.commit()
    conn.close()

def get_object_memory(name):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT last_zone, last_seen, status 
        FROM environmental_memory 
        WHERE LOWER(object_name) = ?
    """, (name.lower(),))
    
    row = cursor.fetchone()
    conn.close()

    if row:
        return {
            "last_zone": row[0],
            "last_seen": row[1],
            "status": row[2]
        }
    return None

def get_all_remembered_items():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT object_name, last_zone, last_seen, status FROM environmental_memory")
    rows = cursor.fetchall()
    conn.close()
    return rows

def save_person_visual_profile(name, face_hist, cloth_hist):
    """Permanently saves a person's visual fingerprint into SQLite!"""
    conn = sqlite3.connect(DB_PATH)
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
    conn.close()
    print(f"💾 Permanently saved visual profile for {name} in SQLite database!")

def load_all_people_visual_profiles():
    """Loads all known people and their visual signatures from SQLite on startup!"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT name, face_hist, cloth_hist FROM people_profiles")
    rows = cursor.fetchall()
    conn.close()

    people_dict = {}
    for name, f_bytes, c_bytes in rows:
        face_hist = np.frombuffer(f_bytes, dtype=np.float32).reshape((16, 20))
        cloth_hist = np.frombuffer(c_bytes, dtype=np.float32).reshape((16, 20))
        people_dict[name] = (face_hist, cloth_hist)

    return people_dict
def save_custom_object_profile(name, base_category, signature):
    """Saves a personal item's visual fingerprint into SQLite."""
    conn = sqlite3.connect(DB_PATH)
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
    conn.close()
    print(f"💾 Permanently memorized custom object: '{name}' in SQLite!")

def load_all_custom_object_profiles():
    """Loads all personal items on startup."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT name, base_category, signature FROM custom_object_profiles")
    rows = cursor.fetchall()
    conn.close()

    objects_dict = {}
    for name, cat, s_bytes in rows:
        sig = np.frombuffer(s_bytes, dtype=np.float32).reshape((16, 20))
        objects_dict[name] = {"category": cat, "sig": sig}
    return objects_dict