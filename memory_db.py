import sqlite3
import time

DB_PATH = "sense_memory.db"

def init_database():
    """Creates the memory table if it does not exist yet."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Create our environmental memory table
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
    conn.commit()
    conn.close()
    print("💾 SENSE Database initialized (sense_memory.db ready).")

def save_object_observation(name, zone, x, y, status="VISIBLE"):
    """Saves or updates an object's location and timestamp in the database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    current_time = time.time()

    # UPSERT: If item exists, update its location and time. If not, insert it!
    cursor.execute("""
        INSERT INTO environmental_memory (object_name, last_zone, last_x, last_y, last_seen, status)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(object_name) DO UPDATE SET
            last_zone = excluded.last_zone,
            last_x = excluded.last_x,
            last_y = excluded.last_y,
            last_seen = excluded.last_seen,
            status = excluded.status
    """, (name, zone, x, y, current_time, status))

    conn.commit()
    conn.close()

def get_object_memory(name):
    """Retrieves the last known position and timestamp of a specific object."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT last_zone, last_seen, status 
        FROM environmental_memory 
        WHERE object_name = ?
    """, (name,))
    
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
    """Loads all items currently saved in the database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("SELECT object_name, last_zone, last_seen, status FROM environmental_memory")
    rows = cursor.fetchall()
    conn.close()

    return rows

# -------------------------------------------------------------
# QUICK SELF-TEST: Let's test the database right now!
# -------------------------------------------------------------
if __name__ == "__main__":
    print("Testing SENSE Database...")
    init_database()

    # Simulate saving a phone
    save_object_observation("phone", "on your left", 120, 240)
    print("Saved test record: phone on your left.")

    # Retrieve it back
    memory = get_object_memory("phone")
    print(f"Retrieved from database: {memory}")

    print("🎉 Database test passed successfully!")