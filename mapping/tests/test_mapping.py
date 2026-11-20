"""Phase 0 home-map tests: schema safety, the four stores, the four commands,
and that existing voice commands are never taken.

    python -m unittest discover -s mapping/tests -t .
"""

import hashlib
import os
import shutil
import sqlite3
import tempfile
import unittest

import answers
import mapping
from mapping import phrases as P
from mapping.schema import SchemaConflict, migrate

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXISTING_TABLES = ("environmental_memory", "people_profiles", "custom_object_profiles",
                   "entities", "events", "face_profiles", "settings")


def table_digest(conn, skip=("rooms", "room_photos", "door_links", "object_rooms", "sqlite_sequence")):
    """Hash of every row and the schema of every table that is NOT ours."""
    h = hashlib.sha256()
    for (name, tbl, sql) in conn.execute("SELECT name, tbl_name, sql FROM sqlite_master "
                                         "WHERE type IN ('table','index') ORDER BY name"):
        if tbl in skip:
            continue
        h.update(f"{name}|{sql}".encode())
        if sql and sql.upper().startswith("CREATE TABLE"):
            for row in conn.execute(f"SELECT * FROM {name} ORDER BY rowid"):
                h.update(repr(row).encode())
    return h.hexdigest()


class Clock:
    def __init__(self, t=1_000_000.0):
        self.t = t

    def __call__(self):
        return self.t


class SchemaTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.db = os.path.join(self.tmp, "copy.db")
        shutil.copy(os.path.join(ROOT, "sense_memory.db"), self.db)   # the real repo memory, copied

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_creates_four_tables(self):
        conn = sqlite3.connect(self.db)
        migrate(conn)
        names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        self.assertTrue({"rooms", "room_photos", "door_links", "object_rooms"} <= names)

    def test_existing_tables_and_records_unchanged(self):
        conn = sqlite3.connect(self.db)
        before = table_digest(conn)
        counts = {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                  for t in ("environmental_memory", "people_profiles", "custom_object_profiles")}
        home = mapping.create("s", conn)
        home.handle("this room is the kitchen")
        home.rooms.register("hall")
        self.assertEqual(table_digest(conn), before)
        for t, n in counts.items():
            self.assertEqual(conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0], n)

    def test_repeatable(self):
        conn = sqlite3.connect(self.db)
        for _ in range(5):
            migrate(conn)
        home = mapping.create("s", conn)
        home.rooms.register("kitchen")
        migrate(conn)
        mapping.create("s2", conn)
        self.assertEqual(home.rooms.names(), ["kitchen"])

    def test_works_with_full_memory_db_schema(self):
        """All tables memory_db creates, plus ours, side by side."""
        import memory_db
        old = memory_db.DB_PATH
        try:
            memory_db.close()
            memory_db.DB_PATH = os.path.join(self.tmp, "full.db")
            memory_db.init_database()
            conn = memory_db._get_conn()
            before = table_digest(conn)
            home = mapping.create("s", conn)
            self.assertTrue(home.enabled)
            self.assertEqual(table_digest(conn), before)
            memory_db.save_object_observation("my bottle", "on your left", 10, 20, status="VISIBLE")
            self.assertIsNotNone(memory_db.find_latest_memory("my bottle"))
        finally:
            memory_db.close()
            memory_db.DB_PATH = old

    def test_conflicting_table_is_left_alone(self):
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE TABLE rooms (x TEXT)")
        conn.execute("INSERT INTO rooms VALUES ('keep me')")
        with self.assertRaises(SchemaConflict):
            migrate(conn)
        self.assertEqual(conn.execute("SELECT x FROM rooms").fetchall(), [("keep me",)])
        self.assertFalse(conn.execute("SELECT name FROM sqlite_master WHERE name='room_photos'").fetchone())
        home = mapping.create("s", conn)            # switched off, never raises
        self.assertFalse(home.enabled)
        self.assertIsNone(home.handle("which rooms do you know"))

    def test_foreign_key_setting_untouched(self):
        conn = sqlite3.connect(":memory:")
        mapping.create("s", conn)
        self.assertEqual(conn.execute("PRAGMA foreign_keys").fetchone()[0], 0)


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.conn = sqlite3.connect(":memory:")
        self.home = mapping.HomeMap(self.conn, "run1", clock=self.clock)

    def test_room_registration(self):
        room, new = self.home.rooms.register("Kitchen", "run1")
        self.assertTrue(new)
        self.assertEqual(room["name"], "kitchen")
        _, new2 = self.home.rooms.register("kitchen", "run1")
        self.assertFalse(new2)
        self.assertEqual(self.home.rooms.names(), ["kitchen"])

    def test_room_photos(self):
        rid = self.home.rooms.register("kitchen")[0]["id"]
        pid = self.home.photos.add(rid, heading_deg=40.0, fingerprint=b"\x01\x02",
                                   landmarks=["Oven", "refrigerator"], session_id="run1")
        self.assertGreater(pid, 0)
        rows = self.home.photos.for_room(rid)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["fingerprint"], b"\x01\x02")
        self.assertEqual(rows[0]["landmarks"], ["oven", "refrigerator"])
        self.assertEqual(self.home.photos.count(rid), 1)

    def test_door_links(self):
        k = self.home.rooms.register("kitchen")[0]["id"]
        h = self.home.rooms.register("hall")[0]["id"]
        link = self.home.doors.link(h, k, heading_from_x=90.0, heading_from_y=-90.0)
        self.assertEqual((link["room_a"], link["room_b"]), (k, h))     # stored in id order
        self.assertEqual(link["door_heading_a"], -90.0)                  # headings swapped with it
        self.home.doors.link(k, h)
        self.assertEqual(self.home.doors.get(k, h)["walks"], 2)
        self.assertEqual(self.home.doors.neighbours(k), [(h, -90.0, 2)])
        with self.assertRaises(ValueError):
            self.home.doors.link(k, k)

    def test_object_rooms(self):
        k = self.home.rooms.register("kitchen")[0]["id"]
        b = self.home.rooms.register("bedroom")[0]["id"]
        self.home.objects.set("My Bottle", k)
        self.assertEqual(self.home.objects.get("my bottle")["room_name"], "kitchen")
        self.home.objects.set("my bottle", b)
        self.assertEqual(self.home.objects.get("my bottle")["room_name"], "bedroom")
        self.home.objects.rename("my bottle", "my flask")
        self.assertIsNone(self.home.objects.get("my bottle"))
        self.assertEqual(self.home.objects.in_room(b), ["my flask"])
        self.home.objects.forget("my flask")
        self.assertIsNone(self.home.objects.get("my flask"))

    def test_delete_room_removes_children(self):
        k = self.home.rooms.register("kitchen")[0]["id"]
        h = self.home.rooms.register("hall")[0]["id"]
        self.home.photos.add(k)
        self.home.doors.link(k, h)
        self.home.objects.set("cup", k)
        self.assertTrue(self.home.rooms.delete("kitchen"))
        for t in ("room_photos", "door_links", "object_rooms"):
            self.assertEqual(self.conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0], 0)
        self.assertEqual(self.home.rooms.names(), ["hall"])


class CommandTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.home = mapping.HomeMap(sqlite3.connect(":memory:"), "run1", clock=self.clock)

    def say(self, text, names=()):
        return self.home.handle(text, self.clock(), known_names=names)

    def test_empty_state(self):
        self.assertIn("No rooms are registered yet", self.say("Which rooms do you know?"))
        self.assertIn("I don't know which room you are in", self.say("Where am I?"))
        self.assertIn("No rooms are registered yet", self.say("Where am I?"))
        self.assertIn("I don't know the bedroom yet", self.say("Take me to the bedroom."))

    def test_teach_then_ask(self):
        self.assertIn("saved the kitchen", self.say("This room is the kitchen."))
        self.clock.t += 120
        self.assertEqual(self.say("Where am I?"),
                         "The last room you told me was the kitchen, 2 minutes ago. "
                         "I can't recognise rooms by sight yet.")
        self.assertIn("already in the kitchen", self.say("take me to the kitchen"))
        self.say("this room is the bedroom")
        self.assertIn("can't guide you between rooms yet", self.say("take me to the kitchen"))
        self.assertEqual(self.say("which rooms do you know"),
                         "I know 2 rooms: the kitchen and the bedroom.")

    def test_where_am_i_expires_and_is_per_run(self):
        self.say("this room is the kitchen")
        self.clock.t += 31 * 60
        self.assertTrue(self.say("where am i").startswith("I don't know which room"))
        other_run = mapping.HomeMap(self.home.rooms.conn, "run2", clock=self.clock)
        self.clock.t -= 31 * 60
        self.assertTrue(other_run.handle("where am i", self.clock()).startswith("I don't know which room"))

    def test_teach_without_name(self):
        self.assertIn("didn't catch", self.say("this room is"))
        self.assertEqual(self.home.rooms.names(), [])

    def test_errors_never_escape(self):
        self.home.rooms.conn.close()
        self.assertIsNone(self.say("which rooms do you know"))


class ClashTests(unittest.TestCase):
    """Sentences the existing code owns must return None (= untouched)."""
    EXISTING = [
        "this is sneha", "This is Sneha.", "is sneha", "this is my bottle", "this is my cup",
        "this is the kitchen", "this is a bottle", "my name is krish", "call this my bag",
        "take me to my phone", "take me to the door", "take me to sneha", "guide me to the chair",
        "find my bottle", "find the kitchen", "where is my phone", "where is the bed",
        "what's in this room", "what is in the room", "describe the room", "what do you see",
        "how many people are in the room", "what colour is the bottle", "what time is it",
        "forget my bottle", "forget sneha", "rename my cup to my mug", "what changed",
        "who is here", "read this", "stop looking", "be quiet", "speak slower",
        "where was my cup", "what did you see recently", "is there anything on my right",
        "room", "the room", "rooms", "",
    ]

    def test_existing_commands_not_taken(self):
        for s in self.EXISTING:
            self.assertIsNone(P.parse(s, known_rooms=("kitchen",), known_names={"sneha", "my bottle"}), s)

    def test_new_commands_recognised(self):
        cases = {
            "This room is the kitchen.": (P.TEACH, "kitchen"),
            "call this room the study": (P.TEACH, "study"),
            "this room is called the den": (P.TEACH, "den"),
            "Where am I?": (P.WHERE_AM_I, None),
            "which room am I in": (P.WHERE_AM_I, None),
            "Take me to the bedroom.": (P.GO, "bedroom"),
            "how do I get to the bed room": (P.GO, "bedroom"),
            "take me to sneha's room": (P.GO, "sneha's room"),
            "Which rooms do you know?": (P.LIST, None),
            "list the rooms": (P.LIST, None),
        }
        for s, (kind, room) in cases.items():
            c = P.parse(s)
            self.assertIsNotNone(c, s)
            self.assertEqual((c.kind, c.room), (kind, room), s)

    def test_known_person_or_object_wins(self):
        self.assertIsNone(P.parse("take me to the study", known_names={"study"}))
        self.assertIsNone(P.parse("take me to sneha", known_rooms=("sneha",), known_names={"sneha"}))

    def test_existing_intents_for_these_sentences_today(self):
        """Documents what each new phrase did BEFORE Phase 0 (it now goes to mapping instead)."""
        self.assertEqual(answers.classify_intent("this room is the kitchen"), "OVERVIEW")
        self.assertEqual(answers.classify_intent("where am i"), "WHERE")
        self.assertEqual(answers.classify_intent("take me to the bedroom"), "FIND")
        self.assertEqual(answers.classify_intent("which rooms do you know"), "OVERVIEW")


if __name__ == "__main__":
    unittest.main()
