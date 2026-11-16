"""Regression tests from the stabilisation audit (Nov 2026).

Each test reproduces a bug that was found by running the code, and checks the
behaviour a user expects (not how the code happens to do it).
"""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import answers  # noqa: E402
import memory_db  # noqa: E402


class DatabaseTestCase(unittest.TestCase):
    def setUp(self):
        memory_db.close()
        self._old_path = memory_db.DB_PATH
        self.tmp = tempfile.mkdtemp()
        memory_db.DB_PATH = os.path.join(self.tmp, "sense_memory.db")
        memory_db.init_database()

    def tearDown(self):
        memory_db.close()
        memory_db.DB_PATH = self._old_path


class TestForgetAfterRename(DatabaseTestCase):
    """Bug: 'rename my bottle to my medicine' stored the events as 'My Medicine',
    but 'forget' deleted only lower-case names, so the history survived a
    'forget my medicine' (the README promises profile, memory AND history)."""

    def _bottle_with_history(self):
        memory_db.save_object_observation("my bottle", "on your left", 10, 20, seen_at=100.0)
        memory_db.log_event(100.0, "my bottle", "APPEARED", "on your left")
        memory_db.log_event(160.0, "my bottle", "MOVED", "on your right", detail="from on your left")

    def test_history_follows_the_new_name(self):
        self._bottle_with_history()
        memory_db.rename("my bottle", "my medicine")
        self.assertEqual(len(memory_db.get_events_since(0, names=["my medicine"])), 2)
        self.assertEqual(memory_db.get_events_since(0, names=["my bottle"]), [])

    def test_forget_after_rename_deletes_history(self):
        self._bottle_with_history()
        memory_db.rename("my bottle", "my medicine")
        memory_db.forget("my medicine")
        self.assertEqual(memory_db.get_events_since(0), [])
        self.assertIsNone(memory_db.find_latest_memory("medicine"))

    def test_forget_also_deletes_rows_renamed_by_older_versions(self):
        # older versions wrote the renamed event name in Title Case
        memory_db._get_conn().execute(
            "INSERT INTO events (ts, name, event) VALUES (1.0, 'My Medicine', 'APPEARED')")
        memory_db.forget("my medicine")
        self.assertEqual(memory_db.get_events_since(0), [])


class TestPersistence(DatabaseTestCase):
    """Memory must survive a restart (close + reopen of the database)."""

    def test_memory_events_and_settings_survive_restart(self):
        memory_db.save_object_observation("white cup", "on your right", 5, 6, world_angle=20.0,
                                          session_id="s1", dist_feet=4.0, seen_at=50.0)
        memory_db.log_event(50.0, "white cup", "APPEARED", "on your right")
        memory_db.set_setting("speech_rate", 160)
        memory_db.close()
        memory_db.init_database()          # what the next run of SENSE does
        rec = memory_db.find_latest_memory("cup")
        self.assertEqual((rec["name"], rec["last_zone"], rec["session_id"]), ("white cup", "on your right", "s1"))
        self.assertEqual(memory_db.get_setting("speech_rate"), "160")
        self.assertEqual([e["type"] for e in memory_db.get_item_history("cup")], ["APPEARED"])

    def test_init_twice_keeps_records(self):
        memory_db.save_object_observation("black phone", "in front of you", 1, 1, seen_at=10.0)
        memory_db.init_database()
        self.assertIsNotNone(memory_db.find_latest_memory("phone"))

    def test_gone_status_is_kept_with_last_position(self):
        memory_db.save_object_observation("white cup", "on your left", 1, 1, seen_at=10.0)
        memory_db.set_object_status("white cup", "GONE")
        rec = memory_db.find_latest_memory("cup")
        self.assertEqual((rec["status"], rec["last_zone"], rec["last_seen"]), ("GONE", "on your left", 10.0))


class TestVoiceCommands(unittest.TestCase):
    """The spoken commands in DEMO_CHECKLIST.md must reach the intended intent."""

    def test_demo_commands(self):
        for text, intent in [
            ("this is sneha", "LEARN"), ("this is my water bottle", "LEARN"),
            ("find my bottle", "FIND"), ("help me find my phone", "FIND"), ("take me to the door", "FIND"),
            ("stop looking", "STOP_FIND"), ("i found it", "STOP_FIND"),
            ("what changed?", "CHANGES"), ("where was my bottle before?", "PREVIOUS"),
            ("when did you last see the bottle?", "WHEN"), ("where is the bottle?", "WHERE"),
            ("what do you see?", "OVERVIEW"), ("who was here today?", "WHO"),
            ("be quiet", "QUIET"), ("normal mode", "NORMAL"), ("talk more", "VERBOSE"),
            ("speak slower", "SLOWER"), ("speak faster", "FASTER"),
        ]:
            self.assertEqual(answers.classify_intent(text), intent, text)

    def test_whisper_prompt_echo_is_not_a_command(self):
        """Bug: on silence Whisper can repeat its hint prompt. The prompt used to end
        with 'rename, forget', and the echo was obeyed: in a headless run SENSE said
        'Done. I have forgotten bottle.' The hint words left in the prompt (names and
        object words) must never classify as a command that changes memory."""
        echo = "sneha, water bottle, medicine bottle, phone, laptop."
        self.assertNotIn(answers.classify_intent(echo), ("FORGET", "RENAME", "LEARN"))


class TestOldRecordsAreNotPresentedAsCurrent(unittest.TestCase):
    """Old database rows must never be described as 'I can see it'."""

    def test_previous_session_is_never_confirmed(self):
        for age in (3, 30, 300):
            sentence, level, _ = answers.describe_memory("the cup", age, "VISIBLE", "on your left",
                                                         same_session=False)
            self.assertNotEqual(level, "confirmed", age)
            self.assertNotIn("I can see", sentence)
            self.assertIn("before I was restarted", sentence)

    def test_disappeared_item_is_not_probably_there(self):
        sentence, level, _ = answers.describe_memory("the cup", 20, "GONE", "on your left", True)
        self.assertNotIn("probably still there", sentence)
        self.assertIn("disappeared", sentence)


if __name__ == "__main__":
    unittest.main()
