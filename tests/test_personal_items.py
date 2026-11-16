"""
"My cup" is ONE physical cup (personal/ package) - unit and scenario tests.

The seven scenarios krish asked for (TEST 1-6 here; TEST 7 = the whole suite
via tools/run_checks.py), plus the edge cases behind them: ByteTrack id
reuse and id switches, identical-looking objects, questions about "my <kind>".

Run:  python -m unittest tests.test_personal_items -v
"""

import os
import sqlite3
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from personal_sim import Room, chroma_signature, cv2, render  # noqa: E402

import personal  # noqa: E402
from personal import phrasing  # noqa: E402
from personal.manager import _Disabled  # noqa: E402

CATS = {"cup", "bottle", "phone", "laptop", "chair", "remote", "book", "mouse", "bag", "table"}


def make(conn=None, session="s1", known=None):
    return personal.create(session, known, CATS, conn=conn or sqlite3.connect(":memory:"))


@unittest.skipIf(cv2 is None, "needs OpenCV")
class TestScenarios(unittest.TestCase):

    def setUp(self):
        self.p = make()
        self.room = Room(self.p, seed=1)

    def answer(self, question):
        """What the WHERE path would resolve and add (see sense_master)."""
        kind, value = self.p.query(question, self.room.now)
        if kind == "say":
            return kind, value
        return value, self.p.note(value, self.room.now, self.room.heading)

    # TEST 1 ------------------------------------------------------------
    def test_1_cup_b_is_never_my_cup(self):
        r = self.room
        r.add("cupA", "cup", "white", 0)
        r.add("cupB", "cup", "white", 45)          # an IDENTICAL white cup, 45 degrees away
        r.run(5)
        r.teach("my cup", "cupA")
        self.assertEqual(r.run(10)["cupA"], "my cup")
        r.turn_to(45)
        seen = r.run(40)
        self.assertIsNone(seen["cupB"], "Cup B was called my cup")
        self.assertEqual(r.wrong_labels("my cup", "cupA"), [])
        target, note = self.answer("where is my cup?")
        self.assertEqual(target, "my cup")
        self.assertIn("can't see it", note)
        self.assertNotIn("I see your cup", note)

    def test_1b_old_threshold_would_have_failed(self):
        """The bug, measured: the two white cups correlate far above the old 0.55."""
        import numpy as np
        rng = np.random.default_rng(0)
        a = chroma_signature(render("white", rng))
        b = chroma_signature(render("white", rng, light=0.92))
        self.assertGreater(cv2.compareHist(a, b, cv2.HISTCMP_CORREL), 0.55)

    # TEST 2 ------------------------------------------------------------
    def test_2_camera_rotation_keeps_the_original_bottle(self):
        r = self.room
        r.add("bottleA", "bottle", "blue", 0)
        r.add("bottleB", "bottle", "blue", 45)
        r.run(5)
        r.teach("my bottle", "bottleA")
        r.run(5)
        r.turn_to(45)
        self.assertIsNone(r.run(30)["bottleB"])
        r.turn_to(0)                                 # back: bottle A gets a NEW track id
        self.assertEqual(r.run(20)["bottleA"], "my bottle", "original bottle not re-identified")
        self.assertEqual(r.wrong_labels("my bottle", "bottleA"), [])
        target, note = self.answer("where is my bottle")
        self.assertEqual((target, note), ("my bottle", ""))      # visible: the live answer is used

    # TEST 3 ------------------------------------------------------------
    def test_3_two_personal_items_never_reassigned(self):
        r = self.room
        r.add("phoneA", "phone", "black", -10)
        r.add("bottleA", "bottle", "green", 10)
        r.run(5)
        r.teach("my phone", "phoneA")
        r.teach("my bottle", "bottleA")
        r.run(5)
        r.add("phoneB", "phone", "black", 50)
        r.add("bottleB", "bottle", "green", 70)
        r.turn_to(60)
        r.run(30)
        r.turn_to(0)
        r.run(20)
        r.turn_to(60)
        r.run(20)
        self.assertEqual(r.wrong_labels("my phone", "phoneA"), [])
        self.assertEqual(r.wrong_labels("my bottle", "bottleA"), [])
        self.assertGreater(r.label_count("my phone", "phoneA"), 0)
        self.assertGreater(r.label_count("my bottle", "bottleA"), 0)

    def test_3b_same_kind_in_the_same_picture(self):
        """Phone B right next to phone A: B is noted as a look-alike, never mine."""
        r = self.room
        r.add("phoneA", "phone", "black", -8)
        r.add("phoneB", "phone", "black", 8)
        r.run(5)
        r.teach("my phone", "phoneA")
        r.run(30)
        self.assertEqual(r.wrong_labels("my phone", "phoneA"), [])
        self.assertGreaterEqual(len(self.p.matcher.items["my phone"].lookalikes), 1)
        r.turn_to(90)                                 # both tracks lost
        r.turn_to(0)
        r.run(30)                                     # identical phones, both near the spot
        self.assertEqual(r.wrong_labels("my phone", "phoneA"), [])

    # TEST 4 ------------------------------------------------------------
    def test_4_temporary_disappearance(self):
        r = self.room
        r.add("cupA", "cup", "red", 0)
        r.run(5)
        r.teach("my cup", "cupA")
        r.run(5)
        r.turn_to(-90)                                # nothing there
        r.run(20)
        target, note = self.answer("where is my cup")
        self.assertEqual(target, "my cup")
        self.assertIn("can't see it", note)
        r.add("cupB", "cup", "red", 180)              # another red cup, behind
        r.turn_to(180)
        r.run(30)
        self.assertEqual(r.wrong_labels("my cup", "cupA"), [])
        target, note = self.answer("where is my cup")
        self.assertIn("looks like yours, but I can't be sure", note)
        item = self.p.matcher.items["my cup"]
        self.assertLess(abs(item.world_angle), 5)     # last known place is still cup A's

    # TEST 5 ------------------------------------------------------------
    def test_5_restart_keeps_identity_and_place(self):
        path = os.path.join(tempfile.mkdtemp(), "mem.db")
        conn = sqlite3.connect(path)
        p1 = make(conn, "run1")
        r = Room(p1, seed=2)
        r.add("bottleA", "bottle", "blue", 0)
        r.add("bottleB", "bottle", "yellow", 20)     # a different-looking bottle beside it
        r.run(5)
        r.teach("my bottle", "bottleA")
        r.run(10)
        p1.flush()
        before = p1.matcher.items["my bottle"]
        conn.close()

        conn2 = sqlite3.connect(path)                # "restart SENSE"
        p2 = make(conn2, "run2", known={"my bottle": {"category": "bottle", "sig": before.taught_sig}})
        item = p2.matcher.items["my bottle"]
        self.assertEqual(item.instance_id, before.instance_id)
        self.assertAlmostEqual(item.world_angle, before.world_angle, places=3)
        self.assertEqual(item.session_id, "run1")
        self.assertEqual(len(item.lookalikes), len(before.lookalikes))
        self.assertEqual(p2.query("where is my bottle")[1], "my bottle")

        r2 = Room(p2, seed=3)
        r2.heading = 20
        r2.add("bottleB", "bottle", "yellow", 20)    # first only the OTHER bottle in view
        r2.run(30)
        self.assertEqual(r2.wrong_labels("my bottle", "bottleA"), [])
        r2.add("bottleA", "bottle", "blue", 0)       # still where it was
        r2.turn_to(0)
        # after a restart SENSE never confirms by itself (the heading may have
        # restarted facing another way): it says "looks like yours"
        self.assertIsNone(r2.run(20)["bottleA"])
        note = p2.note("my bottle", r2.now, r2.heading)
        self.assertIn("looks like yours", note)
        self.assertIn("If it is, say: this is my bottle.", note)
        r2.teach("my bottle", "bottleA")              # "this is my bottle" - confirmed again
        self.assertEqual(r2.run(5)["bottleA"], "my bottle")
        self.assertEqual(p2.matcher.items["my bottle"].instance_id, before.instance_id)
        self.assertEqual(r2.wrong_labels("my bottle", "bottleA"), [])

    def test_5e_optional_confirm_after_restart(self):
        """confirm_after_restart=True: a colourful item at its old place is
        recognised again without re-teaching."""
        conn = sqlite3.connect(":memory:")
        p1 = make(conn, "run1")
        r = Room(p1, seed=12)
        r.add("bottleA", "bottle", "blue", 0)
        r.run(5)
        r.teach("my bottle", "bottleA")
        r.run(5)
        p1.flush()
        p2 = make(conn, "run2")
        p2.cfg.confirm_after_restart = True
        r2 = Room(p2, seed=13)
        r2.add("bottleA", "bottle", "blue", 0)
        self.assertEqual(r2.run(20)["bottleA"], "my bottle")

    def test_5d_restart_white_item_needs_reconfirming(self):
        """White / grey / black items: after a restart, even at the same place,
        SENSE says "looks like yours" (an identical one could stand there)."""
        conn = sqlite3.connect(":memory:")
        p1 = make(conn, "run1")
        r = Room(p1, seed=10)
        r.add("cupA", "cup", "white", 0)
        r.run(5)
        r.teach("my cup", "cupA")
        r.run(5)
        p1.flush()
        p2 = make(conn, "run2")
        r2 = Room(p2, seed=11)
        r2.add("cupB", "cup", "white", 0)            # an identical cup where A was
        r2.run(30)
        self.assertEqual(r2.label_count("my cup", "cupB"), 0)
        self.assertIn("looks like yours", p2.note("my cup", r2.now, r2.heading))

    def test_5c_restart_and_moved_elsewhere_is_only_a_lookalike(self):
        conn = sqlite3.connect(":memory:")
        p1 = make(conn, "run1")
        r = Room(p1, seed=8)
        r.add("bottleA", "bottle", "orange", 0)
        r.run(5)
        r.teach("my bottle", "bottleA")
        r.run(5)
        p1.flush()
        p2 = make(conn, "run2")
        r2 = Room(p2, seed=9)
        r2.heading = 90
        r2.add("bottleX", "bottle", "orange", 90)    # same look, a different direction
        r2.run(30)
        self.assertEqual(r2.label_count("my bottle", "bottleX"), 0)
        note = p2.note("my bottle", r2.now, r2.heading)
        self.assertIn("looks like yours, but I can't be sure", note)

    def test_5b_restart_with_identical_lookalike_stays_unsure(self):
        conn = sqlite3.connect(":memory:")
        p1 = make(conn, "run1")
        r = Room(p1, seed=4)
        r.add("cupA", "cup", "white", -10)
        r.add("cupB", "cup", "white", 10)            # identical, seen together
        r.run(5)
        r.teach("my cup", "cupA")
        r.run(10)
        p1.flush()
        p2 = make(conn, "run2")
        r2 = Room(p2, seed=5)
        r2.add("cupB", "cup", "white", 0)
        r2.run(30)
        self.assertEqual(r2.wrong_labels("my cup", "cupA"), [])
        target, note = (lambda q: (q[1], p2.note(q[1], r2.now, r2.heading)))(p2.query("where is my cup"))
        self.assertIn("looks like yours", note)

    # TEST 6 ------------------------------------------------------------
    def test_6_ambiguous_two_candidates_neither_named(self):
        conn = sqlite3.connect(":memory:")
        p1 = make(conn, "run1")
        r = Room(p1, seed=6)
        r.add("cupA", "cup", "pink", 0)
        r.run(5)
        r.teach("my cup", "cupA")
        r.run(5)
        p1.flush()
        p2 = make(conn, "run2")                      # restart: no position to help
        r2 = Room(p2, seed=7)
        r2.add("cupA", "cup", "pink", -10)
        r2.add("cupC", "cup", "pink", 10)            # an identical cup, never seen before
        r2.run(30)
        self.assertEqual(r2.wrong_labels("my cup", "cupA"), [])
        self.assertEqual(r2.label_count("my cup", "cupA"), 0, "guessed between two identical cups")
        note = p2.note("my cup", r2.now, r2.heading)
        self.assertIn("two cups that look like yours", note)

    def test_6b_two_personal_bottles_asks_which(self):
        r = self.room
        r.add("b1", "bottle", "blue", -10)
        r.add("b2", "bottle", "red", 10)
        r.run(5)
        r.teach("my water bottle", "b1")
        r.teach("my medicine bottle", "b2")
        kind, text = self.p.query("where is my bottle")
        self.assertEqual(kind, "say")
        self.assertIn("Do you mean your medicine bottle or your water bottle?", text)
        self.assertEqual(self.p.query("where is my water bottle"), ("target", "my water bottle"))

    # Tracker edge cases -------------------------------------------------
    def test_bytetrack_id_switch_while_turning(self):
        r = self.room
        r.add("cupA", "cup", "white", -15)
        r.add("cupB", "cup", "white", 15)
        r.run(5)
        r.teach("my cup", "cupA")
        r.run(5)
        r.heading = 3
        r.step(stable=False, track_switch=("cupA", "cupB"))   # ids swapped during motion
        r.now += 0.9
        r.run(20)
        self.assertEqual(r.wrong_labels("my cup", "cupA"), [])

    def test_track_id_reused_after_gap(self):
        p = self.p.matcher
        import numpy as np
        rng = np.random.default_rng(3)
        sig_a = chroma_signature(render("red", rng))
        sig_b = chroma_signature(render("blue", rng))
        p.teach("my cup", "cup", sig_a, 7, 0.0, 100.0)
        self.assertEqual(p.label(7, "cup", sig_a, 0.0, 100.1, True), "my cup")
        p.end_frame(100.1, True)
        # the same id 5 s later on a different cup far away: not trusted
        self.assertIsNone(p.label(7, "cup", sig_b, 60.0, 105.0, True))

    def test_doubt_removes_label_when_look_changes(self):
        p = self.p.matcher
        import numpy as np
        rng = np.random.default_rng(4)
        sig_a = chroma_signature(render("red", rng))
        sig_b = chroma_signature(render("green", rng))
        p.teach("my cup", "cup", sig_a, 9, 0.0, 100.0)
        t = 100.0
        labels = []
        for _ in range(10):
            t += 0.1
            labels.append(p.label(9, "cup", sig_b, 0.0, t, True))
            p.end_frame(t, True)
        self.assertIsNone(labels[-1])

    def test_category_switch_drops_label(self):
        p = self.p.matcher
        import numpy as np
        sig = chroma_signature(render("red", np.random.default_rng(5)))
        p.teach("my cup", "cup", sig, 11, 0.0, 100.0)
        p.label(11, "bottle", sig, 0.0, 100.1, True)
        p.end_frame(100.1, True)
        self.assertIsNone(p.label(11, "cup", sig, 0.0, 100.2, True))

    def test_reteach_moves_the_name(self):
        r = self.room
        r.add("cupA", "cup", "red", -10)
        r.add("cupB", "cup", "blue", 10)
        r.run(5)
        r.teach("my cup", "cupA")
        r.run(5)
        r.teach("my cup", "cupB")                      # "no, THIS is my cup"
        seen = r.run(10)
        self.assertEqual(seen["cupB"], "my cup")
        self.assertIsNone(seen["cupA"])

    def test_rename_and_forget(self):
        r = self.room
        r.add("bottleA", "bottle", "blue", 0)
        r.run(5)
        r.teach("my bottle", "bottleA")
        self.p.rename("my bottle", "my medicine")
        self.assertEqual(r.run(3)["bottleA"], "my medicine")
        self.p.forget("my medicine")
        self.assertIsNone(r.run(3)["bottleA"])
        self.assertEqual(self.p.store.load(), [])


class TestQuestions(unittest.TestCase):
    def setUp(self):
        import numpy as np
        self.p = make()
        sig = np.zeros((16, 20), np.float32)
        sig[8, 10] = 1
        for name, cat in [("my cup", "cup"), ("my laptop", "laptop"), ("my bag", "bag")]:
            self.p.matcher.teach(name, cat, sig, hash(name) % 1000, None, 1.0)

    def test_questions_find_the_personal_item(self):
        for q, name in [("Where is my cup?", "my cup"), ("where's my mug", "my cup"),
                        ("Have you seen my bag?", "my bag"), ("have you seen my backpack", "my bag"),
                        ("When did you last see my laptop?", "my laptop"),
                        ("where did i leave my cup", "my cup")]:
            self.assertEqual(self.p.query(q), ("target", name), q)

    def test_untaught_kind_is_not_answered_with_any_object(self):
        kind, text = self.p.query("where is my phone")
        self.assertEqual(kind, "say")
        self.assertIn("I don't know which phone is yours yet", text)
        kind, text = self.p.query("where is my cell phone")
        self.assertIn("which phone", text)

    def test_other_questions_untouched(self):
        for q in ["where is the cup", "where is Sneha", "what is on my left", "where is the door",
                  "where is my friend", "what do you see"]:
            self.assertIsNone(self.p.query(q), q)

    def test_spoken_and_plural(self):
        self.assertEqual(phrasing.spoken("my cup"), "your cup")
        self.assertEqual(phrasing.plural("mouse"), "mice")


class TestBagSupport(unittest.TestCase):
    def test_backpack_and_handbag_are_bags(self):
        with open(os.path.join(ROOT, "sense_master.py"), encoding="utf-8") as f:
            src = f.read()
        self.assertIn('"backpack": "bag"', src)
        self.assertIn('"handbag": "bag"', src)


class TestSafety(unittest.TestCase):
    def test_disabled_stand_in(self):
        d = _Disabled()
        self.assertFalse(d.enabled)
        self.assertIsNone(d.label(1, "cup", None, 0, 0, True))
        self.assertEqual(d.note("my cup", 0, 0), "")
        self.assertIsNone(d.query("where is my cup"))

    def test_create_never_raises(self):
        class Broken:
            def cursor(self):
                raise RuntimeError("no db")
        self.assertFalse(personal.create("s", {}, CATS, conn=Broken()).enabled)

    def test_errors_are_contained(self):
        p = make()
        p.matcher.items = None                          # break it on purpose
        self.assertIsNone(p.label(1, "cup", None, 0.0, 1.0, True))
        self.assertIsNone(p.end_frame(1.0, True))

    def test_migrates_items_taught_before(self):
        import numpy as np
        sig = np.ones((16, 20), np.float32)
        p = make(known={"my bottle": {"category": "bottle", "sig": sig}})
        self.assertIn("my bottle", p.matcher.items)
        self.assertEqual(len(p.store.load()), 1)

    def test_forgotten_items_are_not_resurrected(self):
        import numpy as np
        conn = sqlite3.connect(":memory:")
        p = make(conn, known={"my bottle": {"category": "bottle", "sig": np.ones((16, 20), np.float32)}})
        self.assertIn("my bottle", p.matcher.items)
        p2 = make(conn, known={})      # custom_object_profiles no longer has it
        self.assertNotIn("my bottle", p2.matcher.items)


if __name__ == "__main__":
    unittest.main()
