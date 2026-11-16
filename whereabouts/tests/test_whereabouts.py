"""Unit, voting, storage and persistence tests for whereabouts/.

    python -m unittest discover -s whereabouts/tests -t .
"""
import os
import sqlite3
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from whereabouts import WhereaboutsConfig, WhereaboutsTracker, create          # noqa: E402
from whereabouts.furniture import surface_under                                # noqa: E402
from whereabouts.landmarks import nearest_landmark                             # noqa: E402
from whereabouts.voting import UNDECIDED, AnchorVoter                          # noqa: E402
from whereabouts import phrasing                                               # noqa: E402

W, H = 1280, 720
CFG = WhereaboutsConfig()
TABLE = (200, 300, 1000, 650)          # top 60% of the box: y 300..510 is table top


def phone_at(cx, bottom, w=60, h=40):
    return (cx - w / 2, bottom - h, cx + w / 2, bottom)


class FurnitureTests(unittest.TestCase):
    def test_on_table(self):
        self.assertEqual(surface_under(phone_at(500, 420), [("table", TABLE)], H, CFG), "table")

    def test_on_desk(self):
        self.assertEqual(surface_under(phone_at(500, 420), [("desk", TABLE)], H, CFG), "desk")

    def test_on_bed(self):
        bed = (100, 250, 1100, 700)    # top 75%: y 250..587
        self.assertEqual(surface_under(phone_at(600, 520), [("bed", bed)], H, CFG), "bed")

    def test_on_couch(self):
        couch = (150, 200, 1050, 600)  # backrest at the top, seat lower: top 80% = y 200..520
        self.assertEqual(surface_under(phone_at(600, 480), [("couch", couch)], H, CFG), "couch")

    def test_in_front_of_table_is_not_on_it(self):
        # bottom in the legs/floor part of the box: in front, not on
        self.assertIsNone(surface_under(phone_at(500, 620), [("table", TABLE)], H, CFG))

    def test_behind_table_is_not_on_it(self):
        # bottom at/above the box's top edge: further back than the table
        self.assertIsNone(surface_under(phone_at(500, 305), [("table", TABLE)], H, CFG))

    def test_off_the_side(self):
        self.assertIsNone(surface_under(phone_at(1050, 420), [("table", TABLE)], H, CFG))
        self.assertIsNone(surface_under(phone_at(205, 420), [("table", TABLE)], H, CFG))   # on the very edge

    def test_too_big_to_be_on_it(self):
        big = (300, 310, 900, 500)     # "phone" box as big as the table top
        self.assertIsNone(surface_under(big, [("table", TABLE)], H, CFG))

    def test_two_kinds_of_furniture_is_ambiguous(self):
        bed = (150, 280, 900, 700)
        self.assertIsNone(surface_under(phone_at(500, 420), [("table", TABLE), ("bed", bed)], H, CFG))

    def test_two_tables_is_fine(self):
        self.assertEqual(surface_under(phone_at(500, 420), [("table", TABLE), ("table", (150, 290, 1050, 660))],
                                       H, CFG), "table")

    def test_not_furniture_ignored(self):
        self.assertIsNone(surface_under(phone_at(500, 420), [("chair", TABLE)], H, CFG))

    def test_table_cut_off_by_picture_bottom(self):
        near = (0, 350, 1280, 720)     # only the top is visible: most of it is surface
        self.assertEqual(surface_under(phone_at(640, 660), [("table", near)], H, CFG), "table")
        # the same point on an uncut box is in its lower part -> not on it
        self.assertIsNone(surface_under(phone_at(640, 660), [("table", (0, 350, 1280, 715))], H, CFG))


LAPTOP = (600, 300, 900, 450)


class LandmarkTests(unittest.TestCase):
    def lm(self, target_box, others, name="phone"):
        return nearest_landmark(name, target_box, others, W, CFG)

    def test_left_of(self):
        self.assertEqual(self.lm(phone_at(500, 445), [("laptop", "laptop", LAPTOP)]),
                         ("laptop", "laptop", "left of"))

    def test_right_of(self):
        self.assertEqual(self.lm(phone_at(1000, 445), [("laptop", "laptop", LAPTOP)]),
                         ("laptop", "laptop", "right of"))

    def test_next_to(self):
        # sideways overlap but not in front: side by side, touching
        self.assertEqual(self.lm((570, 400, 630, 450), [("laptop", "laptop", LAPTOP)])[2], "next to")

    def test_no_landmark_when_far(self):
        self.assertIsNone(self.lm(phone_at(100, 445, w=40), [("laptop", "laptop", (900, 300, 1200, 450))]))

    def test_no_landmark_alone(self):
        self.assertIsNone(self.lm(phone_at(500, 445), []))

    def test_heavy_overlap_is_ambiguous(self):
        # phone box inside the laptop box: on it? in front of it? -> say nothing
        self.assertIsNone(self.lm((700, 380, 760, 430), [("laptop", "laptop", LAPTOP)]))

    def test_overlap_from_the_landmark_side(self):
        # tiny landmark inside a big target (mouse on a laptop) -> ambiguous too
        self.assertIsNone(self.lm(LAPTOP, [("mouse", "mouse", (700, 400, 740, 440))], name="laptop"))

    def test_never_itself(self):
        self.assertIsNone(self.lm(phone_at(500, 445), [("Phone", "phone", phone_at(500, 445))]))

    def test_people_and_pets_are_not_landmarks(self):
        others = [("Sneha", "person", (560, 100, 700, 450)), ("dog", "dog", (380, 380, 460, 450))]
        self.assertIsNone(self.lm(phone_at(500, 445), others))

    def test_different_level_ignored(self):
        shelf_cup = (560, 100, 600, 160)   # bottom far above the phone's
        self.assertIsNone(self.lm(phone_at(500, 445), [("cup", "cup", shelf_cup)]))

    def test_nearest_wins(self):
        others = [("laptop", "laptop", LAPTOP), ("book", "book", (300, 400, 420, 450))]
        self.assertEqual(self.lm(phone_at(500, 445), others)[0], "laptop")    # gap 70 vs 50*1.4
        others = [("laptop", "laptop", LAPTOP), ("book", "book", (380, 400, 460, 450))]
        self.assertEqual(self.lm(phone_at(500, 445), others)[0], "book")      # gap 70 vs 10*1.4

    def test_stable_landmark_preferred(self):
        others = [("laptop", "laptop", LAPTOP), ("cup", "cup", (380, 400, 445, 450))]   # gaps 70 / 25
        self.assertEqual(self.lm(phone_at(500, 445), others)[0], "cup")
        others = [("laptop", "laptop", LAPTOP), ("cup", "cup", (380, 400, 418, 450))]   # gaps 70 / 52*1.4
        self.assertEqual(self.lm(phone_at(500, 445), others)[0], "laptop")


class VotingTests(unittest.TestCase):
    LEFT = ("laptop", "laptop", "left of")

    def setUp(self):
        self.v = AnchorVoter(CFG)
        self.t = 0.0

    def add(self, surface, landmark, angle=0.0):
        self.t += 0.5
        return self.v.add("phone", self.t, surface, landmark, angle)

    def test_three_of_five_decides(self):
        self.add("table", self.LEFT)
        self.add("table", self.LEFT)
        self.assertEqual(self.v.decide("phone")[:2], (UNDECIDED, UNDECIDED))
        self.add("table", self.LEFT)
        self.assertEqual(self.v.decide("phone")[:2], ("table", self.LEFT))

    def test_noise_does_not_win(self):
        for s, lm in [("table", self.LEFT), ("bed", None), ("table", self.LEFT), (None, None),
                      ("table", self.LEFT)]:
            self.add(s, lm)
        self.assertEqual(self.v.decide("phone"), ("table", self.LEFT, 3, 3))

    def test_two_of_five_is_undecided(self):
        for s in ["table", "bed", "table", "couch", None]:
            self.add(s, None)
        self.assertIs(self.v.decide("phone")[0], UNDECIDED)

    def test_confirmed_nothing(self):
        for _ in range(3):
            self.add(None, None)
        self.assertEqual(self.v.decide("phone")[:2], (None, None))

    def test_window_slides(self):
        for _ in range(5):
            self.add("table", None)
        for _ in range(3):
            self.add("bed", None)
        self.assertEqual(self.v.decide("phone")[0], "bed")

    def test_throttled(self):
        self.assertTrue(self.v.add("phone", 1.0, "table", None))
        self.assertFalse(self.v.add("phone", 1.1, "table", None))   # same picture, nearly
        self.assertTrue(self.v.add("phone", 1.5, "table", None))

    def test_move_resets(self):
        for _ in range(3):
            self.add("table", None, angle=0.0)
        self.add("bed", None, angle=40.0)       # the item moved: old votes dropped
        self.assertIs(self.v.decide("phone")[0], UNDECIDED)

    def test_long_gap_resets(self):
        for _ in range(3):
            self.add("table", None)
        self.t += 30
        self.add("bed", None)
        self.assertIs(self.v.decide("phone")[0], UNDECIDED)


def obs(box, category, angle=0.0, zone="in front of you"):
    return {"box": box, "category": category, "world_angle": angle, "zone": zone,
            "x": (box[0] + box[2]) / 2, "y": (box[1] + box[3]) / 2, "dist_feet": None}


SCENE = {"Phone": obs(phone_at(500, 445), "phone"),
         "Laptop": obs(LAPTOP, "laptop"),
         "Brown Table": obs(TABLE, "table")}


class TrackerTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.path = os.path.join(self.dir, "mem.db")
        self.conn = sqlite3.connect(self.path)
        self.tr = WhereaboutsTracker(self.conn, "run1")
        self.t = 1000.0

    def tearDown(self):
        self.conn.close()

    def feed(self, scene, n=5, step=0.5):
        for _ in range(n):
            self.t += step
            self.tr.observe(scene, self.t, W, H, True)

    def rec(self, name="phone", last_seen=None, status="VISIBLE"):
        return {"name": name, "last_seen": self.t if last_seen is None else last_seen, "status": status}

    def test_anchor_saved(self):
        self.feed(SCENE)
        a = self.tr.store.get("phone")
        self.assertEqual((a["surface"], a["landmark"], a["relation"]), ("table", "laptop", "left of"))
        self.assertGreaterEqual(a["surface_votes"], 3)   # written as soon as 3 agree
        self.assertEqual(self.tr.relation_for(self.rec()), "on the table, to the left of the laptop")

    def test_laptop_gets_its_own_anchor_furniture_gets_none(self):
        self.feed(SCENE)
        self.assertEqual(self.tr.store.get("laptop")["relation"], "right of")
        self.assertIsNone(self.tr.store.get("brown table"))

    def test_moving_camera_ignored(self):
        for _ in range(5):
            self.t += 0.5
            self.tr.observe(SCENE, self.t, W, H, False)
        self.assertIsNone(self.tr.store.get("phone"))

    def test_single_noisy_frame_keeps_anchor(self):
        self.feed(SCENE)
        noisy = dict(SCENE)
        noisy["Phone"] = obs(phone_at(500, 445), "phone")
        del noisy["Laptop"]          # laptop missed for one sample
        del noisy["Brown Table"]
        self.feed(noisy, n=1)
        self.feed(SCENE, n=1)
        self.feed(noisy, n=1)        # 2 of the last 5 are wrong: still 3 good
        a = self.tr.store.get("phone")
        self.assertEqual((a["surface"], a["landmark"]), ("table", "laptop"))

    def test_undecided_keeps_stored_when_not_moved(self):
        self.feed(SCENE)
        alt = {"Phone": SCENE["Phone"], "Brown Table": SCENE["Brown Table"],
               "Cup": obs((560, 405, 590, 445), "cup")}     # laptop hidden, a cup appears
        self.feed(alt, n=1)
        self.feed(SCENE, n=1)
        self.feed(alt, n=1)
        self.feed({"Phone": SCENE["Phone"], "Brown Table": SCENE["Brown Table"]}, n=1)
        # last 5 landmark samples: laptop, cup, laptop, cup, none -> undecided,
        # and the phone has not moved, so the stored "left of the laptop" stays
        a = self.tr.store.get("phone")
        self.assertEqual((a["surface"], a["landmark"]), ("table", "laptop"))

    def test_moved_item_replaces_anchor(self):
        self.feed(SCENE)
        bottle = obs((1100, 380, 1150, 600), "bottle", angle=40.0)
        moved = {"Phone": obs(phone_at(1050, 600), "phone", angle=40.0), "Bottle": bottle}
        self.feed(moved)
        a = self.tr.store.get("phone")
        self.assertIsNone(a["surface"])
        self.assertEqual((a["landmark"], a["relation"]), ("bottle", "left of"))
        self.assertEqual(self.tr.relation_for(self.rec()), "to the left of the bottle")

    def test_moved_to_empty_spot_clears(self):
        self.feed(SCENE)
        self.feed({"Phone": obs(phone_at(300, 700), "phone", angle=-50.0)})
        self.assertIsNone(self.tr.relation_for(self.rec()))

    def test_persists_across_restart(self):
        self.feed(SCENE)
        self.conn.close()
        # "stop the app, start it again": new connection, new run id
        self.conn = sqlite3.connect(self.path)
        tr2 = WhereaboutsTracker(self.conn, "run2")
        rec = self.rec(last_seen=self.t)
        self.assertEqual(tr2.relation_for(rec), "on the table, to the left of the laptop")
        sentence, level, _ = tr2.describe(rec, "your phone", 600, "on your left", False, None)
        self.assertIn("on the table, to the left of the laptop, 10 minutes ago", sentence)
        self.assertIn("before I was restarted", sentence)
        self.assertNotIn("on your left", sentence)   # old run's directions are not valid now

    def test_describe_matches_requested_shape(self):
        self.feed(SCENE)
        s, level, _ = self.tr.describe(self.rec(last_seen=self.t), "your phone", 300, "on your left",
                                       True, "turn about 60 degrees to your left")
        self.assertEqual(s, "Your phone was on the table, to the left of the laptop, 5 minutes ago. "
                            "It is probably still there, on your left. "
                            "To face it, turn about 60 degrees to your left.")
        self.assertEqual(level, "likely")

    def test_gone_keeps_honesty(self):
        self.feed(SCENE)
        s, level, _ = self.tr.describe(self.rec(status="GONE"), "your phone", 120, "on your left", True)
        self.assertIn("disappeared while I was watching, so it may have been moved", s)
        self.assertEqual(level, "uncertain")

    def test_stale_anchor(self):
        self.feed(SCENE)
        later = self.rec(last_seen=self.t + 600)     # seen later somewhere without an anchor
        self.assertIsNone(self.tr.describe(later, "your phone", 60, "on your left", True))
        self.assertIsNone(self.tr.relation_for(later))
        note = self.tr.stale_note(later, now=self.t + 660)
        self.assertEqual(note, " Earlier, 11 minutes ago, it was on the table, to the left of the laptop, "
                               "but it may have moved since.")

    def test_no_anchor_falls_back(self):
        self.assertIsNone(self.tr.describe(self.rec(), "your phone", 60, "on your left", True))
        self.assertEqual(self.tr.stale_note(self.rec()), "")
        self.assertIsNone(self.tr.relation_for(self.rec("never seen")))
        self.assertIsNone(self.tr.relation_for(None))

    def test_incomplete_record(self):
        self.feed(SCENE)
        self.conn.execute("UPDATE item_anchors SET seen_at = NULL WHERE item_name = 'phone'")
        self.assertIsNone(self.tr.relation_for(self.rec()))
        self.assertIsNone(self.tr.relation_for({"name": "phone"}))   # no last_seen

    def test_forget_and_rename(self):
        self.feed(SCENE)
        self.tr.rename("laptop", "my laptop")
        self.assertEqual(self.tr.relation_for(self.rec()), "on the table, to the left of your laptop")
        self.tr.forget("my laptop")
        self.assertEqual(self.tr.relation_for(self.rec()), "on the table")
        self.tr.rename("phone", "my phone")
        self.assertIsNone(self.tr.store.get("phone"))
        self.assertEqual(self.tr.relation_for(self.rec("my phone")), "on the table")
        self.tr.forget("my phone")
        self.assertIsNone(self.tr.store.get("my phone"))

    def test_errors_never_escape(self):
        self.conn.close()            # every DB call now fails
        self.tr.observe(SCENE, 1.0, W, H, True)
        self.assertIsNone(self.tr.describe(self.rec(), "your phone", 1, "x", True))
        self.assertEqual(self.tr.stale_note(self.rec()), "")
        self.tr.forget("phone")
        self.conn = sqlite3.connect(":memory:")

    def test_observations_without_box_ignored(self):
        self.tr.observe({"Phone": {"zone": "x", "world_angle": 0.0}}, 1.0, W, H, True)
        self.assertIsNone(self.tr.store.get("phone"))


class MigrationTests(unittest.TestCase):
    def test_existing_data_untouched_and_columns_added(self):
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE TABLE environmental_memory (object_name TEXT PRIMARY KEY, last_seen REAL)")
        conn.execute("INSERT INTO environmental_memory VALUES ('phone', 5.0)")
        # an older, smaller item_anchors table with a row in it
        conn.execute("CREATE TABLE item_anchors (item_name TEXT PRIMARY KEY, surface TEXT)")
        conn.execute("INSERT INTO item_anchors VALUES ('cup', 'table')")
        conn.commit()
        tr = WhereaboutsTracker(conn, "r")
        self.assertEqual(conn.execute("SELECT * FROM environmental_memory").fetchall(), [("phone", 5.0)])
        self.assertEqual(tr.store.get("cup")["surface"], "table")
        self.assertIsNone(tr.store.get("cup")["landmark"])
        WhereaboutsTracker(conn, "r2")         # running the migration twice is harmless
        self.assertEqual(tr.store.get("cup")["surface"], "table")

    def test_create_never_raises(self):
        bad = sqlite3.connect(":memory:")
        bad.close()
        tr = create("r", conn=bad)
        self.assertFalse(tr.enabled)
        self.assertIsNone(tr.describe({"name": "x"}, "x", 1, "", True))
        self.assertEqual(tr.stale_note({}), "")
        tr.observe({}, 0, W, H)


class PhrasingTests(unittest.TestCase):
    def test_names(self):
        self.assertEqual(phrasing.thing_name("my bottle"), "your bottle")
        self.assertEqual(phrasing.thing_name("White Cup"), "the white cup")
        self.assertEqual(phrasing.thing_name("yellow dog 2"), "the yellow dog")

    def test_relation_text(self):
        self.assertEqual(phrasing.relation_text({"surface": "bed"}), "on the bed")
        self.assertEqual(phrasing.relation_text({"landmark": "my bottle", "relation": "next to"}),
                         "next to your bottle")
        self.assertIsNone(phrasing.relation_text({"surface": None, "landmark": None}))
        self.assertIsNone(phrasing.relation_text(None))


if __name__ == "__main__":
    unittest.main()
