"""Scene change summary ("what changed since I left?") - scene_diff.py.

Unit tests for each piece, a SQLite round trip, and a randomised simulation:
rooms of random items, the user leaves, random items are moved / removed /
added, the laptop may be turned between runs, and the camera looks at random
directions afterwards. The report must never claim a change that did not
happen (no false "moved", no "gone" for an item whose spot was never looked at).
"""
import os
import random
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import answers  # noqa: E402
import scene_diff  # noqa: E402
import spatial  # noqa: E402
from scene_diff import ViewCoverage, align_sessions, compare, choose_reference, prepare, summarise  # noqa: E402


def label(n):
    return "the " + n


class TestCoverage(unittest.TestCase):
    def test_looked_long_enough(self):
        c = ViewCoverage()
        for _ in range(60):                       # 4 s at 15 fps, facing 0 degrees
            c.add(0.0, 1 / 15)
        self.assertTrue(c.checked(10.0))
        self.assertFalse(c.checked(90.0))         # never faced the right side
        self.assertFalse(c.checked(-29.0))        # at the frame edge: often missed, not counted

    def test_too_short_and_stalls_ignored(self):
        c = ViewCoverage()
        for _ in range(30):                       # only 2 s
            c.add(0.0, 1 / 15)
        c.add(0.0, 5.0)                           # a 5 s stall is not 5 s of looking
        self.assertFalse(c.checked(0.0))

    def test_wraps_around(self):
        c = ViewCoverage()
        for _ in range(60):
            c.add(178.0, 1 / 15)
        self.assertTrue(c.checked(-175.0))
        c.reset()
        self.assertFalse(c.checked(-175.0))


class TestAlign(unittest.TestCase):
    def test_rotation_found_and_mover_is_outlier(self):
        pairs = [("chair", -40, -10), ("tv", 0, 31), ("table", 25, 54), ("bottle", 10, -50)]
        off, inl = align_sessions(pairs)
        self.assertAlmostEqual(off, 30.0, delta=1.5)
        self.assertEqual(inl, {"chair", "tv", "table"})

    def test_one_item_is_not_enough(self):
        self.assertEqual(align_sessions([("chair", 0, 40)]), (None, set()))

    def test_tie_refuses_to_guess(self):
        pairs = [("a", 0, 0), ("b", 20, 20), ("c", -20, 40), ("d", 50, 110)]
        self.assertEqual(align_sessions(pairs)[0], None)

    def test_across_minus_180(self):
        off, _ = align_sessions([("a", 170, -170), ("b", -175, -155)])
        self.assertAlmostEqual(off, 20.0, delta=0.5)


class TestCompare(unittest.TestCase):
    REF = [{"name": "bottle", "world_angle": 0.0}, {"name": "chair", "world_angle": 30.0},
           {"name": "white cup", "world_angle": -20.0}, {"name": "laptop", "world_angle": 100.0},
           {"name": "book", "world_angle": -12.0}, {"name": "remote", "world_angle": 5.0}]

    def test_all_buckets(self):
        cov = ViewCoverage()
        for _ in range(60):
            cov.add(0.0, 1 / 15)                  # looked straight ahead for 4 s
        now = {"bottle": {"world_angle": 25.0, "status": "VISIBLE"},          # moved 25 deg
               "chair": {"world_angle": 31.0, "status": "VISIBLE"},           # same
               "remote": {"world_angle": 5.0, "status": "GONE"},              # watched it go
               "cup": {"world_angle": -19.0, "status": "VISIBLE"},            # renamed by colour vote
               "phone": {"world_angle": -5.0, "status": "VISIBLE"}}           # new
        d = compare(self.REF, now, coverage=cov, offset=0.0)
        names = {k: sorted(i["name"] for i in v) for k, v in d.items()}
        self.assertEqual(names["moved"], ["bottle"])
        self.assertEqual(names["same"], ["chair", "white cup"])
        self.assertEqual(names["gone"], ["book", "remote"])        # book: its spot was looked at
        self.assertEqual(names["unchecked"], ["laptop"])           # at 100 deg: never looked there
        self.assertEqual(names["appeared"], ["phone"])

    def test_unknown_offset_makes_no_position_claims(self):
        now = {"bottle": {"world_angle": 80.0, "status": "VISIBLE", "comparable": False}}
        d = compare(self.REF, now, coverage=ViewCoverage(), offset=None)
        self.assertEqual(d["moved"], [])
        self.assertEqual(d["gone"], [])
        self.assertEqual(len(d["unchecked"]), 5)

    def test_appeared_then_left_is_not_new(self):
        now = {"phone": {"world_angle": 0.0, "status": "GONE"}}
        self.assertEqual(compare([], now, offset=0.0)["appeared"], [])


class TestRenamedAndFlicker(unittest.TestCase):
    def test_new_entity_after_move_is_a_move(self):
        # identity.py numbered the dog '2' at its new place; the old one went GONE
        ref = [{"name": "yellow dog", "world_angle": -20.0}, {"name": "bicycle", "world_angle": 0.0}]
        now = {"yellow dog": {"world_angle": -20.0, "status": "GONE"},
               "yellow dog 2": {"world_angle": 18.0, "status": "VISIBLE"},
               "bicycle": {"world_angle": 0.5, "status": "VISIBLE"}}
        d = compare(ref, now, offset=0.0)
        self.assertEqual([m["name"] for m in d["moved"]], ["yellow dog"])
        self.assertEqual(d["gone"] + d["appeared"], [])

    def test_flickering_item_is_not_new(self):
        ref = [{"name": "white bicycle", "world_angle": -20.0, "status": "GONE"}]
        now = {"white bicycle": {"world_angle": -20.0, "status": "VISIBLE"}}
        d = compare(ref, now, coverage=ViewCoverage(), offset=0.0)
        self.assertEqual(d["appeared"] + d["gone"] + d["unchecked"], [])
        # and if it is not seen again it is not reported gone either
        d = compare(ref, {}, coverage=ViewCoverage(), offset=0.0)
        self.assertEqual(d["gone"] + d["unchecked"], [])

    def test_speakable(self):
        self.assertEqual(scene_diff.speakable("yellow dog 2"), "yellow dog")
        self.assertEqual(scene_diff.speakable("my bottle"), "my bottle")


class TestReference(unittest.TestCase):
    def test_rules(self):
        snaps = [{"id": 4, "ts": 400, "session_id": "B", "kind": "auto"},
                 {"id": 3, "ts": 300, "session_id": "A", "kind": "auto"},
                 {"id": 2, "ts": 200, "session_id": "A", "kind": "mark"},
                 {"id": 1, "ts": 100, "session_id": "Z", "kind": "auto"}]
        # in run B: own auto skipped; A's auto skipped because A has an unanswered mark
        self.assertEqual(choose_reference(snaps, "B", 0)["id"], 2)
        # once that mark was answered, A's end-of-run picture is the newer 'before'
        self.assertEqual(choose_reference(snaps, "B", 250)["id"], 3)
        self.assertIsNone(choose_reference([snaps[0]], "B", 0))


class TestPrepareAcrossRuns(unittest.TestCase):
    def test_turned_laptop(self):
        ref = {"id": 1, "ts": 100.0, "session_id": "A"}
        ref_items = [{"name": "chair", "world_angle": -30.0}, {"name": "tv", "world_angle": 0.0},
                     {"name": "table", "world_angle": -10.0},
                     {"name": "bottle", "world_angle": 20.0}, {"name": "sneha", "world_angle": 5.0}]
        # run B: the laptop is turned 40 deg; the bottle was moved 30 deg further
        rows = [{"name": "chair", "world_angle": 10.0, "session_id": "B", "status": "VISIBLE", "last_seen": 200},
                {"name": "tv", "world_angle": 40.0, "session_id": "B", "status": "VISIBLE", "last_seen": 200},
                {"name": "table", "world_angle": 31.0, "session_id": "B", "status": "VISIBLE", "last_seen": 200},
                {"name": "bottle", "world_angle": 90.0, "session_id": "B", "status": "VISIBLE", "last_seen": 200},
                {"name": "lamp", "world_angle": 0.0, "session_id": "A", "status": "VISIBLE", "last_seen": 50}]
        items, now, off, known = prepare(ref, ref_items, rows, "B", is_person=lambda n: n == "sneha")
        self.assertTrue(known)
        self.assertAlmostEqual(off, 40.0, delta=0.5)
        self.assertNotIn("lamp", now)                     # seen before the snapshot
        self.assertNotIn("sneha", [i["name"] for i in items])
        d = compare(items, now, offset=off)
        self.assertEqual([m["name"] for m in d["moved"]], ["bottle"])
        self.assertEqual(sorted(s["name"] for s in d["same"]), ["chair", "table", "tv"])

    def test_two_of_three_is_not_enough(self):
        # two moved items that happen to agree must not out-vote the one that stayed
        ref = {"id": 1, "ts": 100.0, "session_id": "A"}
        ref_items = [{"name": "chair", "world_angle": 0.0}, {"name": "tv", "world_angle": 30.0},
                     {"name": "bag", "world_angle": -60.0}]
        rows = [{"name": "chair", "world_angle": 0.0, "session_id": "B", "status": "VISIBLE", "last_seen": 200},
                {"name": "tv", "world_angle": 104.0, "session_id": "B", "status": "VISIBLE", "last_seen": 200},
                {"name": "bag", "world_angle": 18.0, "session_id": "B", "status": "VISIBLE", "last_seen": 200}]
        self.assertFalse(prepare(ref, ref_items, rows, "B")[3])

    def test_only_one_shared_item(self):
        ref = {"id": 1, "ts": 100.0, "session_id": "A"}
        rows = [{"name": "chair", "world_angle": 10.0, "session_id": "B", "status": "VISIBLE", "last_seen": 200}]
        items, now, off, known = prepare(ref, [{"name": "chair", "world_angle": -30.0}], rows, "B")
        self.assertFalse(known)
        d = compare(items, now, offset=off)
        self.assertEqual(d["moved"], [])
        self.assertIn("can't tell if anything was moved", summarise(d, 0.0, label, "x", positions_known=known))


class TestSummary(unittest.TestCase):
    def test_wording(self):
        d = {"moved": [{"name": "bottle", "old_angle": 0.0, "new_angle": 60.0}],
             "gone": [{"name": "remote"}], "appeared": [{"name": "phone", "world_angle": -40.0}],
             "same": [{"name": "chair"}], "unchecked": [{"name": "laptop"}]}
        s = summarise(d, 0.0, label, "you left 5 minutes ago", people=["Sneha"])
        self.assertEqual(s, "Since you left 5 minutes ago: The bottle was moved from in front of you to on your "
                            "right. The remote is no longer where it was. New: the phone on your left. Sneha was here. "
                            "I haven't looked where the laptop was yet. Everything else is where it was.")

    def test_nothing_changed(self):
        d = {"moved": [], "gone": [], "appeared": [], "same": [{"name": "chair"}], "unchecked": []}
        self.assertEqual(summarise(d, 0.0, label, "you left 1 minute ago"),
                         "Since you left 1 minute ago: Nothing has changed that I can see.")

    def test_long_lists_are_capped(self):
        d = {"moved": [], "gone": [{"name": f"item{i}"} for i in range(6)], "appeared": [],
             "same": [], "unchecked": []}
        s = summarise(d, 0.0, label, "x")
        self.assertIn("3 more are no longer where they were", s)


class TestIntents(unittest.TestCase):
    def test_phrases(self):
        cases = {"I'm leaving": "MARK", "remember the room please": "MARK", "I’m back": "SINCE_LEFT",
                 "what changed since I left?": "SINCE_LEFT", "what happened while I was away": "SINCE_LEFT",
                 "what changed?": "CHANGES", "what changed in the last 10 minutes": "CHANGES",
                 "this is my bottle": "LEARN", "where was my bottle before": "PREVIOUS",
                 "remember as my keys": "LEARN", "what do you see in the room": "OVERVIEW"}
        for text, intent in cases.items():
            self.assertEqual(answers.classify_intent(text), intent, text)

    def test_time_window(self):
        self.assertEqual(answers.parse_time_window("what changed in the last 10 minutes"), 600)
        self.assertEqual(answers.parse_time_window("in the last hour"), 3600)
        self.assertEqual(answers.parse_time_window("past five minutes"), 300)
        self.assertEqual(answers.parse_time_window("half an hour"), 1800)
        self.assertIsNone(answers.parse_time_window("what changed"))


class TestSnapshotsInSQLite(unittest.TestCase):
    def setUp(self):
        import memory_db
        self.db = memory_db
        self.tmp = tempfile.TemporaryDirectory()
        memory_db.close()
        memory_db.DB_PATH = os.path.join(self.tmp.name, "t.db")
        memory_db.init_database()

    def tearDown(self):
        self.db.close()
        self.tmp.cleanup()

    def test_auto_replaced_marks_kept_old_pruned(self):
        items = [{"name": "Bottle", "zone": "ahead", "world_angle": 3.0, "dist_feet": None, "last_seen": 1.0}]
        self.db.save_snapshot(10, "A", "auto", 0.0, items)
        self.db.save_snapshot(20, "A", "auto", 0.0, items)
        self.db.save_snapshot(30, "A", "mark", 0.0, items)
        snaps = self.db.list_snapshots()
        self.assertEqual([(s["ts"], s["kind"]) for s in snaps], [(30, "mark"), (20, "auto")])
        self.assertEqual(self.db.get_snapshot_items(snaps[0]["id"])[0]["name"], "bottle")
        for k in range(30):
            self.db.save_snapshot(100 + k, "M", "mark", 0.0, items)
        self.assertEqual(len(self.db.list_snapshots()), self.db.MAX_SNAPSHOTS)
        n_rows = self.db._get_conn().execute("SELECT COUNT(*) FROM snapshot_items").fetchone()[0]
        self.assertEqual(n_rows, self.db.MAX_SNAPSHOTS)        # items of pruned snapshots deleted too

    def test_forget_and_rename_reach_snapshots(self):
        sid = self.db.save_snapshot(10, "A", "mark", 0.0, [
            {"name": "my bottle", "world_angle": 0.0, "last_seen": 1.0},
            {"name": "white cup", "world_angle": 9.0, "last_seen": 1.0}])
        self.db.rename("my bottle", "my medicine")
        self.db.forget("white cup")
        self.assertEqual([i["name"] for i in self.db.get_snapshot_items(sid)], ["my medicine"])


class TestRandomRooms(unittest.TestCase):
    """2,000 random rooms. Ground truth is known, so every claim can be checked."""

    NOUNS = ["bottle", "cup", "chair", "laptop", "book", "remote", "phone", "tv", "plant", "bag"]

    def run_room(self, rng, same_run):
        n = rng.randint(2, 7)
        names = rng.sample(self.NOUNS, n)
        before = {nm: rng.uniform(-150, 150) for nm in names}
        rotation = 0.0 if same_run else rng.choice([0.0, rng.uniform(-90, 90)])
        truth = {}
        after = {}
        for nm, a in before.items():
            r = rng.random()
            if r < 0.15:
                truth[nm] = "removed"
            elif r < 0.30:
                truth[nm] = "moved"
                after[nm] = a + rng.choice([-1, 1]) * rng.uniform(25, 90)
            else:
                truth[nm] = "same"
                after[nm] = a                                   # not moved
        if rng.random() < 0.5:
            extra = rng.choice([x for x in self.NOUNS if x not in before] or ["mug"])
            truth[extra] = "added"
            after[extra] = rng.uniform(-150, 150)

        # the camera looks at a few random directions for random lengths of time
        cov = ViewCoverage()
        looked = []
        for _ in range(rng.randint(0, 3)):
            h, frames = rng.uniform(-180, 180), rng.randint(0, 90)
            for _ in range(frames):
                cov.add(h, 1 / 15)
            if frames:
                looked.append(h)
        session = "A" if same_run else "B"
        rows = []
        for nm, a in after.items():
            cur = spatial.wrap_angle(a + rotation)
            # seen only if looked at; the outer 5 degrees of the view are missed
            if any(abs(spatial.wrap_angle(cur - h)) < spatial.CAMERA_FOV_DEG / 2 - 5 for h in looked):
                measured = spatial.wrap_angle(cur + rng.gauss(0, 2.0))       # angle measurement noise
                rows.append({"name": nm, "world_angle": measured, "session_id": session,
                             "status": "VISIBLE", "last_seen": 200.0})
        ref = {"id": 1, "ts": 100.0, "session_id": "A"}
        items = [{"name": nm, "world_angle": spatial.wrap_angle(a)} for nm, a in before.items()]
        items, now, off, known = prepare(ref, items, rows, session)
        return truth, compare(items, now, coverage=cov, offset=off,
                                    offset_estimated=not same_run), cov, rotation, known

    def check(self, same_run):
        rng = random.Random(7 if same_run else 8)
        stats = {"rooms": 0, "moved_found": 0, "moved_total": 0}
        for _ in range(2000):
            truth, d, cov, rotation, known = self.run_room(rng, same_run)
            stats["rooms"] += 1
            for m in d["moved"]:
                self.assertEqual(truth[m["name"]], "moved", (truth, d))
            for g in d["gone"]:     # said as "no longer where it was": removed OR moved away
                self.assertIn(truth[g["name"]], ("removed", "moved"), (truth, d))
            for a in d["appeared"]:
                self.assertEqual(truth[a["name"]], "added", (truth, d))
            stats["moved_total"] += sum(1 for v in truth.values() if v == "moved")
            stats["moved_found"] += len(d["moved"])
        return stats

    def test_same_run_never_false(self):
        s = self.check(same_run=True)
        self.assertGreater(s["moved_found"], 0.15 * s["moved_total"])  # it does find changes
        # (most moves are missed only because the random camera never looked there)

    def test_across_runs_never_false(self):
        s = self.check(same_run=False)
        self.assertGreater(s["moved_found"], 0)


if __name__ == "__main__":
    unittest.main()
