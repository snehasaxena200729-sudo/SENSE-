"""Unit tests for the roadmap modules. Run from the project folder:

    python -m unittest discover -s tests -v

They need only numpy (and OpenCV if installed); no camera, mic or YOLO.
"""
import os
import sys
import tempfile
import types
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import answers  # noqa: E402
import spatial  # noqa: E402
from events import EventTracker  # noqa: E402
from identity import EntityResolver  # noqa: E402
from notifier import Notifier, P1, P2, P3  # noqa: E402


def sig_for(seed):
    """A fake 16x20 chroma histogram; same seed = same appearance."""
    rng = np.random.default_rng(seed)
    h = rng.random((16, 20)).astype(np.float32)
    return h / h.max()


class TestIdentity(unittest.TestCase):
    def test_colour_flicker_gives_one_named_entity(self):
        r = EntityResolver()
        colours = ["red"] * 7 + ["orange"] * 3 + ["red"] * 5
        eids = {r.resolve(1, "phone", sig_for(1), c, (100, 100), t * 0.03) for t, c in enumerate(colours)}
        self.assertEqual(len(eids), 1)
        self.assertEqual(r.name_of(eids.pop()), "red phone")

    def test_reidentifies_after_track_loss(self):
        r = EntityResolver()
        for t in range(10):
            first = r.resolve(1, "cup", sig_for(7), "white", (200, 200), t * 0.03)
        # object hidden, comes back later with a NEW ByteTrack id
        again = r.resolve(42, "cup", sig_for(7) * 0.98, "white", (205, 198), 30.0)
        self.assertEqual(first, again)

    def test_different_appearance_is_new_entity(self):
        r = EntityResolver()
        a = r.resolve(1, "cup", sig_for(1), "white", (100, 100), 0.0)
        b = r.resolve(2, "cup", sig_for(99), "white", (400, 100), 0.0)
        self.assertNotEqual(a, b)

    def test_duplicate_names_get_numbered(self):
        r = EntityResolver()
        for t in range(10):
            a = r.resolve(1, "cup", sig_for(1), "white", (100, 100), t * 0.03)
            b = r.resolve(2, "cup", sig_for(99), "white", (400, 100), t * 0.03)
        self.assertEqual({r.name_of(a), r.name_of(b)}, {"white cup", "white cup 2"})

    def test_unclear_colour_stays_unnamed(self):
        r = EntityResolver()
        for t, c in enumerate(["red", "blue", "green", "gray"] * 3):
            e = r.resolve(1, "book", sig_for(3), c, (10, 10), t)
        self.assertIsNone(r.name_of(e))


class TestSpatial(unittest.TestCase):
    def test_turn_after_seeing_object(self):
        # Seen dead ahead while the user faced +90 degrees...
        angle = spatial.world_angle(320, 640, heading=90)
        # ...still facing +90: it is in front (the old code said "on your left")
        self.assertEqual(spatial.direction_phrase(spatial.relative_angle(angle, 90)), "in front of you")
        # turned back to 0: it is now on the right, about 90 degrees
        rel = spatial.relative_angle(angle, 0)
        self.assertEqual(spatial.direction_phrase(rel), "on your right")
        self.assertEqual(spatial.turn_instruction(rel), "turn about 90 degrees to your right")

    def test_behind_and_wrap(self):
        self.assertEqual(spatial.direction_phrase(spatial.relative_angle(175, -175)), "in front of you")
        self.assertEqual(spatial.direction_phrase(spatial.relative_angle(0, 180)), "behind you")

    def test_relation(self):
        t = {"name": "phone", "world_angle": 10.0, "last_y": 200, "last_seen": 100}
        others = [{"name": "laptop", "world_angle": 11.0, "last_y": 210, "last_seen": 99},
                  {"name": "cup", "world_angle": -20.0, "last_y": 200, "last_seen": 99}]
        n = spatial.nearest_neighbour(t, others)
        self.assertEqual(n["name"], "laptop")
        self.assertEqual(spatial.relation_phrase(t, n), "next to the laptop")


class TestEvents(unittest.TestCase):
    def obs(self, angle, zone="in front of you"):
        return {"zone": zone, "world_angle": angle, "x": 0, "y": 0, "dist_feet": 2}

    def test_single_missed_frame_is_not_gone(self):
        et = EventTracker()
        et.update({"Cup": self.obs(0)}, 0.0, True, 0)
        self.assertEqual(et.update({}, 0.5, True, 0), [])
        self.assertEqual(et.update({"Cup": self.obs(0)}, 0.6, True, 0), [])

    def test_gone_vs_out_of_view(self):
        et = EventTracker()
        et.update({"Cup": self.obs(0), "Book": self.obs(0)}, 0.0, True, 0)
        et.update({"Cup": self.obs(0), "Book": self.obs(0)}, 4.0, True, 0)
        ev = et.update({}, 4.0 + EventTracker.GONE_AFTER_S + 0.5, True, heading=0)  # still facing the spot
        self.assertEqual({e["type"] for e in ev}, {"GONE"})
        et2 = EventTracker()
        et2.update({"Cup": self.obs(0)}, 0.0, True, 0)
        ev2 = et2.update({}, EventTracker.GONE_AFTER_S + 1, True, heading=120)  # user turned away
        self.assertEqual(ev2[0]["type"], "OUT_OF_VIEW")

    def test_moved_needs_votes_and_ignores_head_turns(self):
        et = EventTracker()
        et.update({"Cup": self.obs(0, "on your left")}, 0, True, 0)
        types = []
        for i in range(1, 6):
            types += [e["type"] for e in et.update({"Cup": self.obs(30, "on your right")}, i, True, 0)]
        self.assertEqual(types, ["MOVED"])
        # same world angle seen from a different head direction = not moved
        et2 = EventTracker()
        et2.update({"Cup": self.obs(0, "in front of you")}, 0, True, 0)
        for i in range(1, 6):
            self.assertEqual(et2.update({"Cup": self.obs(1, "on your left")}, i, True, 30), [])

    def test_returned_after_gone(self):
        et = EventTracker()
        et.update({"Cup": self.obs(0)}, 0, True, 0)
        et.update({}, EventTracker.GONE_AFTER_S + 1, True, 0)
        ev = et.update({"Cup": self.obs(25)}, EventTracker.GONE_AFTER_S + 5, True, 0)
        self.assertEqual(ev[0]["type"], "RETURNED")
        self.assertIn("from_zone", ev[0])


    def test_brief_sighting_is_flagged(self):
        """A 'phone' seen for one second (likely a false detection) must not be announced."""
        et = EventTracker()
        et.update({"Phone": self.obs(0)}, 0, True, 0)
        et.update({"Phone": self.obs(0)}, 1, True, 0)
        ev = et.update({}, 1 + EventTracker.GONE_AFTER_S + 0.1, True, 0)
        self.assertEqual(ev[0]["type"], "GONE")
        self.assertTrue(ev[0].get("brief"))
        # a long, steady sighting is a real disappearance
        et.update({"Cup": self.obs(0)}, 20, True, 0)
        et.update({"Cup": self.obs(0)}, 30, True, 0)
        ev = et.update({}, 30 + EventTracker.GONE_AFTER_S + 0.1, True, 0)
        self.assertFalse(ev[0].get("brief"))


class TestNotifier(unittest.TestCase):
    def setUp(self):
        self.t = [0.0]
        self.said = []
        self.busy = [False]
        self.n = Notifier(self.said.append, lambda: self.busy[0], clock=lambda: self.t[0])

    def test_cooldown_and_levels(self):
        self.assertTrue(self.n.push(P2, "cup", "Cup moved"))
        self.n.tick()
        self.t[0] = 10
        self.assertFalse(self.n.push(P2, "cup", "Cup moved again"))   # within 30 s
        self.assertFalse(self.n.push(P3, "book", "Book appeared"))    # P3 hidden at normal
        self.assertEqual(self.said, ["Cup moved."])

    def test_waits_while_speaking_but_p1_interrupts(self):
        self.busy[0] = True
        self.n.push(P2, "cup", "Cup moved")
        self.assertIsNone(self.n.tick())
        self.n.push(P1, "chair", "Careful, chair ahead")
        self.assertEqual(self.n.tick(), "Careful, chair ahead.")

    def test_merges_two_and_drops_stale(self):
        self.n.push(P2, "a", "A moved")
        self.n.push(P2, "b", "B moved")
        self.assertEqual(self.n.tick(), "A moved. B moved.")
        self.t[0] = 100
        self.n.push(P2, "c", "C moved")
        self.t[0] = 120     # waited 20 s > MAX_WAIT_S
        self.assertIsNone(self.n.tick())

    def test_hold_while_listening_only_p1(self):
        self.n.push(P2, "cup", "Cup moved")
        self.assertIsNone(self.n.tick(hold=True))
        self.n.push(P1, "obstacle", "Stop")
        self.assertEqual(self.n.tick(hold=True), "Stop.")
        self.t[0] = Notifier.MIN_GAP_S + 1
        self.assertEqual(self.n.tick(), "Cup moved.")

    def test_quiet_mode(self):
        self.n.set_level("quiet")
        self.assertFalse(self.n.push(P2, "cup", "Cup moved"))
        self.assertTrue(self.n.push(P1, "x", "Careful"))


class TestAnswers(unittest.TestCase):
    def test_intents(self):
        cases = {
            "what changed while i was away": "CHANGES",
            "where was my bottle before": "PREVIOUS",
            "when did you last see my phone": "WHEN",
            "who was here today": "WHO",
            "what do you see": "OVERVIEW",
            "where is the cup": "WHERE",
            "this is my water bottle": "LEARN",
            "forget sneha": "FORGET",
            "be quiet": "QUIET",
        }
        for text, intent in cases.items():
            self.assertEqual(answers.classify_intent(text), intent, text)

    def test_fuzzy(self):
        names = ["sneha", "white cup", "my bottle", "phone"]
        self.assertEqual(answers.fuzzy_find("where is the white cup", names), "white cup")
        self.assertEqual(answers.fuzzy_find("where is sneh", names), "sneha")
        self.assertEqual(answers.fuzzy_find("where is my bottle", names), "my bottle")
        self.assertIsNone(answers.fuzzy_find("where is the elephant", names))

    def test_confidence_tiers(self):
        self.assertEqual(answers.tier(answers.confidence(1, "VISIBLE", True)), "confirmed")
        self.assertEqual(answers.tier(answers.confidence(60, "VISIBLE", True)), "likely")
        self.assertEqual(answers.tier(answers.confidence(60, "GONE", True)), "uncertain")
        self.assertEqual(answers.tier(answers.confidence(7200, "VISIBLE", True)), "uncertain")
        s, level, _ = answers.describe_memory("the cup", 300, "GONE", "on your left", True)
        self.assertIn("disappeared", s)

    def test_summarise_changes(self):
        evs = [{"name": "the cup", "type": "MOVED", "zone": "on your left", "ts": 1},
               {"name": "your bottle", "type": "GONE", "zone": "in front of you", "ts": 2}]
        self.assertEqual(answers.summarise_changes(evs),
                         "2 things changed: your bottle disappeared; the cup was moved, now on your left.")
        self.assertEqual(answers.summarise_changes([]), "Nothing has changed that I noticed.")


class TestRename(unittest.TestCase):
    def test_parse_and_intent(self):
        for text in ["rename my bottle to medicine", "change my bottle to medicine",
                     "change the name of my bottle to my medicine",
                     "call my bottle my medicine instead"]:
            self.assertEqual(answers.classify_intent(text), "RENAME", text)
            self.assertEqual(answers.parse_rename(text), ("my bottle", "my medicine"), text)
        self.assertEqual(answers.classify_intent("what changed"), "CHANGES")
        self.assertEqual(answers.classify_intent("this is my medicine bottle"), "LEARN")
        # extra words around the old name are fixed later by fuzzy matching
        old, new = answers.parse_rename("change input of my bottle to medicine")
        self.assertEqual(answers.fuzzy_find(old, ["my bottle", "white cup"]), "my bottle")


class TestHazards(unittest.TestCase):
    def setUp(self):
        import hazards
        if hazards.cv2 is None:
            self.skipTest("OpenCV not installed")
        self.hazards = hazards
        rng = np.random.default_rng(0)
        tex = (rng.random((60, 80)) * 255).astype(np.uint8)
        import cv2
        self.cv2 = cv2
        self.base = cv2.resize(tex, (1280, 960), interpolation=cv2.INTER_CUBIC)

    def frame(self, scale=1.0, shift=0):
        cv2 = self.cv2
        h, w = 480, 640
        m = np.float32([[scale, 0, w / 2 - scale * 640 + shift], [0, scale, h / 2 - scale * 480]])
        return cv2.warpAffine(self.base, m, (w, h), borderMode=cv2.BORDER_REFLECT)

    def run_seq(self, scales, shifts):
        d = self.hazards.LoomingDetector()
        alerts = [d.update(self.frame(s, x), 1 / 15.0) for s, x in zip(scales, shifts)]
        return [a for a in alerts if a is not None]

    def test_fast_approach_alerts(self):
        # 8% bigger per frame at 15 fps = contact in ~0.8 s (true TTC = 1/0.08 frames)
        self.assertTrue(self.run_seq([1.08 ** i for i in range(10)], [0] * 10))

    def test_still_and_panning_do_not_alert(self):
        self.assertFalse(self.run_seq([1.0] * 10, [0] * 10))
        self.assertFalse(self.run_seq([1.0] * 10, [3 * i for i in range(10)]))

    def test_box_growth(self):
        t = self.hazards.ApproachTracker()
        self.assertFalse(t.update(1, 0.0, 120, 480, 0))
        self.assertTrue(t.update(1, 0.4, 180, 480, 0))      # 25% -> 37% of the view
        # someone sitting at the webcam fills the view: no warning without real growth
        t3 = self.hazards.ApproachTracker()
        t3.update(3, 0.0, 300, 480, 0, is_person=True)
        self.assertFalse(t3.update(3, 0.4, 320, 480, 0, is_person=True))
        t2 = self.hazards.ApproachTracker()
        t2.update(2, 0.0, 120, 480, 40)
        self.assertFalse(t2.update(2, 0.4, 180, 480, 40))   # off to the side


class TestSceneHazards(unittest.TestCase):
    """Door warnings with a fake Open Images model (the real one needs PyTorch)."""
    class FakeModel:
        names = {0: "Person", 164: "Door", 165: "Door handle", 489: "Stairs"}

        def __init__(self):
            self.h = 150
            self.calls = 0

        def predict(self, frame, classes=None, **kw):
            self.calls += 1
            assert set(classes) == {164, 165, 489}
            box = types.SimpleNamespace(cls=[164], conf=[0.3], xyxy=[np.array([270, 240 - self.h // 2, 370, 240 + self.h // 2])])
            return [types.SimpleNamespace(boxes=[box])]

    def test_door_detected_every_n_frames(self):
        # announcements / proximity moved to continuity.py (tests/test_continuity.py)
        import hazards
        fake = self.FakeModel()
        det = hazards.SceneHazardDetector(fake)
        frame = np.zeros((480, 640, 3), np.uint8)
        results = [det.detect(frame, i, i / 15.0) for i in range(1, 16)]
        self.assertEqual(fake.calls, 3)
        self.assertEqual([r[0][0] for r in results if r], ["door"] * 3)
        self.assertTrue(det.boxes and det.boxes[0][0] == "door")
        self.assertIn("door", det.visible(1.0))


class TestMemoryDB(unittest.TestCase):
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

    def test_latest_memory_wins_over_stale_row(self):
        self.db.save_object_observation("bottle", "on your left", 1, 1, seen_at=100)
        self.db.save_object_observation("white bottle", "on your right", 1, 1, seen_at=200,
                                        world_angle=20.0, session_id="s")
        rec = self.db.find_latest_memory("bottle")
        self.assertEqual(rec["name"], "white bottle")
        self.assertEqual(rec["world_angle"], 20.0)

    def test_events_and_forget(self):
        self.db.log_event(10, "my bottle", "MOVED", "on your left", 5.0, "s", 2.0, "from on your right")
        self.db.save_object_observation("my bottle", "on your left", 1, 1, seen_at=10)
        self.assertEqual(self.db.get_item_history("bottle")[0]["detail"], "from on your right")
        self.assertEqual(len(self.db.get_events_since(0)), 1)
        self.assertGreater(self.db.forget("my bottle"), 0)
        self.assertIsNone(self.db.find_latest_memory("bottle"))

    def test_rename_keeps_profile_and_history(self):
        self.db.save_custom_object_profile("my bottle", "bottle", sig_for(1))
        self.db.save_object_observation("my bottle", "on your left", 1, 1, seen_at=10)
        self.db.log_event(10, "My Bottle", "MOVED", "on your left")
        self.assertGreater(self.db.rename("my bottle", "my medicine"), 0)
        self.assertIn("my medicine", self.db.load_all_custom_object_profiles())
        self.assertEqual(self.db.find_latest_memory("medicine")["name"], "my medicine")
        self.assertEqual(len(self.db.get_item_history("medicine")), 1)
        self.assertIsNone(self.db.find_latest_memory("bottle"))

    def test_entities_roundtrip_and_settings(self):
        self.db.save_entity("cup-1", "white cup", "cup", "white", sig_for(1), 5.0)
        ents = self.db.load_entities()
        self.assertEqual(ents["cup-1"]["name"], "white cup")
        self.assertTrue(np.allclose(ents["cup-1"]["sig"], sig_for(1)))
        self.db.set_setting("notify_level", "quiet")
        self.assertEqual(self.db.get_setting("notify_level"), "quiet")


if __name__ == "__main__":
    unittest.main()
