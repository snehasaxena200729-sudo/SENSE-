"""Foundation fixes 2 + 3: object continuity and proximity alerts.

A pinhole-camera simulator walks towards (or past, or away from) objects and
feeds noisy, sometimes missing detections with ByteTrack ids that change, the
way the field tests behaved. True distances are known, so alerts can be checked.
"""
import math
import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import continuity  # noqa: E402
import distance  # noqa: E402

W, H, FPS = 1280, 720, 15.0
F = distance.focal_px(W)
CAM_H = 1.3          # camera (chest / head) height in metres


def project(lateral, dist, height, width, base=0.0):
    """Box of an upright object standing on the floor (or at `base` m)."""
    cx = W / 2 + F * lateral / dist
    y_top = H / 2 - F * (base + height - CAM_H) / dist
    y_bot = H / 2 + F * (CAM_H - base) / dist
    return (max(0.0, cx - F * width / dist / 2), max(0.0, y_top),
            min(W, cx + F * width / dist / 2), min(H, y_bot))


def visible(box):
    return box[2] - box[0] > 4 and box[3] - box[1] > 4 and box[0] < W and box[2] > 0


class Sim:
    """Generates detections for a list of objects along a camera path."""

    def __init__(self, seed=0, miss=0.25, id_switch_every=12, sway_px=12, noise=0.03):
        self.rng = random.Random(seed)
        self.miss, self.switch, self.sway, self.noise = miss, id_switch_every, sway_px, noise
        self.tracker = continuity.ObjectContinuity({"chair", "bed", "door", "person"})
        self.alerts = []
        self.ids_seen = {}
        self.prev_sway = (0.0, 0.0)

    def step(self, i, objects):
        """objects: list of (name, cls, lateral m, distance m, height m, width m)."""
        now = i / FPS
        sway = (self.sway * math.sin(i / 6.0), 0.5 * self.sway * math.cos(i / 9.0))
        shift = (sway[0] - self.prev_sway[0], sway[1] - self.prev_sway[1])
        self.prev_sway = sway
        dets = []
        for k, (name, cls, lat, d, h, w) in enumerate(objects):
            box = project(lat, d, h, w)
            box = (box[0] + sway[0], box[1] + sway[1], box[2] + sway[0], box[3] + sway[1])
            box = tuple(min(max(v, 0.0), lim) for v, lim in zip(box, (W, H, W, H)))
            if not visible(box) or self.rng.random() < self.miss:
                continue
            j = lambda: 1 + self.rng.uniform(-self.noise, self.noise)
            cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            bw, bh = (box[2] - box[0]) * j(), (box[3] - box[1]) * j()
            box = (max(0, cx - bw / 2), max(0, cy - bh / 2), min(W, cx + bw / 2), min(H, cy + bh / 2))
            bt = 100 * k + i // self.switch            # ByteTrack renumbers regularly
            est = distance.object_distance(cls, box, W, H, F)
            dets.append({"cls": cls, "box": box, "conf": 0.6, "bt_id": bt, "can_create": True,
                         "est": est, "name": name})
        matched, created = self.tracker.update(dets, now, shift)
        for obj, d in matched + created:
            self.ids_seen.setdefault(d["name"], set()).add(obj.oid)
        for obj, words in self.tracker.proximity_alerts(now, W):
            true = {name: dist for name, _, _, dist, _, _ in objects}
            self.alerts.append((i, obj.cls, words, obj.oid))
        return now


class TestContinuity(unittest.TestCase):
    def test_walking_to_a_chair_keeps_one_identity(self):
        sim = Sim()
        for i in range(int(4.0 * FPS)):                 # 3.5 m -> 0.9 m at 0.65 m/s
            sim.step(i, [("chair", "chair", 0.0, 3.5 - 0.65 * i / FPS, 0.9, 0.5)])
        self.assertEqual(len(sim.ids_seen["chair"]), 1, sim.ids_seen)

    def test_one_alert_when_near_not_before(self):
        sim = Sim()
        truth = []
        for i in range(int(4.0 * FPS)):
            d = 3.5 - 0.65 * i / FPS
            truth.append(d)
            sim.step(i, [("chair", "chair", 0.0, d, 0.9, 0.5)])
        self.assertEqual(len(sim.alerts), 1, sim.alerts)
        frame = sim.alerts[0][0]
        self.assertLess(truth[frame], 2.0)              # not far away
        self.assertGreater(truth[frame], 0.8)           # before you hit it

    def test_standing_still_near_a_chair_is_quiet(self):
        sim = Sim()
        for i in range(int(10 * FPS)):
            sim.step(i, [("chair", "chair", 0.0, 1.2, 0.9, 0.5)])
        self.assertEqual(sim.alerts, [])

    def test_turning_the_head_is_not_approaching(self):
        sim = Sim(sway_px=0)
        for i in range(int(6 * FPS)):                   # object sweeps across the view
            lat = -1.0 + 2.0 * i / (6 * FPS)
            sim.step(i, [("bed", "bed", lat, 1.3, 0.6, 1.6)])
        self.assertEqual(sim.alerts, [])

    def test_turning_past_a_big_object_cut_off_at_the_side_is_quiet(self):
        """Field bug found in the smoke run: a big object sliding in from the
        frame edge while you turn 'grows' (more of it becomes visible). That is
        not approaching."""
        for seed in range(10):
            sim = Sim(seed=seed, sway_px=0)
            for i in range(int(8 * FPS)):
                lat = 1.6 - 3.2 * i / (8 * FPS)
                sim.step(i, [("bed", "bed", lat, 1.4, 0.6, 1.8)])
            self.assertEqual(sim.alerts, [], seed)

    def test_walking_past_to_the_side_is_quiet(self):
        sim = Sim()
        for i in range(int(4 * FPS)):
            sim.step(i, [("chair", "chair", 1.2, 3.0 - 0.6 * i / FPS, 0.9, 0.5)])
        self.assertEqual(sim.alerts, [])

    def test_alert_again_only_after_moving_away(self):
        sim = Sim()
        path = ([3.5 - 0.65 * t / FPS for t in range(int(3.5 * FPS))] +
                [1.2 + 0.6 * t / FPS for t in range(int(3 * FPS))] +
                [3.0 - 0.65 * t / FPS for t in range(int(3.5 * FPS))])
        for i, d in enumerate(path):
            sim.step(i, [("chair", "chair", 0.0, d, 0.9, 0.5)])
        self.assertEqual(len(sim.alerts), 2, sim.alerts)

    def test_two_chairs_keep_separate_identities(self):
        sim = Sim()
        for i in range(int(5 * FPS)):
            d = 4.0 - 0.4 * i / FPS
            sim.step(i, [("left chair", "chair", -0.45, d, 0.9, 0.5),
                         ("right chair", "chair", 0.45, d + 0.3, 0.9, 0.5)])
        self.assertEqual(len(sim.ids_seen["left chair"]), 1)
        self.assertEqual(len(sim.ids_seen["right chair"]), 1)
        self.assertNotEqual(sim.ids_seen["left chair"], sim.ids_seen["right chair"])

    def test_door_cut_off_close_up_still_alerts_once(self):
        sim = Sim()
        for i in range(int(5 * FPS)):                   # 3.5 m -> 0.6 m: the 2 m door gets cut off
            sim.step(i, [("door", "door", 0.0, 3.5 - 0.6 * i / FPS, 2.0, 0.85)])
        self.assertEqual(len(sim.ids_seen["door"]), 1)
        self.assertEqual(len(sim.alerts), 1, sim.alerts)

    def test_short_gap_keeps_identity_long_gap_does_not(self):
        sim = Sim(miss=0.0)
        obj = [("chair", "chair", 0.0, 2.5, 0.9, 0.5)]
        i = 0
        for _ in range(15):
            sim.step(i, obj); i += 1
        for _ in range(int(1.5 * FPS)):                 # hidden 1.5 s (detector runs, sees nothing)
            sim.step(i, []); i += 1
        for _ in range(5):
            sim.step(i, obj); i += 1
        self.assertEqual(len(sim.ids_seen["chair"]), 1)
        for _ in range(int(3.0 * FPS)):                 # hidden 3 s: forgotten, new identity
            sim.step(i, []); i += 1
        for _ in range(5):
            sim.step(i, obj); i += 1
        self.assertEqual(len(sim.ids_seen["chair"]), 2)

    def test_reappearing_closer_after_a_gap_is_not_an_approach(self):
        """Smoke-run bug: a person left the view and came back much bigger;
        the size jump across the gap was read as walking into them."""
        sim = Sim(miss=0.0, sway_px=0)
        i = 0
        for _ in range(int(1.5 * FPS)):
            sim.step(i, [("chair", "chair", 0.0, 2.6, 0.9, 0.5)]); i += 1
        for _ in range(int(1.0 * FPS)):
            sim.step(i, []); i += 1
        for _ in range(int(3 * FPS)):
            sim.step(i, [("chair", "chair", 0.0, 1.3, 0.9, 0.5)]); i += 1
        self.assertEqual(sim.alerts, [])

    def test_stalled_loop_keeps_identity(self):
        """A 3.5 s stall (model loading, slow frame) with NO detector runs is not
        a disappearance: nothing was looked for, so nothing was missed."""
        sim = Sim(miss=0.0)
        obj = [("chair", "chair", 0.0, 2.5, 0.9, 0.5)]
        for i in range(5):
            sim.step(i, obj)
        for i in range(5 + int(3.5 * FPS), 10 + int(3.5 * FPS)):
            sim.step(i, obj)
        self.assertEqual(len(sim.ids_seen["chair"]), 1)

    def test_missed_object_is_not_reported_visible(self):
        t = continuity.ObjectContinuity({"chair"})
        det = {"cls": "chair", "box": (500, 300, 700, 600), "conf": 0.6, "bt_id": 1,
               "can_create": True, "est": None}
        t.update([det], 0.0)
        t.update([], 0.1)
        self.assertEqual(t.visible(0.1), [])            # coasting, not drawn or spoken
        self.assertEqual(len(t.objects), 1)

    def test_weak_detection_extends_but_never_creates(self):
        t = continuity.ObjectContinuity({"chair"})
        weak = {"cls": "chair", "box": (500, 300, 700, 600), "conf": 0.15, "bt_id": 1,
                "can_create": False, "est": None}
        t.update([weak], 0.0)
        self.assertEqual(t.objects, {})
        t.update([dict(weak, can_create=True)], 0.1)
        t.update([weak], 0.2)
        self.assertEqual(len(t.objects), 1)
        self.assertEqual(t.visible(0.2)[0].hits, 2)

    def test_other_class_never_matches(self):
        t = continuity.ObjectContinuity({"chair", "bed"})
        t.update([{"cls": "chair", "box": (500, 300, 700, 600), "bt_id": 1, "can_create": True}], 0.0)
        t.update([{"cls": "bed", "box": (500, 300, 700, 600), "bt_id": 1, "can_create": True}], 0.1)
        self.assertEqual(sorted(o.cls for o in t.objects.values()), ["bed", "chair"])


class TestRandomised(unittest.TestCase):
    """Many random walks: invariants must hold every time."""

    def test_random_walks(self):
        for seed in range(40):
            rng = random.Random(seed)
            sim = Sim(seed=seed, miss=rng.uniform(0.0, 0.4), id_switch_every=rng.randint(3, 30),
                      sway_px=rng.uniform(0, 20), noise=rng.uniform(0.0, 0.05))
            start, speed = rng.uniform(3.0, 5.0), rng.uniform(0.4, 1.0)
            for i in range(int((start - 0.7) / speed * FPS)):
                sim.step(i, [("chair", "chair", rng.uniform(-0.05, 0.05), start - speed * i / FPS, 0.9, 0.5)])
            self.assertEqual(len(sim.ids_seen["chair"]), 1, (seed, sim.ids_seen))
            self.assertLessEqual(len(sim.alerts), 1, (seed, sim.alerts))


if __name__ == "__main__":
    unittest.main()
