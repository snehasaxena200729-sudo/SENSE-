"""Door distance enhancement: measurement, smoothing, narration, memory.

Doors are SIMULATED with the pinhole camera model: a real door (width x height)
at a known depth and sideways offset is projected into a 1280x720 frame, with
box jitter, cut-off edges and missed detections like the real door model.
That gives exact ground truth, so errors can be checked in numbers.
It cannot replace the tape-measure test on the real camera
(tools/door_distance_test.py, docs/PHYSICAL_TESTS.md section 8).
"""
import math
import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import distance  # noqa: E402
import door_distance as dd  # noqa: E402
from continuity import ObjectContinuity  # noqa: E402

FW, FH = 1280, 720
F_PX = distance.focal_px(FW)              # 65 degree view, as in SENSE
CAM_H = 0.95                              # MacBook on a table, lens height (m)
FT = 0.3048


def door_box(depth, lateral=0.0, width=0.85, height=2.03, jitter=0.0, rng=None, hide_below=None):
    """Box the detector would give for a door at `depth` m (None if out of view)."""
    def j():
        return rng.gauss(0, jitter) if rng else 0.0
    cx, cy = FW / 2, FH / 2
    x1 = cx + F_PX * (lateral - width / 2) / depth
    x2 = cx + F_PX * (lateral + width / 2) / depth
    bottom_h = 0.0 if hide_below is None else hide_below
    y1 = cy - F_PX * (height - CAM_H) / depth
    y2 = cy + F_PX * (CAM_H - bottom_h) / depth
    w = x2 - x1
    x1, x2 = x1 + j() * w, x2 + j() * w
    y1, y2 = y1 + j() * w, y2 + j() * w
    x1, y1 = max(0.0, x1), max(0.0, y1)
    x2, y2 = min(FW - 1.0, x2), min(FH - 1.0, y2)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    return (x1, y1, x2, y2)


class Sim:
    """continuity.py + DoorRanger fed like sense_master.update_scene_objects()."""

    def __init__(self, cal=None, allow_updates=True):
        self.objs = ObjectContinuity({"door", "stairs", "door handle"}, source="scene")
        self.ranger = dd.DoorRanger(cal)
        self.said = []            # (time, text)
        self.allow = allow_updates

    def step(self, t, box):
        dets = [] if box is None else [{"cls": "door", "box": box, "conf": 0.6, "bt_id": None,
                                        "can_create": True, "est": None}]
        _, created = self.objs.update(dets, t, (0.0, 0.0), frame_size=(FW, FH))
        for obj, _ in created:
            self.ranger.on_created(obj, t, 0.0)
        for _, text in self.ranger.update(self.objs.objects, t, FW, FH, F_PX, allow_updates=self.allow):
            self.said.append((t, text))


def spoken_feet(text):
    words = text.split()
    for a, b in zip(words, words[1:]):
        if a == "approximately":
            return int(b)
    return None


RATE = 6.0   # door model runs per second (every 5th frame at ~30 fps)


class TestMeasure(unittest.TestCase):
    def test_static_distances_uncalibrated(self):
        """2/4/6/8 ft: a number within the stated uncertainty, or honest 'very close' at 2 ft."""
        for ft in (4, 6, 8, 10, 12):
            m = dd.measure(door_box(ft * FT), FW, FH, F_PX)
            self.assertTrue(m.ok, ft)
            err = abs(m.metres - ft * FT) / (ft * FT)
            self.assertLess(err, 0.05, f"{ft} ft: {m.metres / FT:.2f} ft via {m.cue}")
            self.assertLessEqual(m.unc, dd.MAX_SYS_UNC)
        m = dd.measure(door_box(2 * FT), FW, FH, F_PX)
        self.assertFalse(m.ok)
        self.assertEqual(m.reason, "wider than the view")

    def test_width_cue_close_height_cue_far(self):
        self.assertEqual(dd.measure(door_box(6 * FT), FW, FH, F_PX).cue, "width")    # top/bottom cut
        self.assertEqual(dd.measure(door_box(14 * FT), FW, FH, F_PX).cue, "height")  # whole door

    def test_side_cut_off_is_not_measured(self):
        box = door_box(6 * FT, lateral=1.4)       # sliding out of view on the right
        self.assertIn("right", distance.clipped_edges(box, FW, FH))
        self.assertFalse(dd.measure(box, FW, FH, F_PX).ok)

    def test_partly_hidden_door_rejected(self):
        """Lower part hidden by furniture, whole door otherwise in view: wrong shape."""
        box = door_box(14 * FT, hide_below=1.3)
        m = dd.measure(box, FW, FH, F_PX)
        self.assertFalse(m.ok)
        self.assertIn("door shape", m.reason)

    def test_off_centre_uses_range_not_depth(self):
        depth, lateral = 8 * FT, 1.0
        m = dd.measure(door_box(depth, lateral=lateral), FW, FH, F_PX)
        true_range = math.hypot(depth, lateral)
        self.assertLess(abs(m.metres - true_range) / true_range, 0.05)

    def test_calibration_fixes_unusual_door_width(self):
        """A 0.95 m door: uncalibrated reads ~11% short; calibrated at 6 ft fixes it."""
        wide = dict(width=0.95)
        b6 = door_box(6 * FT, **wide)
        cal = {"w": dd.calibration_value(b6[2] - b6[0], 6 * FT, 0.0, FW)}
        for ft in (4, 8):
            raw = dd.measure(door_box(ft * FT, **wide), FW, FH, F_PX)
            calm = dd.measure(door_box(ft * FT, **wide), FW, FH, F_PX, cal=cal)
            self.assertGreater(abs(raw.metres - ft * FT) / (ft * FT), 0.08)
            self.assertLess(abs(calm.metres - ft * FT) / (ft * FT), 0.02)
            self.assertLess(calm.unc, raw.unc)

    def test_calibration_loading(self):
        vals = {"door_cal_w": "0.53", "door_cal_h": "bad"}
        self.assertEqual(dd.load_calibration(lambda k, d=None: vals.get(k, d)), {"w": 0.53})

    def test_directions(self):
        self.assertEqual(dd.door_direction(0), "ahead")
        self.assertEqual(dd.door_direction(15), "ahead on your right")
        self.assertEqual(dd.door_direction(-30), "to your left")
        self.assertEqual(dd.door_direction(170), "behind you")


class TestNarration(unittest.TestCase):
    def test_first_announcement_has_distance(self):
        sim = Sim()
        rng = random.Random(1)
        for k in range(12):
            sim.step(k / RATE, door_box(8 * FT, jitter=0.01, rng=rng))
        self.assertEqual(len(sim.said), 1, sim.said)
        t, text = sim.said[0]
        self.assertTrue(text.startswith("Door detected approximately"), text)
        self.assertTrue(text.endswith("feet ahead"), text)
        self.assertIn(spoken_feet(text), (7, 8, 9))
        self.assertLessEqual(t, 1.0)          # within a second of first sight

    def test_walking_towards_door(self):
        """10 ft -> 2.5 ft at 0.45 m/s (careful walking): fresh numbers, decreasing,
        near the truth, not chatty."""
        sim = Sim()
        rng = random.Random(2)
        speed, start = 0.45, 10 * FT
        t, truth = 0.0, {}
        while True:
            depth = start - speed * t
            if depth < 2.5 * FT:
                break
            sim.step(t, door_box(depth, jitter=0.01, rng=rng))
            truth[round(t, 3)] = depth
            t += 1 / RATE
        nums = [(tt, spoken_feet(x)) for tt, x in sim.said if spoken_feet(x)]
        self.assertGreaterEqual(len(nums), 3, sim.said)
        self.assertLessEqual(len(sim.said), 6, sim.said)
        self.assertEqual([n for _, n in nums], sorted([n for _, n in nums], reverse=True))
        for tt, n in nums:
            true_ft = truth[round(tt, 3)] / FT
            self.assertLess(abs(n - true_ft), max(1.2, 0.2 * true_ft), (tt, n, true_ft, sim.said))
        self.assertTrue(sim.said[1][1].startswith("The door is approximately"), sim.said)

    def test_walking_away_updates_too(self):
        sim = Sim()
        rng = random.Random(3)
        t = 0.0
        while t < 6.0:
            sim.step(t, door_box(1.0 + 0.4 * t, jitter=0.01, rng=rng))
            t += 1 / RATE
        nums = [spoken_feet(x) for _, x in sim.said if spoken_feet(x)]
        self.assertGreaterEqual(len(nums), 2, sim.said)
        self.assertEqual(nums, sorted(nums))

    def test_standing_still_with_jitter_stays_quiet(self):
        sim = Sim()
        rng = random.Random(4)
        for k in range(int(20 * RATE)):
            sim.step(k / RATE, door_box(5 * FT, jitter=0.02, rng=rng))
        self.assertEqual(len(sim.said), 1, sim.said)

    def test_sideways_at_constant_depth(self):
        """Stepping sideways 1 m while 6 ft from the door: at most one update, numbers stay near truth."""
        sim = Sim()
        rng = random.Random(5)
        for k in range(int(8 * RATE)):
            t = k / RATE
            lateral = -0.5 + t / 8.0
            sim.step(t, door_box(6 * FT, lateral=lateral, jitter=0.01, rng=rng))
        self.assertLessEqual(len(sim.said), 2, sim.said)
        for _, text in sim.said:
            n = spoken_feet(text)
            if n:
                self.assertIn(n, (5, 6, 7), sim.said)

    def test_glitch_box_is_ignored(self):
        """One wrong box (door merged with a cupboard, twice as wide) changes nothing."""
        sim = Sim()
        rng = random.Random(6)
        for k in range(int(10 * RATE)):
            box = door_box(6 * FT, jitter=0.01, rng=rng)
            if k == 30:
                box = (box[0] - (box[2] - box[0]) * 0.45, box[1], box[2] + (box[2] - box[0]) * 0.45, box[3])
            sim.step(k / RATE, box)
        self.assertEqual(len(sim.said), 1, sim.said)

    def test_unreliable_says_so(self):
        """A door seen only cut off at the side: no number is invented."""
        sim = Sim()
        for k in range(12):
            sim.step(k / RATE, door_box(6 * FT, lateral=1.4))
        self.assertEqual(len(sim.said), 1)
        self.assertIn("cannot estimate its distance reliably", sim.said[0][1])
        self.assertIsNone(spoken_feet(sim.said[0][1]))

    def test_very_close_door(self):
        sim = Sim()
        for k in range(4):
            sim.step(k / RATE, door_box(2 * FT))
        self.assertEqual(len(sim.said), 1)
        self.assertIn("very close", sim.said[0][1])
        self.assertIn("less than about 3 feet", sim.said[0][1])

    def test_short_loss_keeps_identity_and_stays_quiet(self):
        sim = Sim()
        rng = random.Random(7)
        t = 0.0
        for k in range(int(12 * RATE)):
            t = k / RATE
            hidden = 3.0 <= t < 4.2 or 6.0 <= t < 11.0     # 1.2 s, then 5 s out of view
            sim.step(t, None if hidden else door_box(7 * FT, jitter=0.01, rng=rng))
        self.assertEqual(len(sim.said), 1, sim.said)

    def test_updates_suppressed_when_not_allowed(self):
        """Find-door mode or quiet mode: only the first sighting is spoken."""
        sim = Sim(allow_updates=False)
        t = 0.0
        while t < 6:
            sim.step(t, door_box(3.0 - 0.35 * t))
            t += 1 / RATE
        self.assertEqual(len(sim.said), 1, sim.said)

    def test_no_update_right_after_careful_alert(self):
        sim = Sim()
        t = 0.0
        while t < 3:
            sim.step(t, door_box(3.0 - 0.5 * t))
            t += 1 / RATE
        oid = next(iter(sim.ranger.doors))
        sim.ranger.note_alert(oid, t)
        n = len(sim.said)
        end = t + dd.AFTER_ALERT_S - 0.3
        while t < end:
            sim.step(t, door_box(max(0.75, 3.0 - 0.5 * t)))
            t += 1 / RATE
        self.assertEqual(len(sim.said), n, sim.said)


class TestQuestions(unittest.TestCase):
    def test_visible_vs_remembered(self):
        sim = Sim()
        rng = random.Random(8)
        for k in range(12):
            sim.step(k / RATE, door_box(5 * FT, lateral=0.4, jitter=0.01, rng=rng))
        now = 12 / RATE
        ans = sim.ranger.where_answer(now, heading=0.0, frame_w=FW, f_px=F_PX)
        self.assertTrue(ans.startswith("The door is approximately"), ans)
        self.assertIn("on your right", ans)
        self.assertEqual(len(sim.ranger.overview_items(now, FW, F_PX)), 1)
        # gone from view (you turned 90 degrees to the right), 20 s later
        for k in range(12, 30):
            sim.step(k / RATE, None)
        later = now + 20
        ans = sim.ranger.where_answer(later, heading=90.0, frame_w=FW, f_px=F_PX)
        self.assertTrue(ans.startswith("I can't see the door right now. When I last saw it"), ans)
        self.assertIn("approximately 5 feet away", ans)
        self.assertIn("to your left", ans)
        self.assertEqual(sim.ranger.overview_items(later, FW, F_PX), [])

    def test_never_seen(self):
        self.assertIsNone(dd.DoorRanger().where_answer(1.0))

    def test_visible_but_unreliable(self):
        sim = Sim()
        for k in range(3):
            sim.step(k / RATE, door_box(6 * FT, lateral=1.4))
        ans = sim.ranger.where_answer(3 / RATE, frame_w=FW, f_px=F_PX)
        self.assertIn("cannot estimate its distance reliably", ans)


class TestRandomSessions(unittest.TestCase):
    def test_random_walks(self):
        """300 random sessions: random door width (0.75-1.0 m), start distance, speed,
        sideways drift, 2% jitter, 15% missed runs, occasional bad boxes. Every
        spoken number must be near the truth and updates must stay spaced."""
        rng = random.Random(42)
        worst = 0.0
        for _ in range(300):
            sim = Sim()
            width = rng.uniform(0.75, 1.0)
            depth, speed = rng.uniform(1.0, 4.0), rng.choice([0.0, 0.3, 0.5, -0.3])
            lateral, drift = rng.uniform(-0.6, 0.6), rng.uniform(-0.2, 0.2)
            truth = {}
            for k in range(int(10 * RATE)):
                t = k / RATE
                d = max(0.5, depth - speed * t)
                lat = lateral + drift * t
                truth[k] = math.hypot(d, lat)
                box = None if rng.random() < 0.15 else door_box(d, lat, width=width, jitter=0.02, rng=rng)
                if box and rng.random() < 0.03:     # a wrong box: much too wide or too narrow
                    w = box[2] - box[0]
                    f = rng.choice([0.5, 1.6])
                    box = (box[0], box[1], min(FW - 1.0, box[0] + w * f), box[3])
                sim.step(t, box)
            times = [tt for tt, x in sim.said]
            for a, b in zip(times[1:], times[2:]):
                self.assertGreaterEqual(b - a, dd.UPDATE_MIN_GAP_S - 1e-6, sim.said)
            for tt, text in sim.said:
                n = spoken_feet(text)
                if n is None:
                    continue
                true_ft = truth[round(tt * RATE)] / FT
                err = abs(n - true_ft) / true_ft
                worst = max(worst, err)
                self.assertLess(abs(n - true_ft), max(1.5, 0.35 * true_ft), (text, true_ft, width, sim.said))


class TestAccuracyTable(unittest.TestCase):
    def test_print_error_table(self):
        """Simulated accuracy (perfect camera model, 1% box jitter). Real-camera numbers
        come from tools/door_distance_test.py."""
        rows = []
        for ft in (2, 4, 6, 8):
            sim = Sim()
            rng = random.Random(ft)
            for k in range(int(3 * RATE)):
                sim.step(k / RATE, door_box(ft * FT, jitter=0.01, rng=rng))
            st = next(iter(sim.ranger.doors.values()))
            d = st.distance(3.0)
            rows.append((ft, None if d is None else d / FT, sim.said[0][1]))
        for ft, est, _ in rows:
            if est is not None:
                self.assertLess(abs(est - ft) / ft, 0.10, rows)
        self.assertIsNone(rows[0][1])        # 2 ft: wider than the view -> bound, not a number


if __name__ == "__main__":
    unittest.main()
