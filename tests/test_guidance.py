""""Find my bottle" guidance (guidance.py) - unit and integration tests.

A simulated user stands in a room, turns towards a remembered item and walks
up to it. The camera sees the item only while it is inside the 65 degree view,
detections are noisy and sometimes missing, exactly like the live loop.
"""
import math
import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import answers  # noqa: E402
import guidance  # noqa: E402
import spatial  # noqa: E402
from guidance import FindGuide  # noqa: E402
from notifier import Notifier, P1, P2  # noqa: E402

FPS = 15.0
HALF_FOV = spatial.CAMERA_FOV_DEG / 2


class World:
    """User at the origin, item at (world angle, distance). Heading turns, user walks."""

    def __init__(self, item_angle, item_dist, seed=0, miss=0.2, angle_noise=1.5, dist_noise=0.08):
        self.item_angle, self.item_dist = item_angle, item_dist
        self.rng = random.Random(seed)
        self.miss, self.an, self.dn = miss, angle_noise, dist_noise

    def sighting(self, heading, dist):
        rel = spatial.wrap_angle(self.item_angle - heading)
        if abs(rel) > HALF_FOV - 2 or self.rng.random() < self.miss:
            return []
        return [{"angle": rel + self.rng.gauss(0, self.an),
                 "dist_m": dist * (1 + self.rng.uniform(-self.dn, self.dn)), "max_m": None}]


def run_user(guide, world, start_heading=0.0, turn_speed=30.0, walk_speed=0.5, max_s=60.0,
             can_speak_fn=lambda t: True, detector_every=3, first=None):
    """A cooperative user: turns towards the side SENSE names, walks when told
    'straight ahead'. Returns [(t, sentence)] of everything guidance said."""
    heading, dist, said = start_heading, world.item_dist, []
    instruction = None
    if first:
        low = first.lower()
        instruction = "walk" if "straight" in low else "left" if "left" in low else "right" if "right" in low else None
    for i in range(int(max_s * FPS)):
        t = i / FPS
        if i % detector_every == 0:
            guide.observe(t, world.sighting(heading, dist), heading)
        out = guide.poll(t, heading, can_speak=can_speak_fn(t))
        if out:
            said.append((t, out))
            low = out.lower()
            if "straight" in low:
                instruction = "walk"
            elif "left" in low:
                instruction = "left"
            elif "right" in low:
                instruction = "right"
        if not guide.active:
            break
        if instruction == "left":
            heading -= turn_speed / FPS
        elif instruction == "right":
            heading += turn_speed / FPS
        elif instruction == "walk":
            dist = max(0.3, dist - walk_speed / FPS)
            # walking straight but slightly off: the item drifts a little in the image
            heading += 0.4 * math.sin(i / 20.0) / FPS
        heading = spatial.wrap_angle(heading)
        if instruction in ("left", "right") and abs(spatial.wrap_angle(world.item_angle - heading)) < 5:
            instruction = "walk"       # user stops turning once it is ahead
    return said, dist


class TestDirectionWords(unittest.TestCase):
    def test_zones(self):
        self.assertEqual(guidance.direction_zone(0), "straight")
        self.assertEqual(guidance.direction_zone(-15), "little_left")
        self.assertEqual(guidance.direction_zone(18), "little_right")
        self.assertEqual(guidance.direction_zone(-30), "left")
        self.assertEqual(guidance.direction_zone(30), "right")

    def test_hysteresis_keeps_zone_near_border(self):
        z = "straight"
        for a in [9, 11, 12.5, 9.5, 13, 10.5]:      # jitter around the 10 deg border
            z = guidance.direction_zone(a, z)
            self.assertEqual(z, "straight", a)
        self.assertEqual(guidance.direction_zone(15, "straight"), "little_right")   # clearly past it
        z = "little_left"
        for a in [-9, -7, -12, -26, -28]:
            z = guidance.direction_zone(a, z)
            self.assertEqual(z, "little_left", a)

    def test_distance_bands(self):
        self.assertEqual(guidance.distance_band(0.6), "reach")
        self.assertEqual(guidance.distance_band(1.2), "close")
        self.assertEqual(guidance.distance_band(2.5), "steps")
        self.assertEqual(guidance.distance_band(5.0), "far")
        self.assertIsNone(guidance.distance_band(None))
        self.assertEqual(guidance.distance_band(None, max_m=0.9), "reach")   # cut-off box, upper bound only
        self.assertIsNone(guidance.distance_band(None, max_m=3.0))


class TestIntents(unittest.TestCase):
    def test_find_and_stop(self):
        for text in ["find my bottle", "Find my water bottle.", "help me find my phone",
                     "guide me to the door", "take me to the chair", "look for my keys", "locate the cup"]:
            self.assertEqual(answers.classify_intent(text), "FIND", text)
        for text in ["stop looking", "stop searching", "cancel", "never mind", "I found it"]:
            self.assertEqual(answers.classify_intent(text), "STOP_FIND", text)

    def test_existing_intents_unchanged(self):
        for text, intent in [("where is my bottle", "WHERE"), ("this is sneha", "LEARN"),
                             ("what do you see", "OVERVIEW"), ("stop talking", "QUIET"),
                             ("forget my bottle", "FORGET"), ("what changed", "CHANGES"),
                             ("rename my bottle to medicine", "RENAME"), ("who was here", "WHO")]:
            self.assertEqual(answers.classify_intent(text), intent, text)


class TestGuidance(unittest.TestCase):
    def assert_rate_limited(self, said):
        for (t0, a), (t1, b) in zip(said, said[1:]):
            gap = guidance.REACH_GAP_S if "within reach" in b else guidance.MIN_GAP_S
            self.assertGreaterEqual(t1 - t0, gap - 1e-6, said)
            if a == b and "Keep turning" not in a:
                self.assertGreaterEqual(t1 - t0, guidance.REPEAT_S - 1e-6, f"repeated too soon: {said}")

    def test_visible_at_start_then_walk_to_reach(self):
        for seed in range(20):
            g = FindGuide()
            w = World(item_angle=-18, item_dist=3.5, seed=seed)
            first = g.start_visible("your bottle", 0.0, {"angle": -18 + w.rng.gauss(0, 1.5),
                                                         "dist_m": 3.5, "max_m": None}, 0.0)
            self.assertIn("your bottle", first)
            self.assertIn("left", first)
            said, dist = run_user(g, w, first=first)
            self.assertFalse(g.active, f"seed {seed}: never reached: {said}")
            self.assertIn("within reach", said[-1][1])
            self.assertEqual(sum("within reach" in s for _, s in said), 1)
            self.assertLessEqual(dist, 1.0, f"seed {seed}: said 'within reach' at {dist:.2f} m")
            self.assert_rate_limited(said)
            self.assertLess(len(said), 25, said)

    def test_memory_fallback_turn_then_find(self):
        """Not in view: SENSE uses the remembered world angle to turn the user."""
        for seed in range(20):
            g = FindGuide()
            w = World(item_angle=-90, item_dist=2.0, seed=seed)
            rel = spatial.relative_angle(-90, 0.0)
            first = g.start("your bottle", 0.0, memory={
                "world_angle": -90.0, "age_text": "2 minutes ago", "where": spatial.direction_phrase(rel),
                "relation": "next to the laptop", "turn": spatial.turn_instruction(rel)})
            self.assertIn("I last saw it 2 minutes ago, on your left, next to the laptop", first)
            self.assertIn("Turn about 90 degrees to your left", first)

            # user turns left slowly at first, guided only by the first sentence
            heading, said = 0.0, []
            for i in range(int(60 * FPS)):
                t = i / FPS
                if i % 3 == 0:
                    g.observe(t, w.sighting(heading, 2.0), heading)
                out = g.poll(t, heading)
                if out:
                    said.append((t, out))
                if g.state == guidance.GUIDING:
                    break
                heading = spatial.wrap_angle(heading - 12.0 / FPS)
            self.assertEqual(g.state, guidance.GUIDING, said)
            self.assertTrue(all("right" not in s for _, s in said), f"wrong side: {said}")
            # then the normal guidance takes over and gets the user there
            rest, dist = run_user(g, w, start_heading=heading)
            self.assertTrue(rest and rest[0][1].startswith("Found your bottle"), rest)
            self.assertIn("within reach", rest[-1][1])
            self.assert_rate_limited(said + rest)

    def test_keep_turning_updates_with_heading(self):
        g = FindGuide()
        g.start("your cup", 0.0, memory={"world_angle": 120.0, "age_text": "a minute ago", "where": "on your right",
                                         "relation": None, "turn": "turn about 120 degrees to your right"})
        out = g.poll(guidance.SEARCH_REPEAT_S + 0.1, heading=40.0)
        self.assertEqual(out, "Keep turning right.")
        # overshoot: now it is on the LEFT -> the words must change side
        out = g.poll(2 * guidance.SEARCH_REPEAT_S + 0.2, heading=170.0)
        self.assertIn("left", out)
        self.assertNotIn("right", out)

    def test_should_be_here_but_not_seen(self):
        g = FindGuide()
        g.start("your bottle", 0.0, memory={"world_angle": 5.0, "age_text": "a minute ago", "where": "in front of you",
                                            "relation": None, "turn": None})
        out = g.poll(guidance.SEARCH_REPEAT_S + 0.1, heading=0.0)
        self.assertIn("should be in front of you, but I can't see it", out)
        out = g.poll(2 * guidance.SEARCH_REPEAT_S + 0.2, heading=0.0)
        self.assertEqual(out, "Keep turning slowly.")     # no repeat of the long sentence

    def test_no_memory(self):
        g = FindGuide()
        out = g.start("your keys", 0.0)
        self.assertIn("Turn slowly in a circle", out)
        self.assertIsNone(g.poll(1.0, 0.0))
        self.assertEqual(g.poll(guidance.SEARCH_REPEAT_S + 0.1, 0.0), "Keep turning slowly.")

    def test_search_timeout_gives_up_once(self):
        g = FindGuide()
        g.start("your keys", 0.0)
        msgs = [m for m in (g.poll(i / FPS, 0.0) for i in range(int((guidance.SEARCH_TIMEOUT_S + 5) * FPS))) if m]
        self.assertIn("I couldn't find your keys", msgs[-1])
        self.assertFalse(g.active)
        self.assertIsNone(g.poll(100.0, 0.0))

    def test_lost_then_reacquired(self):
        g = FindGuide()
        g.start_visible("your bottle", 0.0, {"angle": -20, "dist_m": 2.5, "max_m": None}, 0.0)
        for i in range(1, 15):     # visible ~1 s
            g.observe(i / FPS, [{"angle": -20, "dist_m": 2.5, "max_m": None}], 0.0)
            g.poll(i / FPS, 0.0)
        # user turned right by 60 deg and it is out of view for 3 s
        said = [m for m in (g.poll(1.0 + i / FPS, 60.0) for i in range(45)) if m]
        self.assertTrue(any(m.startswith("I lost sight of your bottle. It was on your left") for m in said), said)
        self.assertEqual(g.state, guidance.SEARCHING)
        # back in view
        for i in range(6):
            g.observe(4.0 + i / FPS, [{"angle": -5, "dist_m": 2.4, "max_m": None}], -15.0)
        out = g.poll(4.5, -15.0)
        self.assertTrue(out.startswith("Found your bottle, straight ahead"), out)

    def test_single_outlier_does_not_change_cue(self):
        g = FindGuide()
        g.start_visible("your bottle", 0.0, {"angle": 0, "dist_m": 2.5, "max_m": None}, 0.0)
        said = []
        for i in range(1, 200):
            t = i / FPS
            a = 40.0 if i == 100 else random.Random(i).gauss(0, 2)    # one bad box at frame 100
            g.observe(t, [{"angle": a, "dist_m": 2.5, "max_m": None}], 0.0)
            out = g.poll(t, 0.0)
            if out:
                said.append(out)
        self.assertFalse(any("right" in s or "left" in s for s in said), said)

    def test_picks_instance_near_remembered_spot(self):
        g = FindGuide()
        g.start("the bottle", 0.0, memory={"world_angle": 20.0, "age_text": "just now", "where": "a little right",
                                           "relation": None, "turn": None})
        g.observe(0.1, [{"angle": -25, "dist_m": 2.0, "max_m": None},
                        {"angle": 19, "dist_m": 2.0, "max_m": None}], 0.0)
        self.assertAlmostEqual(g.world_angle, 19.0)

    def test_unknown_distance_never_says_reach(self):
        g = FindGuide()
        g.start_visible("the chair", 0.0, {"angle": 0, "dist_m": None, "max_m": None}, 0.0)
        for i in range(1, 300):
            g.observe(i / FPS, [{"angle": 0, "dist_m": None, "max_m": None}], 0.0)
            out = g.poll(i / FPS, 0.0)
            self.assertFalse(out and "reach" in out, out)

    def test_reach_needs_two_readings_and_roughly_ahead(self):
        g = FindGuide()
        g.start_visible("your cup", 0.0, {"angle": 0, "dist_m": 1.2, "max_m": None}, 0.0)
        g.observe(0.2, [{"angle": 2, "dist_m": 0.6, "max_m": None}], 0.0)
        self.assertNotIn("reach", g.poll(0.2, 0.0) or "")
        g.observe(0.4, [{"angle": 35, "dist_m": 0.6, "max_m": None}], 0.0)      # off to the side
        g.observe(0.6, [{"angle": 35, "dist_m": 0.6, "max_m": None}], 0.0)
        self.assertNotIn("reach", g.poll(2.0, 0.0) or "")
        g.observe(2.2, [{"angle": 3, "dist_m": 0.6, "max_m": None}], 0.0)
        g.observe(2.4, [{"angle": 3, "dist_m": 0.6, "max_m": None}], 0.0)
        g.observe(2.6, [{"angle": 3, "dist_m": 0.6, "max_m": None}], 0.0)
        out = g.poll(4.0, 0.0)
        self.assertIn("within reach, straight ahead", out)
        self.assertFalse(g.active)

    def test_big_object_filling_the_view(self):
        """A bed or door close up only has an upper-bound distance; filling the
        view from side to side ends the guidance with 'right in front of you'."""
        g = FindGuide()
        g.start_visible("the bed", 0.0, {"angle": 0, "dist_m": 2.5, "max_m": None}, 0.0)
        out = None
        for i in range(1, 60):
            g.observe(i / FPS, [{"angle": 1, "dist_m": None, "max_m": 1.6, "filled": i > 30}], 0.0)
            out = g.poll(i / FPS, 0.0) or out
            if not g.active:
                break
        self.assertFalse(g.active)
        self.assertEqual(out, "The bed is right in front of you. Slow down.")

    def test_stop_by_user(self):
        g = FindGuide()
        g.start("your bottle", 0.0)
        self.assertEqual(g.stop("user"), "Okay, I stopped looking for your bottle.")
        self.assertFalse(g.active)
        self.assertIsNone(g.poll(10.0, 0.0))

    def test_total_timeout(self):
        g = FindGuide()
        g.start_visible("your bottle", 0.0, {"angle": -40, "dist_m": 3.0, "max_m": None}, 0.0)
        out = None
        for i in range(int((guidance.TOTAL_TIMEOUT_S + 2) * FPS)):     # seen but never reached
            t = i / FPS
            g.observe(t, [{"angle": -40, "dist_m": 3.0, "max_m": None}], 0.0)
            out = g.poll(t, 0.0) or out
            if not g.active:
                break
        self.assertFalse(g.active)
        self.assertIn("stopping the search", out)


class TestHazardPriority(unittest.TestCase):
    """Integration: guidance + notifier, gated exactly like the main loop."""

    def test_guidance_yields_to_safety_alerts(self):
        clock = {"t": 0.0}
        log = []
        n = Notifier(lambda s: log.append((clock["t"], "notifier", s)), lambda: False, clock=lambda: clock["t"])
        g = FindGuide()
        g.start_visible("your bottle", 0.0, {"angle": -30, "dist_m": 3.0, "max_m": None}, 0.0)
        rng = random.Random(1)
        for i in range(int(20 * FPS)):
            t = clock["t"] = i / FPS
            if i % 3 == 0:
                g.observe(t, [{"angle": -30 + rng.gauss(0, 1), "dist_m": 3.0, "max_m": None}], 0.0)
            if i in (60, 61, 62, 150):                        # chair alerts at 4 s and 10 s
                n.push(P1, "near:coco:7", "Careful, a chair in front of you, about 4 feet away")
            if i == 30:
                n.push(P2, "my phone", "I see your phone on your right")      # non-urgent news
            spoken = n.tick(camera_moving=True, hold=g.active)
            if g.active:
                cue = g.poll(t, 0.0, can_speak=spoken is None and n.safety_clear())
                if cue:
                    log.append((t, "guide", cue))
        p1_times = [t for t, who, s in log if who == "notifier" and s.startswith("Careful")]
        self.assertEqual(len(p1_times), 2, log)                # P1 still spoken, still de-duplicated
        for t, who, s in log:
            if who == "guide":
                for tp in p1_times:
                    self.assertFalse(0 <= t - tp < 2.0, f"guide cue {s!r} at {t} crowds alert at {tp}")
        self.assertFalse(any("phone" in s for _, _, s in log), "non-urgent news should wait during guidance")
        self.assertTrue(any(who == "guide" for _, who, _ in log))

    def test_safety_clear(self):
        clock = {"t": 0.0}
        n = Notifier(lambda s: None, lambda: False, clock=lambda: clock["t"])
        self.assertTrue(n.safety_clear())
        n.push(P1, "x", "Careful")
        self.assertFalse(n.safety_clear())      # waiting
        n.tick()
        self.assertFalse(n.safety_clear())      # just spoken
        clock["t"] = 2.5
        self.assertTrue(n.safety_clear())
        n.push(P2, "y", "I see your cup")
        self.assertTrue(n.safety_clear())       # only P1 matters


if __name__ == "__main__":
    unittest.main()
