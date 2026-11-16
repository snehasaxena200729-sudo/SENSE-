"""Quick turns must still count as turns (heading_fusion.py).

A simulated room (procedural texture, seeded) is 'filmed' while turning 180
degrees at 12 fps; sense_master's own Lucas-Kanade step is reproduced with the
same parameters. Also: a person walking past a still camera is not a turn, and
the spoken "you have turned ..." sentence.
"""
import os
import sys
import unittest

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from heading_fusion import TurnEstimator   # noqa: E402
import spatial                              # noqa: E402

W, H, FOV = 1280, 720, 65.0
CIRC = int(360 / FOV * W)                   # pixels for a full circle
FEATURES = dict(maxCorners=150, qualityLevel=0.25, minDistance=12, blockSize=7)   # as sense_master.py
LK = dict(winSize=(31, 31), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 15, 0.03))


def make_room(seed=3):
    rng = np.random.default_rng(seed)
    room = np.full((H, CIRC, 3), 140, np.uint8)
    for _ in range(900):                    # furniture, frames, shelves: boxes of many sizes
        x, y = int(rng.integers(0, CIRC)), int(rng.integers(0, H))
        w, h = int(rng.integers(20, 220)), int(rng.integers(20, 220))
        cv2.rectangle(room, (x, y), (x + w, y + h), [int(c) for c in rng.integers(0, 255, 3)], -1)
    room = cv2.GaussianBlur(room, (3, 3), 0)
    return np.hstack([room, room[:, :W + 2]])


ROOM = make_room()


def view(x, blur_px):
    x = int(x) % CIRC
    f = ROOM[:, x:x + W]
    if blur_px > 1:                          # motion blur during the exposure
        k = np.full((1, int(blur_px)), 1.0 / int(blur_px), np.float32)
        f = cv2.filter2D(f, -1, k)
    return cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)


class LK1:
    """sense_master.py SUB-SYSTEM A, the LK part, unchanged."""
    def __init__(self):
        self.p0, self.old = None, None

    def step(self, g):
        mask = np.zeros_like(g)
        mask[: int(H * 0.45)] = 255
        dx, n = 0.0, 0
        if self.p0 is not None and len(self.p0) > 0:
            p1, st, _ = cv2.calcOpticalFlowPyrLK(self.old, g, self.p0, None, **LK)
            gn, go = p1[st == 1], self.p0[st == 1]
            if len(gn) >= 4:
                dx, n = float(np.median((gn - go)[:, 0])), len(gn)
                self.p0 = gn.reshape(-1, 1, 2)
                if len(gn) < 60:
                    fr = cv2.goodFeaturesToTrack(g, mask=mask, **FEATURES)
                    if fr is not None:
                        self.p0 = np.vstack((self.p0, fr))
            else:
                self.p0 = cv2.goodFeaturesToTrack(g, mask=mask, **FEATURES)
        else:
            self.p0 = cv2.goodFeaturesToTrack(g, mask=mask, **FEATURES)
        self.old = g
        return dx, n


def turn(deg, seconds, fps=12, fused=True):
    n = int(seconds * fps)
    total = deg / FOV * W
    xs = [total * (0.5 - 0.5 * np.cos(np.pi * i / n)) for i in range(n + 1)] + [total] * fps
    lk, te, heading = LK1(), TurnEstimator(), 0.0
    for i, x in enumerate(xs):
        g = view(x, abs(x - xs[i - 1]) * 0.5 if i else 0)
        dx, pts = lk.step(g)
        use = te.update(g, dx, pts) if fused else dx
        heading += te.add_turn(-use * FOV / W)
    return heading, te


class QuickTurnTests(unittest.TestCase):
    def test_slow_turn_unchanged(self):
        lk_only, _ = turn(180, 4, fused=False)
        fused, _ = turn(180, 4)
        self.assertAlmostEqual(lk_only, 180, delta=8)
        self.assertAlmostEqual(fused, lk_only, delta=3)

    def test_quick_turn_counted(self):
        lk_only, _ = turn(180, 1.2, fused=False)
        fused, _ = turn(180, 1.2)
        print(f"\n  180 deg in 1.2 s: LK alone {lk_only:.0f}, fused {fused:.0f}")
        self.assertAlmostEqual(fused, 180, delta=20)
        self.assertLessEqual(abs(fused - 180), abs(lk_only - 180) + 1)

    def test_left_turn_is_negative(self):
        fused, _ = turn(-120, 1.5)
        self.assertAlmostEqual(fused, -120, delta=15)

    def test_still_camera_person_walking_past(self):
        rng = np.random.default_rng(0)
        base = cv2.cvtColor(ROOM[:, 2000:2000 + W], cv2.COLOR_BGR2GRAY)
        lk, te, heading = LK1(), TurnEstimator(), 0.0
        for i in range(90):
            g = base.copy()
            x = 100 + i * 10
            g[40:700, x:x + 260] = 50            # a dark coat crossing the view
            g = np.clip(g + rng.normal(0, 2, g.shape), 0, 255).astype(np.uint8)
            dx, pts = lk.step(g)
            heading += te.add_turn(-te.update(g, dx, pts) * FOV / W)
        self.assertLess(abs(heading), 5)


class SinceSentenceTests(unittest.TestCase):
    def test_270_right_means_on_your_right(self):
        te = TurnEstimator()
        world = spatial.world_angle(W / 2, W, 0.0)   # Sneha straight ahead
        te.note_seen(["Sneha"])
        heading = 0.0
        for _ in range(27):                          # turn right 270 in 10 degree steps
            heading = spatial.wrap_angle(heading + te.add_turn(10.0))
        self.assertEqual(te.since_sentence("sneha", "Sneha"),
                         "You have turned about 270 degrees to your right since I last saw Sneha. ")
        rel = spatial.relative_angle(world, heading)
        self.assertEqual(spatial.direction_phrase(rel), "on your right")
        self.assertEqual(spatial.turn_instruction(rel), "turn about 90 degrees to your right")

    def test_180_is_behind(self):
        te = TurnEstimator()
        te.note_seen(["Sneha"])
        heading = spatial.wrap_angle(te.add_turn(-180.0))
        self.assertIn("180 degrees to your left", te.since_sentence("Sneha", "Sneha"))
        rel = spatial.relative_angle(0.0, heading)
        self.assertEqual(spatial.direction_phrase(rel), "behind you")
        self.assertEqual(spatial.turn_instruction(rel), "turn around")

    def test_small_or_unknown_turn_says_nothing(self):
        te = TurnEstimator()
        self.assertEqual(te.since_sentence("phone", "your phone"), "")
        te.note_seen(["Phone"])
        te.add_turn(15.0)
        self.assertEqual(te.since_sentence("phone", "your phone"), "")
        te.add_turn(25.0)
        self.assertEqual(te.since_sentence("phone", "your phone"),
                         "You have turned about 40 degrees to your right since I last saw your phone. ")
        te.note_seen(["Phone"])                       # seen again: counting restarts
        self.assertEqual(te.since_sentence("phone", "your phone"), "")


if __name__ == "__main__":
    unittest.main()
