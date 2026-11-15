"""Audit fix 2: distance geometry, cut-off boxes and honest wording.

The camera is simulated with the pinhole model, so the true distance of every
synthetic box is known. Real-world accuracy still has to be measured with
tools/validate_distance.py and a tape measure.
"""
import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import distance  # noqa: E402

W, H = 1280, 720
F = distance.focal_px(W)       # simulated camera matches the 65 degree assumption


def person_box(d, height=1.70, width=0.45, cam_height=1.0, f=F):
    """Box a standing person at distance d (m) makes in a W x H frame, clamped
    to the frame like sense_master.py does."""
    y_top = H / 2 - f * (height - cam_height) / d
    y_bot = H / 2 + f * cam_height / d
    half_w = f * width / d / 2
    return (max(0, W / 2 - half_w), max(0, y_top), min(W, W / 2 + half_w), min(H, y_bot))


class TestFocalLength(unittest.TestCase):
    def test_scales_with_resolution(self):
        self.assertAlmostEqual(distance.focal_px(1280), 2 * distance.focal_px(640))

    def test_old_constant_was_wrong_for_1280(self):
        self.assertAlmostEqual(distance.focal_px(1280), 1005, delta=3)   # not 600

    def test_calibration_overrides(self):
        self.assertAlmostEqual(distance.focal_px(1000, focal_ratio=0.9), 900)


class TestPersonDistance(unittest.TestCase):
    def test_truth_always_inside_range_or_below_bound(self):
        for d in [0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0]:
            est = distance.person_distance(person_box(d), W, H, F)
            self.assertTrue(est.known, d)
            if est.upper_bound_only:
                self.assertGreaterEqual(est.high, d, d)
            else:
                self.assertLessEqual(est.low, d, (d, est))
                self.assertGreaterEqual(est.high, d, (d, est))

    def test_the_five_feet_bug_is_gone(self):
        # Old formula said ~4.5-5 ft for everyone between 1.5 and 2.5 m.
        said = {d: distance.phrase(distance.person_distance(person_box(d), W, H, F))
                for d in (1.5, 2.0, 2.5)}
        self.assertEqual(len(set(said.values())), 3, said)

    def test_very_close_is_within_reach_or_bounded(self):
        est = distance.person_distance(person_box(0.5), W, H, F)
        self.assertLess(est.low if est.low else 0, 0.6)
        self.assertNotIn("about 5", distance.phrase(est))

    def test_cut_off_box_is_detected(self):
        self.assertIn("bottom", distance.clipped_edges(person_box(1.0), W, H))
        self.assertEqual(distance.clipped_edges(person_box(4.0), W, H), set())

    def test_eye_distance_cue(self):
        for d in (0.4, 0.8, 1.5):
            eye_px = F * 0.063 / d
            est = distance.person_distance(person_box(d), W, H, F, eye_px=eye_px)
            self.assertEqual(est.cue, "eyes")
            self.assertAlmostEqual(est.metres, d, places=6)
            self.assertLess(est.high - est.low, 0.5 * d)   # tighter than body cues

    def test_fully_cut_off_gives_only_a_bound(self):
        est = distance.person_distance((0, 0, W, H), W, H, F)
        self.assertTrue(est.upper_bound_only)
        self.assertIsNone(distance.feet_or_none(est))


class TestObjects(unittest.TestCase):
    def test_object_range_contains_truth(self):
        for name, (size, _) in distance.OBJECT_SIZES.items():
            d = max(1.2, F * size / (0.6 * W))     # far enough to fit in the frame
            px = F * size / d
            box = (100, 100, 100 + px, 100 + px * 0.5)
            est = distance.object_distance(name, box, W, H, F)
            self.assertLessEqual(est.low, d, name)
            self.assertGreaterEqual(est.high, d, name)

    def test_tilted_object_is_never_reported_further_than_it_is(self):
        d, size = 1.0, 0.15
        px = F * size / d * math.cos(math.radians(50))   # phone tilted away
        est = distance.object_distance("phone", (600, 300, 600 + px, 340), W, H, F)
        self.assertLessEqual(est.low, d)

    def test_object_cut_off_is_bound(self):
        est = distance.object_distance("laptop", (0, 300, 400, 600), W, H, F)
        self.assertTrue(est.upper_bound_only)

    def test_unknown_category(self):
        self.assertFalse(distance.object_distance("giraffe", (1, 1, 50, 50), W, H, F).known)


class TestWording(unittest.TestCase):
    def test_phrases(self):
        E = distance.Estimate
        self.assertEqual(distance.phrase(E()), "distance unclear")
        self.assertEqual(distance.phrase(E(0.5, 0.4, 0.6)), "within arm's reach")
        self.assertEqual(distance.phrase(E(6, 5, 7)), "far away")
        self.assertEqual(distance.phrase(E(1.5, 1.2, 1.8)), "about 4 to 6 feet away")
        self.assertEqual(distance.phrase(E(None, None, 2.0, upper_bound_only=True)),
                         "closer than about 7 feet")


if __name__ == "__main__":
    unittest.main()
