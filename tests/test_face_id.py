"""Audit fix 4: face identity decisions.

The decision rules are tested with synthetic embeddings (no models needed).
If the real models are in models/ (they download on the first SENSE run),
an extra test checks them on the Ultralytics sample photo.
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import face_id  # noqa: E402
from people import UNKNOWN  # noqa: E402


def emb(seed, noise=0.0, base=None):
    rng = np.random.default_rng(seed)
    v = base if base is not None else rng.normal(size=128)
    return (v + noise * rng.normal(size=128)).astype(np.float32)


def face_row(score=0.95, eye=60.0, nose_offset=0.0):
    """A YuNet-style row: x, y, w, h, right eye, left eye, nose, mouth x2, score."""
    rx, ry, lx, ly = 100.0, 100.0, 100.0 + eye, 100.0
    nx = (rx + lx) / 2 + nose_offset * eye
    return np.array([80, 70, eye * 2, eye * 2.4, rx, ry, lx, ly, nx, 130, 110, 160, 150, 160, score], np.float32)


class TestDecide(unittest.TestCase):
    def setUp(self):
        self.sneha = emb(1)
        self.mummy = emb(2)
        self.profiles = {"Sneha": np.vstack([emb(10, 0.3, self.sneha) for _ in range(5)]),
                         "Mummy": np.vstack([emb(20, 0.3, self.mummy) for _ in range(5)])}

    def test_same_person_is_verified(self):
        name, best, _ = face_id.decide(emb(99, 0.3, self.sneha), self.profiles)
        self.assertEqual(name, "Sneha")
        self.assertGreaterEqual(best, face_id.MATCH_THRESHOLD)

    def test_stranger_is_unknown(self):
        name, best, _ = face_id.decide(emb(3), self.profiles)
        self.assertEqual(name, UNKNOWN)

    def test_ambiguous_between_two_people_is_unknown(self):
        a, b = self.sneha / np.linalg.norm(self.sneha), self.mummy / np.linalg.norm(self.mummy)
        profiles = {"Sneha": a[None, :], "Mummy": b[None, :]}
        name, best, second = face_id.decide(a + b, profiles)   # equally like both
        self.assertEqual(name, UNKNOWN)            # no clear margin

    def test_nobody_enrolled(self):
        self.assertEqual(face_id.decide(emb(1), {})[0], UNKNOWN)


class TestQuality(unittest.TestCase):
    def test_good_frontal_face(self):
        self.assertTrue(face_id.face_quality_ok(face_row()))

    def test_too_small(self):
        self.assertFalse(face_id.face_quality_ok(face_row(eye=15)))

    def test_turned_away(self):
        self.assertFalse(face_id.face_quality_ok(face_row(nose_offset=0.45)))

    def test_low_detector_score(self):
        self.assertFalse(face_id.face_quality_ok(face_row(score=0.6)))


class TestEnrolment(unittest.TestCase):
    def test_collects_and_checks_consistency(self):
        base = emb(5)
        e = face_id.Enrolment("Sneha", 3, 0.0)
        for i in range(5):
            e.add(emb(i, 0.3, base))
        self.assertTrue(e.finished(1.0) and e.ok() and e.consistent())

    def test_track_switch_during_enrolment_is_rejected(self):
        e = face_id.Enrolment("Sneha", 3, 0.0)
        for i in range(3):
            e.add(emb(i, 0.3, emb(5)))
        e.add(emb(77))                              # someone else
        self.assertFalse(e.consistent())

    def test_timeout_with_too_few_samples(self):
        e = face_id.Enrolment("Sneha", 3, 0.0)
        e.add(emb(1))
        self.assertTrue(e.finished(10.0))
        self.assertFalse(e.ok())


@unittest.skipUnless(os.path.exists(face_id.model_paths()["recognizer"]), "face models not downloaded")
class TestRealModels(unittest.TestCase):
    def test_sample_photo(self):
        import cv2
        try:
            import ultralytics
        except ImportError:
            self.skipTest("ultralytics not installed")
        img = cv2.imread(os.path.join(os.path.dirname(ultralytics.__file__), "assets", "zidane.jpg"))
        f = face_id.FaceID.create(download=False)
        box = (747, 41, 1140, 712)                     # the person on the right
        result, eye_px, e1 = f.check(img, box)
        self.assertEqual(result, UNKNOWN)              # clear face, nobody enrolled
        self.assertGreater(eye_px, 40)
        f.profiles = {"Test": e1[None, :]}
        darker = (img * 0.6).astype(np.uint8)
        self.assertEqual(f.check(darker, box)[0], "Test")   # same face, different light


if __name__ == "__main__":
    unittest.main()
