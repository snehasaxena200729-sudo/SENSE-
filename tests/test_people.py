"""Audit fix 1: no person name without verified identity.

    python -m unittest discover -s tests -v
"""
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from people import PersonIdentityGate, NO_FACE, UNKNOWN, count_phrase  # noqa: E402


class TestPersonIdentityGate(unittest.TestCase):
    def test_default_is_a_person(self):
        g = PersonIdentityGate()
        g.touch(7, 0.0)
        self.assertIsNone(g.label(7))
        self.assertEqual(g.spoken(7), "a person")

    def test_track_id_alone_never_gives_a_name(self):
        g = PersonIdentityGate()
        for t in range(100):               # tracked for a long time, no face checks
            g.touch(3, t * 0.03)
            g.observe(3, NO_FACE)
        self.assertEqual(g.spoken(3), "a person")

    def test_one_match_is_not_enough(self):
        g = PersonIdentityGate()
        g.observe(1, "Sneha")
        g.observe(1, "Sneha")
        self.assertIsNone(g.label(1))

    def test_three_agreeing_matches_lock_the_name(self):
        g = PersonIdentityGate()
        for _ in range(3):
            g.observe(1, "Sneha")
        self.assertEqual(g.spoken(1), "Sneha")

    def test_disagreement_blocks_the_lock(self):
        g = PersonIdentityGate()
        for r in ["Sneha", "Sneha", UNKNOWN, "Sneha"]:
            g.observe(1, r)
        self.assertIsNone(g.label(1))

    def test_turning_away_keeps_the_name(self):
        g = PersonIdentityGate()
        for _ in range(3):
            g.observe(1, "Sneha")
        for _ in range(20):
            g.observe(1, NO_FACE)
        self.assertEqual(g.label(1), "Sneha")

    def test_a_different_face_releases_the_name(self):
        g = PersonIdentityGate()
        for _ in range(3):
            g.observe(1, "Sneha")
        for _ in range(3):
            g.observe(1, UNKNOWN)          # e.g. ByteTrack swapped ids
        self.assertIsNone(g.label(1))

    def test_name_does_not_jump_to_a_new_track(self):
        g = PersonIdentityGate()
        g.touch(1, 0.0)
        for _ in range(3):
            g.observe(1, "Sneha")
        g.touch(2, 5.0)
        g.prune(5.0)
        self.assertIsNone(g.label(1))      # old track expired
        self.assertIsNone(g.label(2))      # new track must be verified again

    def test_rename_and_forget(self):
        g = PersonIdentityGate()
        for _ in range(3):
            g.observe(1, "Sneha")
        g.rename("sneha", "Didi")
        self.assertEqual(g.label(1), "Didi")
        g.forget_name("didi")
        self.assertIsNone(g.label(1))

    def test_count_phrase(self):
        self.assertEqual(count_phrase(1), "a person")
        self.assertEqual(count_phrase(3), "3 people")


class TestNoColourNaming(unittest.TestCase):
    """Guards against the old bug coming back: names from shirt colour."""

    def setUp(self):
        with open(os.path.join(ROOT, "sense_master.py"), encoding="utf-8") as f:
            self.src = f.read()

    def test_clothing_matcher_is_gone(self):
        self.assertNotIn("def match_person", self.src)
        self.assertNotIn("match_person(", self.src)

    def test_no_hard_coded_names(self):
        # the old code turned "snail" or "neha" into "Sneha" and biased Whisper towards it
        for literal in ('"Sneha"', '"Mummy"', "Sneha, Mummy"):
            self.assertNotIn(literal, self.src)

if __name__ == "__main__":
    unittest.main()
