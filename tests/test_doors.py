"""Audit fix 3: door evidence rules (door box, weak door + handle, handle alone)."""
import os
import sys
import types
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hazards  # noqa: E402


class FakeModel:
    names = {0: "Person", 164: "Door", 165: "Door handle", 489: "Stairs"}

    def __init__(self, dets):
        self.dets = dets          # [(class id, conf, box)]

    def predict(self, frame, classes=None, conf=0.0, **kw):
        boxes = [types.SimpleNamespace(cls=[c], conf=[p], xyxy=[np.array(b)])
                 for c, p, b in self.dets if c in classes and p >= conf]
        return [types.SimpleNamespace(boxes=boxes)]


DOOR = (250, 40, 390, 470)
HANDLE_ON_DOOR = (360, 250, 380, 265)
HANDLE_ELSEWHERE = (40, 250, 60, 265)


def run(dets):
    det = hazards.SceneHazardDetector(FakeModel(dets))
    alerts = det.update(np.zeros((480, 640, 3), np.uint8), det.EVERY_N, 1.0)
    return det, alerts


class TestDoorEvidence(unittest.TestCase):
    def test_handle_class_is_requested(self):
        det = hazards.SceneHazardDetector(FakeModel([]))
        self.assertIn("door handle", det.class_ids.values())

    def test_confident_door_alone(self):
        det, alerts = run([(164, 0.40, DOOR)])
        self.assertEqual([b[0] for b in det.boxes], ["door"])

    def test_weak_door_alone_is_rejected(self):
        det, alerts = run([(164, 0.15, DOOR)])
        self.assertEqual(det.boxes, [])
        self.assertEqual(alerts, [])

    def test_weak_door_with_handle_inside_is_accepted(self):
        det, alerts = run([(164, 0.15, DOOR), (165, 0.45, HANDLE_ON_DOOR)])
        self.assertEqual([b[0] for b in det.boxes], ["door"])     # handle merged into the door
        self.assertTrue(any(k == "new:door" and t.startswith("Door in front of you") for k, t in alerts))

    def test_weak_door_with_handle_elsewhere_is_rejected(self):
        det, alerts = run([(164, 0.15, DOOR), (165, 0.45, HANDLE_ELSEWHERE)])
        self.assertEqual([b[0] for b in det.boxes], ["door handle"])

    def test_handle_alone_is_called_a_handle_not_a_door(self):
        det, alerts = run([(165, 0.45, HANDLE_ON_DOOR)])
        self.assertEqual([b[0] for b in det.boxes], ["door handle"])
        self.assertTrue(any(t.startswith("Door handle") for _, t in alerts))

    def test_weak_handle_is_ignored(self):
        det, alerts = run([(165, 0.20, HANDLE_ON_DOOR)])
        self.assertEqual(det.boxes, [])


if __name__ == "__main__":
    unittest.main()
