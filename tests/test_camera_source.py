"""camera_source.py: choosing built-in vs Camo by name, frame format, freeze
detection, losing and getting back the camera, switching. No real camera:
cv2.VideoCapture is replaced by a scripted fake."""
import json
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import camera_source as cs  # noqa: E402

# Unique IDs shaped like real ones. OpenCV sorts by uniqueID, so Camo can come
# BEFORE the built-in camera: index 0 is then the phone, not the Mac.
FACETIME = ("FaceTime HD Camera", "47B4B64B70674B9CAD2BAE273A71F4B5")
CAMO = ("Camo", "3F2E7A1C-CAMO-0000-0000-000000000001")


class FakeCap:
    """Scripted VideoCapture: `frames` is a list of frames / None (read fails)."""
    def __init__(self, frames, opened=True):
        self.frames = list(frames)
        self.opened = opened
        self.released = False
        self.props = {}

    def isOpened(self):
        return self.opened

    def set(self, k, v):
        self.props[k] = v
        return True

    def read(self):
        if not self.frames:
            return False, None
        f = self.frames.pop(0)
        return (f is not None), f

    def release(self):
        self.released = True


def noisy(h=720, w=1280, seed=0):
    return np.random.default_rng(seed).integers(0, 255, (h, w, 3), dtype=np.uint8)


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


class ParseAndList(unittest.TestCase):
    def test_parse_source(self):
        self.assertEqual(cs.parse_source("camo"), ("camo", None))
        self.assertEqual(cs.parse_source("Phone"), ("camo", None))
        self.assertEqual(cs.parse_source("builtin"), ("builtin", None))
        self.assertEqual(cs.parse_source("0"), ("index", 0))
        self.assertEqual(cs.parse_source("2"), ("index", 2))
        self.assertEqual(cs.parse_source("clips/desk.mp4"), ("file", "clips/desk.mp4"))

    def test_system_profiler_json(self):
        text = json.dumps({"SPCameraDataType": [
            {"_name": "FaceTime HD Camera", "spcamera_model-id": "FaceTime HD Camera",
             "spcamera_unique-id": FACETIME[1]},
            {"_name": "Camo", "spcamera_unique-id": CAMO[1]}]})
        self.assertEqual(cs._parse_system_profiler(text), [FACETIME, CAMO])

    def test_order_is_opencv_order_sorted_by_unique_id(self):
        # "3F2E..." < "47B4..." so Camo is index 0 and FaceTime index 1
        self.assertEqual(cs.order_like_opencv([FACETIME, CAMO]), [(0, "Camo"), (1, "FaceTime HD Camera")])

    def test_order_unknown_without_unique_ids(self):
        self.assertEqual(cs.order_like_opencv([("FaceTime HD Camera", None)]), [(None, "FaceTime HD Camera")])


class Resolve(unittest.TestCase):
    cams = [(0, "Camo"), (1, "FaceTime HD Camera")]

    def test_by_name(self):
        self.assertEqual(cs.resolve("camo", cameras=self.cams)[0], 0)
        self.assertEqual(cs.resolve("builtin", cameras=self.cams)[0], 1)

    def test_camo_missing_gives_clear_message(self):
        with self.assertRaises(cs.CameraNotFound) as ctx:
            cs.resolve("camo", cameras=[(0, "FaceTime HD Camera")])
        self.assertIn("Camo Studio", str(ctx.exception))
        self.assertIn("FaceTime HD Camera", str(ctx.exception))

    def test_verified_index_wins_only_if_name_still_matches(self):
        saved = {"index": 1, "name": "Camo"}
        # the camera at index 1 is not Camo any more -> use the name lookup
        self.assertEqual(cs.resolve("camo", saved, self.cams)[0], 0)
        cams2 = [(0, "FaceTime HD Camera"), (1, "Camo")]
        self.assertEqual(cs.resolve("camo", saved, cams2), (1, "verified earlier with tools/find_cameras.py ('Camo')"))

    def test_builtin_without_order_keeps_old_index_0(self):
        self.assertEqual(cs.resolve("builtin", cameras=[(None, "FaceTime HD Camera"), (None, "Camo")])[0], 0)
        self.assertEqual(cs.resolve("builtin", cameras=[])[0], 0)

    def test_camo_without_order_needs_find_cameras(self):
        with self.assertRaises(cs.CameraNotFound) as ctx:
            cs.resolve("camo", cameras=[(None, "FaceTime HD Camera"), (None, "Camo")])
        self.assertIn("find_cameras", str(ctx.exception))
        self.assertEqual(cs.resolve("camo", {"index": 2, "name": "Camo"},
                                    [(None, "FaceTime HD Camera"), (None, "Camo")])[0], 2)

    def test_no_names_at_all(self):
        self.assertEqual(cs.resolve("builtin", cameras=[])[0], 0)
        with self.assertRaises(cs.CameraNotFound):
            cs.resolve("camo", cameras=[])
        self.assertEqual(cs.resolve("camo", {"index": 1, "name": "Camo"}, [])[0], 1)


class Frames(unittest.TestCase):
    def test_normalize(self):
        self.assertIsNone(cs.normalize_frame(None, 1280))
        g = np.zeros((480, 640), np.uint8)
        self.assertEqual(cs.normalize_frame(g, 1280).shape, (480, 640, 3))
        bgra = np.zeros((480, 640, 4), np.uint8)
        self.assertEqual(cs.normalize_frame(bgra, 1280).shape, (480, 640, 3))
        big = np.zeros((1080, 1920, 3), np.uint8)
        out = cs.normalize_frame(big, 1280)
        self.assertEqual(out.shape, (720, 1280, 3))
        self.assertEqual(out.dtype, np.uint8)
        f32 = np.zeros((480, 640, 3), np.float32)
        self.assertEqual(cs.normalize_frame(f32, 1280).dtype, np.uint8)
        portrait = np.zeros((1920, 1080, 3), np.uint8)   # phone held upright in Camo
        self.assertEqual(cs.normalize_frame(portrait, 1280).shape, (1920, 1080, 3))

    def test_freeze(self):
        fd = cs.FreezeDetector()
        still = noisy(seed=1)
        t = 0.0
        for _ in range(40):                  # 4 s of the identical picture (Camo placeholder)
            frozen = fd.update(still, t)
            t += 0.1
        self.assertTrue(frozen)
        fd.reset()
        for i in range(40):                  # live camera: sensor noise every frame
            frame = np.clip(still.astype(np.int16) + np.random.default_rng(i).integers(-3, 4, still.shape), 0, 255)
            self.assertFalse(fd.update(frame.astype(np.uint8), i * 0.1))


class Source(unittest.TestCase):
    def make(self, spec, script, cams=None, saved=None):
        """script: {index: [FakeCap, FakeCap, ...]} handed out in order on each open."""
        self.clock = Clock()
        self.said, self.logs = [], []
        self.opened = []

        def opener(target):
            self.opened.append(target)
            caps = script.get(target, [])
            return caps.pop(0) if caps else FakeCap([], opened=False)
        src = cs.CameraSource(spec, saved=saved, log=self.logs.append, announce=self.said.append,
                              opener=opener, cameras=cams if cams is not None else [(0, "Camo"), (1, "FaceTime HD Camera")],
                              clock=self.clock)
        return src

    def test_file_ends_normally(self):
        src = self.make("clip.mp4", {"clip.mp4": [FakeCap([noisy(), noisy(seed=2)])]})
        self.assertEqual(src.open().shape, (720, 1280, 3))
        self.assertTrue(src.read()[0])
        self.assertEqual(src.read(), (False, None))
        self.assertTrue(src.is_file)

    def test_camo_opens_index_by_name_and_downscales(self):
        src = self.make("camo", {0: [FakeCap([noisy(1080, 1920)] * 3)]})
        self.assertEqual(src.open().shape, (720, 1280, 3))
        self.assertEqual(self.opened, [0])
        self.assertEqual(src.kind, "camo")

    def test_camera_warms_up_with_empty_frames(self):
        src = self.make("builtin", {1: [FakeCap([None] * 10 + [noisy()])]})
        self.assertEqual(src.open().shape, (720, 1280, 3))

    def test_camo_missing_falls_back_to_builtin_with_message(self):
        src = self.make("camo", {0: [FakeCap([noisy()])]}, cams=[(0, "FaceTime HD Camera")])
        src.open()
        self.assertEqual(src.kind, "builtin")
        self.assertTrue(any("phone camera" in s for s in self.said))
        self.assertTrue(any("Camo Studio" in l for l in self.logs))

    def test_builtin_missing_raises(self):
        src = self.make("builtin", {}, cams=[(0, "FaceTime HD Camera")])
        with self.assertRaises(cs.CameraNotFound):
            src.open()

    def test_disconnect_then_reconnect(self):
        first = FakeCap([noisy()] * 3 + [None])        # works, then the cable is pulled
        second = FakeCap([noisy(seed=3)] * 5)
        src = self.make("camo", {0: [first, FakeCap([], opened=False), second]})
        src.open()
        self.assertTrue(src.read()[0])
        self.assertTrue(src.read()[0])
        self.assertEqual(src.read(), (False, None))     # lost
        self.assertTrue(first.released)
        self.assertEqual(sum("Camera lost" in s for s in self.said), 1)
        self.assertEqual(src.read(), (False, None))     # too early to retry
        self.assertEqual(self.opened, [0])
        self.clock.t += 2.1
        self.assertEqual(src.read(), (False, None))     # retry 1: Camo not back yet
        self.clock.t += 2.1
        ok, frame = src.read()                          # retry 2: back
        self.assertTrue(ok)
        self.assertTrue(any("back" in s for s in self.said))
        self.assertEqual(src.kind, "camo")

    def test_camo_gone_for_good_falls_back_to_builtin(self):
        src = self.make("camo", {0: [FakeCap([noisy(), None])], 1: [FakeCap([noisy(seed=5)] * 5)]})
        src.open()
        src.read()
        self.assertEqual(src.read()[0], False)
        for _ in range(4):
            self.clock.t += 2.1
            src.read()
        self.assertEqual(src.kind, "builtin")
        self.assertTrue(any("laptop camera" in s for s in self.said))
        self.assertTrue(src.read()[0])

    def test_switch_and_failed_switch_keeps_old_camera(self):
        cam_b = FakeCap([noisy()] * 10)
        src = self.make("builtin", {1: [cam_b], 0: [FakeCap([], opened=False)]})
        src.open()
        msg = src.switch("camo")
        self.assertIn("can't use the phone camera", msg)
        self.assertEqual(src.kind, "builtin")
        self.assertFalse(cam_b.released)
        self.assertTrue(src.read()[0])

    def test_switch_resizes_and_flags_reset(self):
        src = self.make("builtin", {1: [FakeCap([noisy(480, 640)] * 5)], 0: [FakeCap([noisy(1080, 1920)] * 5)]})
        src.open()
        self.assertFalse(src.take_size_changed())
        self.assertEqual(src.switch("camo"), "Switched to the phone camera (Camo).")
        self.assertTrue(src.take_size_changed())
        self.assertFalse(src.take_size_changed())
        ok, frame = src.read()
        self.assertEqual(frame.shape, (720, 1280, 3))

    def test_frozen_picture_reported_once(self):
        still = noisy(seed=9)
        src = self.make("camo", {0: [FakeCap([still] * 80)]})
        src.open()
        for _ in range(60):
            self.clock.t += 0.1
            src.read()
        self.assertEqual(sum("frozen" in s for s in self.said), 1)


if __name__ == "__main__":
    unittest.main()
