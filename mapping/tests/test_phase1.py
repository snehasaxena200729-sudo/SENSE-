"""Phase 1 tests: fingerprints, teaching by turning round, recognition, and the
voice commands that use them.

    python -m unittest discover -s mapping/tests -t .

The image-model tests need models/image_classification_mobilenetv2_2022apr.onnx
(downloaded on SENSE's first run); they are skipped without it.
"""

import os
import sqlite3
import unittest

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

import mapping
from mapping import fingerprint as F
from mapping.config import MappingConfig
from mapping.recognise import UNKNOWN, Recogniser, RoomModel
from mapping.teaching import TeachSession
from mapping.tests.test_mapping import ROOT, table_digest

MODEL = os.path.join(ROOT, "models", F.MODEL_FILE)
W, H = 640, 360
FULL = int(640 * 360 / 65)


def room_strip(seed):
    """A wide textured 'room' (posters, furniture blocks) covering 360 degrees."""
    rng = np.random.default_rng(seed)
    base = tuple(int(v) for v in rng.integers(90, 220, 3))
    img = np.full((H, FULL + W, 3), base, np.uint8)
    for _ in range(500):
        x, y = int(rng.integers(0, FULL)), int(rng.integers(0, H))
        w, h = int(rng.integers(10, 120)), int(rng.integers(10, 120))
        c = tuple(int(v) for v in rng.integers(0, 255, 3))
        if rng.random() < 0.5:
            cv2.rectangle(img, (x, y), (x + w, y + h), c, -1)
        else:
            cv2.circle(img, (x, y), w // 2, c, -1)
    img[:, FULL:] = img[:, :W]                       # wrap round
    return img


def frame_at(strip, deg, gamma=1.0):
    off = int((deg % 360) / 360.0 * FULL)
    f = strip[:, off:off + W]
    if gamma != 1.0:
        f = np.clip(255 * (f / 255.0) ** gamma, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(f)


class TinyEmbedder:
    """Stands in for MobileNet in tests: a 12x7 colour thumbnail, unit length."""

    def embed(self, bgr):
        v = cv2.resize(bgr, (12, 7), interpolation=cv2.INTER_AREA).astype(np.float32).ravel() - 128.0
        return v / (np.linalg.norm(v) + 1e-9)


class Clock:
    def __init__(self, t=1_000_000.0):
        self.t = t

    def __call__(self):
        return self.t


@unittest.skipIf(cv2 is None, "OpenCV not installed")
class FingerprintTests(unittest.TestCase):
    def test_pack_unpack_numbers_only(self):
        emb = np.random.default_rng(0).normal(size=1280).astype(np.float32)
        pts, des = F.orb_features(frame_at(room_strip(1), 30))
        blob = F.pack(emb, pts, des)
        e2, p2, d2 = F.unpack(blob)
        self.assertTrue(np.allclose(e2, emb, atol=1e-2))
        self.assertTrue(np.array_equal(d2, des) and np.allclose(p2, pts))
        with np.load(__import__("io").BytesIO(blob), allow_pickle=False) as z:
            self.assertEqual(sorted(z.files), ["des", "emb", "pts"])        # no image stored
        self.assertLess(len(blob), 30_000)

    def test_no_embedding(self):
        blob = F.pack(None, np.zeros((0, 2), np.float32), np.zeros((0, 32), np.uint8))
        self.assertIsNone(F.unpack(blob)[0])

    def test_graph_output_patch_keeps_other_bytes(self):
        fake = b"\x08\x07" + b"\x3a\x03\x0a\x01x" + b"\x12\x02ab"     # ir_version, graph{name}, producer
        out = F.add_graph_output(fake, "464")
        self.assertTrue(out.startswith(b"\x08\x07"))
        self.assertIn(b"\x62\x05\x0a\x03464", out)
        self.assertTrue(out.endswith(b"\x12\x02ab"))

    def test_sharpness_drops_with_blur(self):
        f = frame_at(room_strip(2), 0)
        self.assertGreater(F.sharpness(f), F.sharpness(cv2.GaussianBlur(f, (21, 21), 0)) * 3)

    @unittest.skipUnless(os.path.exists(MODEL), "room image model not downloaded")
    def test_real_model(self):
        e = F.Embedder.create(download=False)
        self.assertIsNotNone(e)
        a, b = frame_at(room_strip(3), 0), frame_at(room_strip(3), 8)
        c = frame_at(room_strip(9), 0)
        va, vb, vc = e.embed(a), e.embed(b), e.embed(c)
        self.assertEqual(va.shape, (1280,))
        self.assertAlmostEqual(float(np.linalg.norm(va)), 1.0, places=4)
        self.assertGreater(float(va @ vb), float(va @ vc))


class TeachingTests(unittest.TestCase):
    cfg = MappingConfig()

    def run_turn(self, deg_per_s, seconds, fps=15, sharp=50.0, start=0.0, stop_at=None):
        s = TeachSession(1, "kitchen", start, 0.0, self.cfg)
        events = []
        for k in range(int(seconds * fps)):
            t = (k + 1) / fps
            h = ((start + deg_per_s * t) + 180) % 360 - 180
            ev = s.feed(h, t, sharp, lambda: {"emb": None, "pts": None, "des": None, "landmarks": []})
            if ev:
                events.append((round(t, 1), ev[0]))
                if ev[0] in ("done", "failed"):
                    break
        return s, events

    def test_full_slow_turn(self):
        s, ev = self.run_turn(20, 30)
        self.assertEqual(ev[-1][1], "done")
        self.assertIn("progress", [e[1] for e in ev])
        self.assertGreaterEqual(len(s.views), 16)
        self.assertLessEqual(len(s.views), 18)
        headings = sorted(v["heading_deg"] for v in s.views)
        self.assertTrue(all(0 <= h < 360 for h in headings))
        self.assertGreaterEqual(min(np.diff(headings)), 5)          # one view per direction

    def test_turn_left_also_works(self):
        s, ev = self.run_turn(-25, 30, start=170)
        self.assertEqual(ev[-1][1], "done")

    def test_no_turn_nudges_then_fails(self):
        s, ev = self.run_turn(0, 60)
        self.assertEqual([e[1] for e in ev], ["nudge", "failed"])
        self.assertAlmostEqual(ev[0][0], 6.0, delta=0.11)

    def test_blurred_frames_are_not_saved(self):
        s, ev = self.run_turn(20, 50, sharp=1.0)
        self.assertEqual(len(s.views), 0)
        self.assertEqual(ev[-1][1], "failed")

    def test_fast_turn_still_one_view_per_bin(self):
        s, ev = self.run_turn(120, 10)
        self.assertLessEqual(len(s.views), 18)

    def test_stop_early(self):
        s, _ = self.run_turn(20, 1.2)
        self.assertEqual(s.stop()[0], "failed" if len(s.views) < 3 else "done")
        s, _ = self.run_turn(20, 6)
        self.assertEqual(s.stop()[0], "done")


class RecogniserTests(unittest.TestCase):
    """Synthetic embeddings: each room = a direction in 64-d space + noise."""

    def setUp(self):
        self.rng = np.random.default_rng(0)
        self.protos = [self.unit(self.rng.normal(size=64)) for _ in range(4)]
        self.cfg = MappingConfig()
        self.rec = Recogniser(self.cfg)
        models = []
        for i, p in enumerate(self.protos[:3]):
            views = [{"emb": self.sample(i), "pts": None, "des": None, "landmarks": []} for _ in range(18)]
            models.append(RoomModel(i, f"room{i}", views))
        self.rec.set_rooms(models)

    @staticmethod
    def unit(v):
        v = np.asarray(v, np.float32)
        return v / np.linalg.norm(v)

    def sample(self, room, noise=0.35):
        return self.unit(self.protos[room] + noise * self.unit(self.rng.normal(size=64)))

    def check(self, room, n=1, t0=0):
        for k in range(n):
            emb = self.sample(room) if room is not None else self.unit(self.rng.normal(size=64))
            self.rec.update(emb, None, None, set(), t0 + k)
        return self.rec.best()

    def test_recognises_each_room(self):
        for r in range(3):
            self.rec.reset()
            state, p = self.check(r, 4)
            self.assertEqual(state, r)
            self.assertGreaterEqual(p, self.cfg.sure_prob)
            self.assertTrue(self.rec.verified(r))

    def test_untaught_place(self):
        state, p = self.check(3, 5)                   # room 3 was never taught
        self.assertEqual(state, UNKNOWN)

    def test_one_odd_frame_does_not_flip(self):
        self.check(0, 5)
        self.rec.update(self.sample(1), None, None, set(), 10)
        self.assertEqual(self.rec.best()[0], 0)

    def test_moving_rooms_switches_within_a_few_checks(self):
        self.check(0, 5)
        for k in range(4):
            self.rec.update(self.sample(2), None, None, set(), 20 + k)
        self.assertEqual(self.rec.best()[0], 2)

    def test_landmarks_shift_odds_but_never_decide(self):
        m = self.rec.rooms[1]
        m.landmark_counts = {"refrigerator": 15}
        f = self.rec.landmark_factor(1, {"refrigerator"})
        self.assertGreater(f, 1.0)
        self.assertLessEqual(f, self.cfg.landmark_clip[1])
        self.assertGreaterEqual(self.rec.landmark_factor(0, {"refrigerator"}), self.cfg.landmark_clip[0])

    def test_no_evidence_no_change(self):
        self.check(0, 4)
        before = dict(self.rec.belief)
        self.rec.update(None, np.zeros((0, 2), np.float32), np.zeros((0, 32), np.uint8), set(), 50)
        self.assertEqual(self.rec.best()[0], 0)
        self.assertGreater(self.rec.belief[0], before[0] * 0.8)


@unittest.skipIf(cv2 is None, "OpenCV not installed")
class HomeMapPhase1Tests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.conn = sqlite3.connect(":memory:")
        self.home = mapping.HomeMap(self.conn, "run1", clock=self.clock, embedder=TinyEmbedder())
        self.strips = {"kitchen": room_strip(11), "bedroom": room_strip(22), "elsewhere": room_strip(33)}

    def say(self, text):
        return self.home.handle(text, self.clock(), heading=self.heading)

    heading = 0.0

    def turn(self, strip, seconds=26, deg_per_s=15, fps=15):
        """Feed frames of a slow full turn; return everything SENSE said."""
        said = []
        for k in range(int(seconds * fps)):
            self.clock.t += 1.0 / fps
            self.heading = ((self.heading + deg_per_s / fps) + 180) % 360 - 180
            msg = self.home.observe(frame_at(strip, self.heading), [("refrigerator", 0.8)], None,
                                    self.heading, camera_stable=False, now=self.clock())
            if msg:
                said.append(msg)
        return said

    def stand(self, strip, deg, seconds=3, fps=15, gamma=1.0):
        said = []
        for k in range(int(seconds * fps)):
            self.clock.t += 1.0 / fps
            msg = self.home.observe(frame_at(strip, deg, gamma), None, None, deg, True, self.clock())
            if msg:
                said.append(msg)
        return said

    def teach(self, name):
        self.assertIn("turn slowly all the way round", self.say(f"this room is the {name}"))
        said = self.turn(self.strips[name])
        self.assertTrue(any(s.startswith(f"Done. I've learned the {name}") for s in said), said)

    def test_teach_then_recognise_and_restart(self):
        self.teach("kitchen")
        self.teach("bedroom")
        rows = self.conn.execute("SELECT COUNT(*) FROM room_photos").fetchone()[0]
        self.assertGreaterEqual(rows, 32)
        self.stand(self.strips["kitchen"], 95, gamma=1.3)
        self.assertEqual(self.say("where am i"), "You are in the kitchen.")
        self.stand(self.strips["bedroom"], 250)
        self.assertEqual(self.say("where am i"), "You are in the bedroom.")
        # restart: a new run on the same database
        home2 = mapping.HomeMap(self.conn, "run2", clock=self.clock, embedder=TinyEmbedder())
        self.home = home2
        self.stand(self.strips["kitchen"], 200)
        self.assertEqual(self.say("where am i"), "You are in the kitchen.")
        self.stand(self.strips["elsewhere"], 10, seconds=4)
        self.assertFalse(self.say("where am i").startswith("You are in"))

    def test_landmarks_saved_with_views(self):
        self.teach("kitchen")
        lm = self.conn.execute("SELECT landmarks FROM room_photos LIMIT 1").fetchone()[0]
        self.assertEqual(lm, "refrigerator")

    def test_teach_known_room_just_declares(self):
        self.teach("kitchen")
        self.assertEqual(self.say("this room is the kitchen"), "Okay, you are in the kitchen.")

    def test_cancel_and_stop(self):
        self.say("this room is the kitchen")
        self.assertIn("Nothing new was saved", self.say("cancel"))
        self.assertIsNone(self.say("cancel"))           # not teaching: not ours any more
        self.assertIsNone(self.say("stop"))
        self.say("this room is the kitchen")
        self.turn(self.strips["kitchen"], seconds=6)
        self.assertTrue(self.say("stop").startswith("Done. I've learned the kitchen"))

    def test_relearn_replaces_only_when_finished(self):
        self.teach("kitchen")
        before = self.conn.execute("SELECT COUNT(*) FROM room_photos").fetchone()[0]
        self.assertIn("relearn the kitchen", self.say("relearn the kitchen"))
        self.turn(self.strips["kitchen"], seconds=3)
        self.say("cancel")
        self.assertEqual(self.conn.execute("SELECT COUNT(*) FROM room_photos").fetchone()[0], before)
        self.stand(self.strips["kitchen"], 40)
        self.assertIn("relearn the kitchen", self.say("relearn this room"))
        self.turn(self.strips["kitchen"])
        after = self.conn.execute("SELECT COUNT(*) FROM room_photos").fetchone()[0]
        self.assertTrue(16 <= after <= 18)

    def test_while_teaching(self):
        self.say("this room is the kitchen")
        self.assertIn("still learning the kitchen", self.say("where am i"))
        self.assertIn("1 room", self.say("which rooms do you know"))
        self.assertIsNone(self.say("what do you see"))   # existing commands still pass through

    def test_relearn_without_room(self):
        self.assertEqual(self.say("relearn this room"), "Which room is this? Say: this room is the kitchen.")
        self.assertIn("don't know the study", self.say("relearn the study"))

    def test_existing_tables_untouched_by_teaching(self):
        import memory_db
        conn = sqlite3.connect(":memory:")
        old_get = memory_db._get_conn
        memory_db._conn = conn
        try:
            memory_db.init_database()
            memory_db.save_object_observation("my bottle", "on your left", 10, 20, status="VISIBLE")
            before = table_digest(conn)
            self.home = mapping.HomeMap(conn, "r", clock=self.clock, embedder=TinyEmbedder())
            self.teach("kitchen")
            self.assertEqual(table_digest(conn), before)
        finally:
            memory_db._conn = None
            memory_db._get_conn = old_get

    def test_speaks_only_when_allowed(self):
        self.say("this room is the kitchen")
        said = []
        for k in range(26 * 15):
            self.clock.t += 1 / 15
            self.heading = ((self.heading + 1.0) + 180) % 360 - 180
            m = self.home.observe(frame_at(self.strips["kitchen"], self.heading), None, None, self.heading,
                                  False, self.clock(), can_speak=False)
            self.assertIsNone(m)
        m = self.home.observe(frame_at(self.strips["kitchen"], 0), None, None, 0, True, self.clock(), can_speak=True)
        self.assertTrue(m.startswith("Done"))      # kept until SENSE may speak


if __name__ == "__main__":
    unittest.main()
