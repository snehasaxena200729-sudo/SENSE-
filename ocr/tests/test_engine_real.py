"""Real Tesseract on synthetic camera-like frames. Skipped when Tesseract is not installed."""
import os
import re

import cv2
import threading
import time
import unittest

from ocr.config import OCRConfig
from ocr.image_preprocessing import prepare
from ocr.ocr_controller import OCRController
from ocr.ocr_engine import TesseractEngine, find_tesseract
from ocr.tests.helpers import scene
from ocr.text_filter import extract_text

HAVE_TESSERACT = find_tesseract(OCRConfig()) is not None
OCR_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@unittest.skipUnless(HAVE_TESSERACT, "tesseract not installed (brew install tesseract)")
class RealTesseractTests(unittest.TestCase):
    def read(self, img):
        cfg = OCRConfig()
        eng = TesseractEngine(cfg)
        try:
            image, _ = prepare(img, cfg)
            for psm in cfg.psm_modes:          # same fallback order as OCRController
                r = eng.read_blocking(image, psm)
                self.assertTrue(r.ok, r.error)
                text = extract_text(r.words, cfg)
                if text:
                    return text
            return ""
        finally:
            eng.close()

    def test_reads_label(self):
        self.assertEqual(self.read(scene(["MEDICINE", "500 MG TABLETS"])), "MEDICINE. 500 MG TABLETS")

    def test_reads_small_text(self):
        self.assertEqual(self.read(scene(["EXIT", "Room 204"], scale=0.7)), "EXIT. Room 204")

    def test_reads_white_on_dark_sign(self):
        self.assertEqual(self.read(scene(["EMERGENCY EXIT"], dark=True)), "EMERGENCY EXIT")

    def test_reads_single_big_line(self):
        # found in the full-app video run: layout mode 11 alone read nothing here
        img = scene([], w=1280, h=720, noise=6)
        cv2.rectangle(img, (300, 220), (920, 390), (232, 232, 225), -1)
        cv2.putText(img, "ROOM 204", (330, 315), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (25, 25, 25), 4, cv2.LINE_AA)
        self.assertEqual(self.read(img), "ROOM 204")

    def test_reads_in_uneven_light(self):
        self.assertEqual(self.read(scene(["Keep Left", "Wet Floor"], gradient=True)), "Keep Left. Wet Floor")

    def test_no_text_gives_nothing(self):
        for seed in range(5):
            self.assertEqual(self.read(scene([], seed=seed)), "")

    def test_start_returns_immediately(self):
        cfg = OCRConfig()
        eng = TesseractEngine(cfg)
        image, _ = prepare(scene(["MEDICINE", "500 MG TABLETS"]), cfg)
        t0 = time.perf_counter()
        eng.start(image)
        started = time.perf_counter() - t0
        self.assertIsNone(eng.poll(), "tesseract cannot be finished already")
        self.assertLess(started, 0.1, f"start() blocked for {started:.3f}s")
        eng.close()

    def test_timeout_kills_process(self):
        eng = TesseractEngine(OCRConfig(timeout_s=0.0))
        eng.start(prepare(scene(["EXIT"]), OCRConfig())[0])
        time.sleep(0.01)
        r = eng.poll()
        self.assertIsNotNone(r)
        self.assertIn("timed out", r.error)
        self.assertFalse(eng.busy)
        eng.close()

    def test_full_controller_with_real_tesseract(self):
        before = threading.active_count()
        ctrl = OCRController(log=lambda *_: None)
        ctrl.handle_command("what does this say", time.time())
        frame = scene(["WET FLOOR"])
        msg, t_end = None, time.time() + 10
        detect_frames = 0
        while msg is None and time.time() < t_end:
            t0 = time.perf_counter()
            ctrl.feed(frame, time.time())
            msg = ctrl.poll(time.time())
            took = time.perf_counter() - t0
            # Tesseract never blocks a frame. The text detector (EAST) runs inside
            # the frame, once per "read this" in the normal case: it may take up
            # to ~1 s on a slow CPU; every other frame stays under 0.25 s.
            if took > 0.25:
                detect_frames += 1
                self.assertLess(took, 1.5, "text detection took too long")
            time.sleep(0.03)        # ~30 fps loop
        self.assertLessEqual(detect_frames, 2, "only the detection frame(s) may be slow")
        ctrl.close()
        self.assertEqual(msg.text, "It says: WET FLOOR")
        self.assertEqual(threading.active_count(), before)


def _source(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


class IsolationTests(unittest.TestCase):
    def test_ocr_package_never_opens_a_camera_or_thread(self):
        for root, _, files in os.walk(OCR_DIR):
            if os.path.basename(root) == "tests":
                continue
            for f in files:
                if f.endswith(".py"):
                    src = _source(os.path.join(root, f))
                    self.assertNotRegex(src, r"VideoCapture\s*\(", f)
                    self.assertNotRegex(src, r"threading|Thread\s*\(", f)

    def test_ocr_package_does_not_import_sense_modules(self):
        sense = {"sense_master", "memory_db", "notifier", "answers", "identity", "events", "spatial",
                 "distance", "hazards", "people", "continuity", "face_id", "guidance", "ultralytics"}
        for root, _, files in os.walk(OCR_DIR):
            for f in files:
                if f.endswith(".py"):
                    src = _source(os.path.join(root, f))
                    imported = set(re.findall(r"^\s*(?:from|import)\s+(\w+)", src, re.M))
                    self.assertFalse(imported & sense, f"{f} imports {imported & sense}")


if __name__ == "__main__":
    unittest.main()
