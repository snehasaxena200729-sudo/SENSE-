"""The OCR workflow with a fake engine and a fake clock (fast, no Tesseract needed).

These tests cover the WHOLE-FRAME path (use_text_detection=False): it is still
used by reading mode ("keep reading"), as the fallback when no text region is
found, and when text detection is switched off. The text-detection path of
"read this" is tested in test_target_reading.py."""
import threading
import unittest

from ocr.config import OCRConfig
from ocr.ocr_controller import UNAVAILABLE_REPLY, OCRController
from ocr.tests.helpers import Clock, FakeEngine, scene, words

SHARP = scene(["EXIT"])
BLURRY = scene(["EXIT"], blur=21)


def make(result_words=None, **engine_kw):
    clock = Clock()
    eng = FakeEngine(clock, result_words or words(("EXIT", 95)), **engine_kw)
    return OCRController(config=OCRConfig(use_text_detection=False), engine=eng, clock=clock,
                         log=lambda *_: None), eng, clock


def run(ctrl, clock, frames, dt=0.1, moving=False, can_speak=True):
    """Feed frames until a message comes out (or frames run out)."""
    for _ in range(frames):
        clock.t += dt
        ctrl.feed(SHARP, clock.t, camera_moving=moving)
        msg = ctrl.poll(clock.t, can_speak=can_speak)
        if msg:
            return msg
    return None


class CommandTests(unittest.TestCase):
    def setUp(self):
        self.ctrl, _, _ = make()

    def test_ocr_phrases(self):
        for t, cmd in [("read this", "ONCE"), ("what does this say?", "ONCE"),
                       ("Can you read the label", "ONCE"), ("keep reading", "CONTINUOUS"),
                       ("start reading", "CONTINUOUS"), ("stop reading", "STOP"),
                       ("reading off", "STOP")]:
            self.assertEqual(self.ctrl.match_command(t), cmd, t)

    def test_existing_sense_questions_are_not_taken(self):
        # every one of these must still reach SENSE's own intents unchanged
        for t in ["where is my phone", "this is my water bottle", "this is sneha", "what changed",
                  "what do you see", "find my bottle", "stop looking", "cancel", "forget my cup",
                  "rename my bottle to my medicine", "who was here today", "speak slower",
                  "where did i leave my bread item", "i'm ready", "be quiet"]:
            self.assertIsNone(self.ctrl.match_command(t), t)
            self.assertIsNone(self.ctrl.handle_command(t, 0), t)

    def test_unavailable_says_how_to_install(self):
        ctrl, _, _ = make(available=False)
        self.assertEqual(ctrl.handle_command("read this", 0), UNAVAILABLE_REPLY)
        self.assertFalse(ctrl.active)


class OnceModeTests(unittest.TestCase):
    def test_off_by_default_and_feed_costs_nothing(self):
        ctrl, eng, clock = make()
        for _ in range(50):
            clock.t += 0.1
            ctrl.feed(None, clock.t)          # None would crash if it were touched
            self.assertIsNone(ctrl.poll(clock.t))
        self.assertEqual(eng.starts, 0)

    def test_read_this_answers_once(self):
        ctrl, eng, clock = make(words(("MEDICINE", 92), ("500", 90), ("MG", 88)))
        self.assertIn("Reading", ctrl.handle_command("read this", clock.t))
        msg = run(ctrl, clock, 20)
        self.assertEqual(msg.text, "It says: MEDICINE 500 MG")
        self.assertTrue(msg.is_answer)
        self.assertFalse(ctrl.active)
        self.assertEqual(eng.starts, 1)
        self.assertIsNone(run(ctrl, clock, 30), "one answer only")

    def test_no_text_still_answers(self):
        ctrl, _, clock = make(words(("qx", 20)))
        ctrl.handle_command("read this", clock.t)
        self.assertIn("can't find any readable text", run(ctrl, clock, 20).text)

    def test_second_layout_mode_tried_only_when_first_finds_nothing(self):
        ctrl, eng, clock = make(words(("qx", 20)))
        ctrl.handle_command("read this", clock.t)
        run(ctrl, clock, 30)
        self.assertEqual(eng.psms, list(ctrl.config.psm_modes))
        ctrl, eng, clock = make()
        ctrl.handle_command("read this", clock.t)
        run(ctrl, clock, 30)
        self.assertEqual(eng.psms, [ctrl.config.psm_modes[0]])

    def test_waits_while_camera_moves(self):
        ctrl, eng, clock = make()
        ctrl.handle_command("read this", clock.t)
        run(ctrl, clock, 10, moving=True)       # 1 s of moving: no read yet
        self.assertEqual(eng.starts, 0)
        self.assertIsNotNone(run(ctrl, clock, 20))
        self.assertEqual(eng.starts, 1)

    def test_blurry_frames_then_gives_up_and_tries_anyway(self):
        ctrl, eng, clock = make()
        ctrl.handle_command("read this", clock.t)
        for _ in range(int(ctrl.config.steady_wait_s / 0.1) - 2):
            clock.t += 0.1
            ctrl.feed(BLURRY, clock.t)
        self.assertEqual(eng.starts, 0, "blurry frames are not read while there is time to wait")
        for _ in range(20):
            clock.t += 0.1
            ctrl.feed(BLURRY, clock.t)
            if ctrl.poll(clock.t):
                break
        self.assertEqual(eng.starts, 1)

    def test_answer_waits_until_sense_is_quiet(self):
        ctrl, _, clock = make()
        ctrl.handle_command("read this", clock.t)
        self.assertIsNone(run(ctrl, clock, 20, can_speak=False))
        clock.t += 0.1
        self.assertIsNotNone(ctrl.poll(clock.t, can_speak=True))

    def test_stale_answer_dropped(self):
        ctrl, _, clock = make()
        ctrl.handle_command("read this", clock.t)
        run(ctrl, clock, 20, can_speak=False)
        clock.t += ctrl.config.answer_max_wait_s + 1
        self.assertIsNone(ctrl.poll(clock.t, can_speak=False))
        self.assertIsNone(ctrl.poll(clock.t, can_speak=True))

    def test_engine_error_is_reported_not_raised(self):
        ctrl, _, clock = make(error="tesseract exited with code 1")
        ctrl.handle_command("read this", clock.t)
        self.assertEqual(run(ctrl, clock, 20).text, "Sorry, I could not read that.")


class ContinuousModeTests(unittest.TestCase):
    def test_new_text_once_then_silence_then_stop(self):
        ctrl, eng, clock = make(words(("WET", 90), ("FLOOR", 90)))
        self.assertIn("Reading mode on", ctrl.handle_command("keep reading", clock.t))
        msg = run(ctrl, clock, 30)
        self.assertEqual(msg.text, "Text: WET FLOOR")
        self.assertFalse(msg.is_answer)
        self.assertIsNone(run(ctrl, clock, 100), "the same sign must not be repeated")
        self.assertGreater(eng.starts, 2, "it keeps reading in the background")
        self.assertEqual(ctrl.handle_command("stop reading", clock.t), "Reading mode off.")
        n = eng.starts
        run(ctrl, clock, 50)
        self.assertEqual(eng.starts, n)
        self.assertFalse(ctrl.active)

    def test_read_this_during_reading_mode_reads_once_and_says_mode_ended(self):
        ctrl, _, clock = make()
        ctrl.handle_command("keep reading", clock.t)
        self.assertIn("Reading mode off", ctrl.handle_command("read this", clock.t))
        self.assertTrue(run(ctrl, clock, 30).is_answer)
        self.assertFalse(ctrl.active)

    def test_stop_cancels_a_running_read(self):
        ctrl, eng, clock = make(delay=5.0)
        ctrl.handle_command("keep reading", clock.t)
        run(ctrl, clock, 3)
        self.assertTrue(eng.busy)
        ctrl.handle_command("stop reading", clock.t)
        self.assertFalse(eng.busy)
        self.assertEqual(eng.cancels, 1)

    def test_interval_between_reads(self):
        ctrl, eng, clock = make(delay=0.1)
        ctrl.handle_command("keep reading", clock.t)
        run(ctrl, clock, 100)                  # 10 s
        cfg = OCRConfig()
        self.assertLessEqual(eng.starts, int(10 / cfg.continuous_interval_s) + 1)


class SafetyTests(unittest.TestCase):
    def test_close_stops_engine(self):
        ctrl, eng, clock = make()
        ctrl.handle_command("keep reading", clock.t)
        run(ctrl, clock, 2)
        ctrl.close()
        self.assertTrue(eng.closed)
        self.assertFalse(ctrl.active)

    def test_no_threads_started(self):
        before = threading.active_count()
        ctrl, _, clock = make()
        ctrl.handle_command("keep reading", clock.t)
        run(ctrl, clock, 50)
        self.assertEqual(threading.active_count(), before)


if __name__ == "__main__":
    unittest.main()
