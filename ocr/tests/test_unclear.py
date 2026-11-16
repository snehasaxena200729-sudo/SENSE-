"""Regression (stabilisation audit, Nov 2026): uncertain OCR must never sound certain.

Reproduced with the real Tesseract on a blurred medicine label: it read
"Do not take more than 8 tablets" as "... more then 6 ..." and SENSE said
"It says: ..." as if that were a fact. The misread words scored 60-80; the
correct ones 85+. Now such a reading gets a spoken warning.
"""
import unittest

from ocr.config import OCRConfig
from ocr.ocr_controller import UNCLEAR_NOTE, OCRController
from ocr.tests.helpers import Clock, FakeEngine, scene, words
from ocr.text_filter import is_unclear

SHARP = scene(["EXIT"])
CFG = OCRConfig(use_text_detection=False)   # wording of the whole-frame answer


def read_once(result_words):
    clock = Clock()
    ctrl = OCRController(config=CFG, engine=FakeEngine(clock, result_words), clock=clock, log=lambda *_: None)
    ctrl.handle_command("read this", clock.t)
    for _ in range(100):
        clock.t += 0.1
        ctrl.feed(SHARP, clock.t)
        msg = ctrl.poll(clock.t)
        if msg:
            return msg.text
    return None


class UnclearReadingTests(unittest.TestCase):
    def test_all_words_sure_is_not_unclear(self):
        self.assertFalse(is_unclear(words(("Room", 95), ("204", 93)), CFG))

    def test_low_but_kept_word_is_unclear(self):
        # "6" passed the 60 filter, so it would be spoken, but Tesseract was unsure
        self.assertTrue(is_unclear(words(("more", 96), ("then", 66), ("6", 71), ("tablets", 92)), CFG))

    def test_dropped_real_word_leaves_a_gap(self):
        # "not" dropped: the spoken text would be "Do take more" - the opposite meaning
        self.assertTrue(is_unclear(words(("Do", 95), ("not", 45), ("take", 95), ("more", 95)), CFG))

    def test_dropped_noise_does_not_count(self):
        # texture noise (very low confidence or no letters) is not a missing word
        self.assertFalse(is_unclear(words(("EXIT", 95), ("~~", 50), ("ee", 20)), CFG))

    def test_sure_answer_has_no_warning(self):
        self.assertEqual(read_once(words(("EXIT", 95))), "It says: EXIT")

    def test_unsure_answer_is_flagged(self):
        reply = read_once(words(("Take", 95), ("2", 70), ("tablets", 95)))
        self.assertTrue(reply.startswith("It says: Take 2 tablets"), reply)
        self.assertTrue(reply.endswith(UNCLEAR_NOTE), reply)
        self.assertNotIn("..", reply)

    def test_no_text_reply_unchanged(self):
        reply = read_once(words(("ee", 20)))
        self.assertTrue(reply.startswith("I can't find any readable text"), reply)


if __name__ == "__main__":
    unittest.main()
