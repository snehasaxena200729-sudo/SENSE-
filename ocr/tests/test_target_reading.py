"""'Read this' with text detection: grouping, target choice, straightening,
quality judging, camera tips and the controller flow.

Fast tests use a fake detector and a fake Tesseract. The tests at the bottom
use the real detector and real Tesseract and are skipped when either is missing."""
import threading
import unittest

import cv2
import numpy as np

from ocr import camera_guidance as tips
from ocr.config import OCRConfig
from ocr.image_preprocessing import apply_recipe, plan_recipes
from ocr.ocr_controller import OCRController
from ocr.ocr_engine import TesseractEngine, Word
from ocr.quality import CLEAR, PARTIAL, UNCLEAR, UNREADABLE, clean_words, combine, evaluate
from ocr.rectify import crop_block
from ocr.target_selector import choose
from ocr.tests.helpers import Clock, FakeEngine, words
from ocr.text_detector import TextDetector, ensure_east_model, make_box
from ocr.text_regions import group_blocks

CFG = OCRConfig()


def word_box(x, y, w, h, angle=0.0, frame=(1280, 720)):
    """A detector word box with top-left (x, y), size w x h (before rotation)."""
    return make_box(((x + w / 2.0, y + h / 2.0), (w, h), angle), 0.9, *frame)


def frame_with_text(lines, centre=(640, 380), size=1.2, angle=0, bg=110, w=1280, h=720, card=(240, 240, 235)):
    img = np.full((h, w, 3), bg, np.uint8)
    rng = np.random.default_rng(3)
    img = np.clip(img.astype(int) + rng.normal(0, 3, img.shape), 0, 255).astype(np.uint8)
    lw = max(cv2.getTextSize(t, cv2.FONT_HERSHEY_DUPLEX, size, 2)[0][0] for t in lines) + 60
    lh = int(45 * size) * len(lines) + 40
    c = np.full((lh, lw, 3), card, np.uint8)
    for i, t in enumerate(lines):
        cv2.putText(c, t, (30, 30 + int(45 * size) * i + int(25 * size)), cv2.FONT_HERSHEY_DUPLEX, size,
                    (20, 20, 20), 2, cv2.LINE_AA)
    M = cv2.getRotationMatrix2D((lw / 2, lh / 2), angle, 1.0)
    M[:, 2] += [centre[0] - lw / 2, centre[1] - lh / 2]
    warped = cv2.warpAffine(c, M, (w, h))
    mask = cv2.warpAffine(np.full((lh, lw), 255, np.uint8), M, (w, h))
    img[mask > 0] = warped[mask > 0]
    return img


class FakeDetector:
    name = "fake"
    last_seconds = 0.0

    def __init__(self, boxes):
        self.boxes = boxes
        self.calls = 0

    def detect(self, frame):
        self.calls += 1
        return list(self.boxes)


# --------------------------------------------------------------- grouping
class GroupingTests(unittest.TestCase):
    def test_words_on_a_line_and_lines_of_a_label_join(self):
        boxes = [word_box(500, 300, 120, 30), word_box(635, 300, 90, 30),      # line 1
                 word_box(500, 345, 150, 30)]                                 # line 2
        blocks = group_blocks(boxes, CFG)
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0].lines, 2)

    def test_far_apart_labels_stay_separate(self):
        boxes = [word_box(100, 300, 120, 30), word_box(900, 300, 120, 30)]
        self.assertEqual(len(group_blocks(boxes, CFG)), 2)

    def test_block_gets_the_yolo_object_it_sits_on(self):
        blocks = group_blocks([word_box(500, 300, 120, 30)], CFG, objects=[("bottle", (450, 200, 700, 500))])
        self.assertEqual(blocks[0].obj, "bottle")


# --------------------------------------------------------------- choosing
class TargetTests(unittest.TestCase):
    def setUp(self):
        self.frame = frame_with_text(["X"])        # content only matters for the sharpness signal

    def test_big_central_label_beats_small_corner_poster(self):
        blocks = group_blocks([word_box(540, 330, 200, 40), word_box(540, 380, 160, 40),
                               word_box(40, 40, 60, 12), word_box(105, 40, 50, 12)], CFG)
        ch = choose(blocks, self.frame, CFG)
        self.assertFalse(ch.ambiguous)
        self.assertGreater(ch.block.rect[0], 400)

    def test_two_equal_signs_far_apart_are_ambiguous(self):
        blocks = group_blocks([word_box(300, 340, 120, 30), word_box(860, 340, 120, 30)], CFG)
        self.assertTrue(choose(blocks, self.frame, CFG).ambiguous)

    def test_blocks_on_one_object_are_one_target(self):
        boxes = [word_box(520, 250, 120, 25), word_box(520, 420, 120, 25)]   # far apart lines
        objs = [("bottle", (480, 200, 700, 500))]
        ch = choose(group_blocks(boxes, CFG, objs), self.frame, CFG)
        self.assertFalse(ch.ambiguous)
        self.assertEqual(len(ch.block.boxes), 2)

    def test_parts_of_one_sign_close_together_are_one_target(self):
        boxes = [word_box(500, 300, 150, 30), word_box(500, 380, 150, 30)]   # gap 50 px < 3 x 30
        ch = choose(group_blocks(boxes, CFG), self.frame, CFG)
        self.assertFalse(ch.ambiguous)
        self.assertEqual(len(ch.block.boxes), 2)

    def test_no_blocks_no_choice(self):
        self.assertIsNone(choose([], self.frame, CFG))


# ------------------------------------------------------------ straightening
class RectifyTests(unittest.TestCase):
    def test_tilted_block_is_deskewed_level_one_is_plain_crop(self):
        gray = cv2.cvtColor(frame_with_text(["X"]), cv2.COLOR_BGR2GRAY)
        cfg = OCRConfig(use_perspective=False)
        tilted = group_blocks([word_box(500, 300, 240, 40, angle=20)], cfg)[0]
        crop, M, how = crop_block(gray, tilted, cfg)
        self.assertEqual(how, "deskew")
        self.assertGreater(crop.shape[1], crop.shape[0] * 3, "the long side of the text is horizontal again")
        level = group_blocks([word_box(500, 300, 240, 40)], cfg)[0]
        self.assertEqual(crop_block(gray, level, cfg)[2], "crop")


# ------------------------------------------------------------ quality
def cw(text, conf, left, top=10, line=1, num=1, w=60, h=30):
    return Word(text, conf, 1, 1, line, num, left, top, w, h)


class QualityTests(unittest.TestCase):
    def test_clear_reading(self):
        r = evaluate([cw("ROOM", 95, 10), cw("204", 96, 80)], CFG, [(40, 25), (110, 25)])
        self.assertEqual((r.status, r.text), (CLEAR, "ROOM 204"))

    def test_missing_words_make_it_partial(self):
        r = evaluate([cw("ROOM", 95, 10)], CFG, [(40, 25), (110, 25), (200, 25)])
        self.assertEqual(r.status, PARTIAL)

    def test_text_cut_by_the_frame_edge_is_partial(self):
        self.assertEqual(evaluate([cw("ROOM", 95, 10)], CFG, [(40, 25)], cut=True).status, PARTIAL)

    def test_unsure_number_is_named(self):
        r = evaluate([cw("Take", 95, 10), cw("8", 70, 80, w=20)], CFG, [(40, 25), (90, 25)])
        self.assertEqual(r.status, UNCLEAR)
        self.assertEqual(r.unsure_numbers, ["8"])

    def test_nothing_trustworthy_is_unreadable(self):
        self.assertEqual(evaluate([cw("xq", 20, 10)], CFG, [(40, 25)]).status, UNREADABLE)

    def test_label_edge_junk_is_cleaned(self):
        ws = [cw("|500", 90, 10), cw("mg", 92, 80, num=2), cw("j", 70, 0, line=2), cw("WET", 95, 20, line=2, num=2),
              cw("FLOOR", 95, 90, line=2, num=3)]
        self.assertEqual([w.text for w in clean_words(ws, CFG)], ["500", "mg", "WET", "FLOOR"])

    def test_voting_keeps_the_most_confident_word_per_place(self):
        a = [cw("PARACETAM0L", 62, 10), cw("500", 95, 10, top=60)]
        b = [cw("PARACETAMOL", 93, 12), cw("5OO", 50, 10, top=60)]
        self.assertEqual([w.text for w in combine([a, b])], ["PARACETAMOL", "500"])


# ------------------------------------------------------------ camera tips
class GuidanceTests(unittest.TestCase):
    def block(self, x=500, y=300, w=200, h=30, angle=0):
        return group_blocks([word_box(x, y, w, h, angle)], CFG)[0]

    def test_tips(self):
        sharp = frame_with_text(["ROOM 204"], size=1.5)
        self.assertEqual(tips.failed_read_tip((sharp * 0.15).astype(np.uint8), self.block(), CFG), tips.TIP_DARK)
        self.assertEqual(tips.failed_read_tip(cv2.GaussianBlur(sharp, (0, 0), 6), self.block(), CFG), tips.TIP_BLUR)
        self.assertEqual(tips.failed_read_tip(sharp, self.block(h=8, w=60), CFG), tips.TIP_SMALL)
        self.assertEqual(tips.failed_read_tip(sharp, self.block(x=0), CFG), tips.TIP_CUT)
        self.assertEqual(tips.failed_read_tip(sharp, self.block(angle=25), CFG), tips.TIP_TILT)
        glare = sharp.copy()
        glare[300:330, 500:700] = 255
        self.assertEqual(tips.failed_read_tip(glare, self.block(), CFG), tips.TIP_GLARE)

    def test_no_text_tips(self):
        room = np.full((720, 1280, 3), 120, np.uint8)
        room = np.clip(room + np.random.default_rng(1).normal(0, 30, room.shape), 0, 255).astype(np.uint8)
        self.assertIn("bottle", tips.no_text_tip(room, [("bottle", (500, 200, 700, 600))], CFG))
        self.assertEqual(tips.no_text_tip(room, [], CFG), tips.TIP_NO_TEXT)
        self.assertEqual(tips.no_text_tip((room * 0.1).astype(np.uint8), [], CFG), tips.TIP_DARK)


class RecipeTests(unittest.TestCase):
    def test_recipes_order_follows_the_image(self):
        self.assertEqual(plan_recipes({"mean": 40, "noise": 3, "uneven": 0.1})[0], "denoise")
        self.assertEqual(plan_recipes({"mean": 150, "noise": 3, "uneven": 0.4})[1], "adaptive")
        self.assertEqual(plan_recipes({"mean": 150, "noise": 3, "uneven": 0.1})[0], "clahe")

    def test_every_recipe_gives_a_padded_gray_image(self):
        g = np.full((60, 200), 200, np.uint8)
        for name in ("clahe", "sharpen", "otsu", "adaptive", "denoise"):
            img, pad = apply_recipe(name, g, 34)
            self.assertEqual(img.ndim, 2)
            self.assertEqual(img.shape, (60 + 2 * pad, 200 + 2 * pad))


# ------------------------------------------------------------ controller flow
LABEL = [word_box(520, 320, 180, 40), word_box(520, 380, 120, 40)]


def make(boxes, result_words, delay=0.3, cfg=None):
    clock = Clock()
    eng = FakeEngine(clock, result_words, delay=delay)
    det = FakeDetector(boxes)
    ctrl = OCRController(config=cfg or OCRConfig(), engine=eng, clock=clock, log=lambda *_: None, detector=det)
    return ctrl, eng, det, clock


def run(ctrl, clock, frame, frames=80, dt=0.1):
    for _ in range(frames):
        clock.t += dt
        ctrl.feed(frame, clock.t)
        msg = ctrl.poll(clock.t)
        if msg:
            return msg
    return None


FRAME = frame_with_text(["X"])


def label_words(conf=95):
    # positions in the scaled crop do not matter for the fake engine: coverage
    # is checked separately in QualityTests
    return [Word("PARACETAMOL", conf, 1, 1, 1, 1, 0, 0, 2000, 2000), Word("500", conf, 1, 1, 2, 1, 0, 0, 2000, 2000)]


class ControllerTests(unittest.TestCase):
    def test_reads_the_chosen_block_once_when_the_first_recipe_is_good(self):
        ctrl, eng, det, clock = make(LABEL, label_words())
        ctrl.handle_command("read this", clock.t)
        msg = run(ctrl, clock, FRAME)
        self.assertEqual(msg.text, "It says: PARACETAMOL. 500")
        self.assertEqual(eng.starts, 1)
        self.assertEqual(eng.psms, [6], "two lines -> block mode")
        self.assertEqual(det.calls, 1)

    def test_bad_readings_try_other_recipes_then_never_narrate_garbage(self):
        ctrl, eng, _, clock = make(LABEL, words(("qx", 20)))
        ctrl.handle_command("read this", clock.t)
        msg = run(ctrl, clock, FRAME)
        self.assertEqual(eng.starts, ctrl.config.max_attempts)
        self.assertFalse(msg.text.startswith("It says"))
        self.assertIn("can't read it", msg.text)

    def test_ambiguous_choice_gives_a_framing_tip_and_reads_nothing(self):
        boxes = [word_box(250, 340, 120, 30), word_box(910, 340, 120, 30)]
        ctrl, eng, _, clock = make(boxes, label_words())
        ctrl.handle_command("read this", clock.t)
        msg = run(ctrl, clock, FRAME)
        self.assertEqual(msg.text, tips.TIP_AMBIGUOUS)
        self.assertEqual(eng.starts, 0)

    def test_no_text_regions_falls_back_to_the_whole_picture_then_a_tip(self):
        ctrl, eng, _, clock = make([], words(("qx", 20)))
        ctrl.set_objects([("bottle", (500, 200, 760, 600))])
        ctrl.handle_command("read this", clock.t)
        msg = run(ctrl, clock, FRAME)
        self.assertEqual(eng.psms, list(ctrl.config.psm_modes), "the old whole-frame read is tried")
        self.assertIn("bottle", msg.text)

    def test_unsure_numbers_are_named(self):
        ws = [Word("Take", 95, 1, 1, 1, 1, 0, 0, 2000, 2000), Word("8", 72, 1, 1, 1, 2, 0, 0, 2000, 2000)]
        ctrl, _, _, clock = make(LABEL, ws)
        ctrl.handle_command("read this", clock.t)
        self.assertIn("I'm not sure about 8", run(ctrl, clock, FRAME).text)

    def test_hud_boxes_only_after_read_and_only_for_a_while(self):
        ctrl, _, _, clock = make(LABEL, label_words())
        canvas = np.zeros((720, 1280, 3), np.uint8)
        ctrl.draw(canvas, clock.t)
        self.assertEqual(canvas.sum(), 0, "nothing is drawn before the user asks")
        ctrl.handle_command("read this", clock.t)
        run(ctrl, clock, FRAME)
        ctrl.draw(canvas, clock.t)
        self.assertGreater(canvas.sum(), 0)
        canvas[:] = 0
        ctrl.draw(canvas, clock.t + ctrl.config.show_boxes_s + 1)
        self.assertEqual(canvas.sum(), 0)

    def test_waits_while_the_camera_moves_and_never_starts_threads(self):
        before = threading.active_count()
        ctrl, eng, det, clock = make(LABEL, label_words())
        ctrl.handle_command("read this", clock.t)
        for _ in range(10):
            clock.t += 0.1
            ctrl.feed(FRAME, clock.t, camera_moving=True)
        self.assertEqual(det.calls, 0)
        self.assertIsNotNone(run(ctrl, clock, FRAME))
        self.assertEqual(threading.active_count(), before)

    def test_switch_off_restores_the_old_whole_frame_reading(self):
        ctrl, eng, det, clock = make(LABEL, label_words(), cfg=OCRConfig(use_text_detection=False))
        ctrl.handle_command("read this", clock.t)
        run(ctrl, clock, FRAME)
        self.assertEqual(det.calls, 0)
        self.assertEqual(eng.psms[0], ctrl.config.psm_modes[0])


# ------------------------------------------------------------ real detector + Tesseract
REAL_OK = TesseractEngine(CFG).available


@unittest.skipUnless(REAL_OK, "tesseract not installed")
class RealReadingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.has_east = ensure_east_model(CFG, download=False, log=lambda *_: None) is not None

    def read(self, frame, objects=()):
        import time
        ctrl = OCRController(log=lambda *_: None)
        ctrl.set_objects(objects)
        ctrl.handle_command("read this", time.time())
        msg, t_end = None, time.time() + 15
        while msg is None and time.time() < t_end:
            ctrl.feed(frame, time.time())
            msg = ctrl.poll(time.time())
            time.sleep(0.03)
        ctrl.close()
        return msg.text

    def test_tilted_label(self):
        if not self.has_east:
            self.skipTest("EAST model not downloaded")
        text = self.read(frame_with_text(["VITAMIN D3", "1000 IU"], angle=22))
        self.assertIn("VITAMIN D3", text)
        self.assertIn("1000", text)

    def test_held_label_is_read_not_the_small_poster(self):
        if not self.has_east:
            self.skipTest("EAST model not downloaded")
        frame = frame_with_text(["METFORMIN 500 mg"], centre=(660, 420), size=1.3)
        cv2.putText(frame, "SALE 50% OFF", (30, 50), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        text = self.read(frame)
        self.assertIn("METFORMIN", text)
        self.assertNotIn("SALE", text)

    def test_blank_wall_is_never_narrated(self):
        wall = np.clip(np.full((720, 1280, 3), 150, np.uint8)
                       + np.random.default_rng(2).normal(0, 6, (720, 1280, 3)), 0, 255).astype(np.uint8)
        self.assertFalse(self.read(wall).startswith("It says"))

    def test_detector_finds_text_and_reports_its_name(self):
        det = TextDetector(CFG, download=False, log=lambda *_: None)
        boxes = det.detect(frame_with_text(["ROOM 204"], size=1.5))
        self.assertTrue(boxes)
        self.assertIn(det.name, ("EAST", "classic"))


if __name__ == "__main__":
    unittest.main()
