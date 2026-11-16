"""
ocr/ocr_controller.py - the OCR workflow, as a small state machine.

SENSE talks to OCR only through four calls, all of them non-blocking:

    ocr = OCRController.create()                # once, at start-up (never raises)
    reply = ocr.handle_command(text, now)       # a voice question: a reply, or None = not about reading
    ocr.feed(frame, now, camera_moving)         # every frame (does nothing while OCR is off)
    msg = ocr.poll(now, can_speak)              # every frame: an OCRMessage to say, or None
    ocr.close()                                 # on quit

It never opens a camera: it only looks at frames SENSE already captured.

States
    OFF      nothing happens; feed() returns immediately (zero cost).
    WAITING  looking for a steady, sharp frame. Gives up after steady_wait_s
             with "hold the text steady" (ONCE mode) or tries again later
             (CONTINUOUS mode).
    READING  tesseract runs as a separate process; poll() checks if it is done.

Modes
    ONCE        "read this": one read, then always one answer (text, or "no text").
    CONTINUOUS  "keep reading": a read every continuous_interval_s; only NEW text
                is announced, silence otherwise. "stop reading" ends it.

Speech: SENSE decides how to say an OCRMessage. is_answer=True means the user
asked for it (say it when SENSE is quiet); False means continuous-mode news
(let the notifier decide, so safety warnings always come first).
"""

import re
import time
from collections import namedtuple

import numpy as np

from . import camera_guidance as tips
from . import hud
from .config import OCRConfig
from .image_preprocessing import (apply_recipe, image_stats, plan_recipes, prepare,
                                  scale_to_letter_height, sharpness, to_gray)
from .ocr_engine import TesseractEngine
from .quality import CLEAR, PARTIAL, UNCLEAR, UNREADABLE, better, combine, evaluate
from .rectify import crop_block, map_points
from .target_selector import choose, where_in_frame
from .text_detector import TextDetector, ensure_east_model
from .text_filter import RepeatFilter, extract_text, is_unclear
from .text_regions import group_blocks

OCRMessage = namedtuple("OCRMessage", "text is_answer key")

OFF, WAITING, READING = "OFF", "WAITING", "READING"
ONCE, CONTINUOUS = "ONCE", "CONTINUOUS"

UNCLEAR_NOTE = "Some words were unclear, so please check it, especially any numbers."


def _with_note(sentence, unclear):
    return sentence.rstrip(" .") + ". " + UNCLEAR_NOTE if unclear else sentence


UNAVAILABLE_REPLY = ("Text reading is not set up. Install Tesseract with: brew install tesseract, "
                     "then restart SENSE.")


def _phrase_regex(phrases):
    # whole words only: "read it" must not match inside "bread item"
    return re.compile(r"\b(" + "|".join(re.escape(p) for p in phrases) + r")\b")


class OCRController:
    def __init__(self, config=None, engine=None, clock=time.time, log=print, detector=None):
        self.config = config or OCRConfig()
        self.engine = engine if engine is not None else TesseractEngine(self.config, clock=clock)
        # Text detection ("where is the text?"). Never downloads here: create()
        # fetches the model once at SENSE start-up; without it the classic
        # detector is used.
        if not self.config.use_text_detection:
            detector = None
        elif detector is None:
            detector = TextDetector(self.config, download=False, log=log)
        self.detector = detector
        self.objects = []                # (name, (x1, y1, x2, y2)) from SENSE's YOLO, last frame
        self.target = None               # TargetRead while reading a chosen block
        self.last_detect_at = -1e9
        self.best_candidate = None       # (quality, frame, blocks, choice) seen while waiting
        self.saw_ambiguous = False
        self.hud_blocks, self.hud_chosen, self.hud_label, self.hud_failed = [], None, "", False
        self.hud_until = 0.0
        self.clock = clock
        self.log = log
        self.state = OFF
        self.mode = None
        self.wait_started = 0.0
        self.next_read_at = 0.0
        self.best_blurry = None          # sharpest rejected frame while waiting (fallback)
        self.image = None                # the preprocessed frame being read
        self.attempt = 0                 # index into config.psm_modes
        self.pending = None              # OCRMessage waiting to be spoken
        self.pending_since = 0.0
        self.repeats = RepeatFilter(self.config.duplicate_similarity, self.config.repeat_cooldown_s)
        self._stop_re = _phrase_regex(self.config.stop_phrases)
        self._cont_re = _phrase_regex(self.config.continuous_phrases)
        self._once_re = _phrase_regex(self.config.once_phrases)

    # ------------------------------------------------------------------ setup
    @classmethod
    def create(cls, config=None, log=print):
        """Safe constructor for SENSE: prints whether OCR is usable, never raises."""
        config = config or OCRConfig()
        if config.use_text_detection and config.use_east:
            ensure_east_model(config, download=True, log=log)     # one-time download, then offline
        ctrl = cls(config=config, log=log)
        if ctrl.available:
            log(f"📖 Text reading ready (Tesseract: {ctrl.engine.cmd}). Say 'read this' or press R.")
            if ctrl.detector is not None:
                try:        # the first network run is slow (memory set-up): do it now, not on R
                    ctrl.detector.detect(np.zeros((360, 640, 3), np.uint8))
                except Exception:
                    pass
                log(f"📖 Text detector: {ctrl.detector.name}"
                    + ("" if ctrl.detector.name == "EAST" else " (EAST model missing: weaker fallback)"))
        else:
            log("📖 Text reading off: Tesseract not found. Install it with: brew install tesseract")
        return ctrl

    @property
    def available(self):
        return self.engine.available

    @property
    def active(self):
        return self.state != OFF

    # --------------------------------------------------------------- commands
    def match_command(self, text):
        """'STOP' | 'CONTINUOUS' | 'ONCE' | None. Stop is checked first."""
        t = text.lower()
        if self._stop_re.search(t):
            return "STOP"
        if self._cont_re.search(t):
            return "CONTINUOUS"
        if self._once_re.search(t):
            return "ONCE"
        return None

    def handle_command(self, text, now=None):
        """Reply sentence when `text` is an OCR command, else None (SENSE goes on as before)."""
        cmd = self.match_command(text)
        if cmd is None:
            return None
        now = self.clock() if now is None else now
        if cmd == "STOP":
            was_on = self.mode == CONTINUOUS
            self.stop()
            return "Reading mode off." if was_on else "Okay."
        if not self.available:
            return UNAVAILABLE_REPLY
        if cmd == "CONTINUOUS":
            self._begin(CONTINUOUS, now)
            return "Reading mode on. I will read new text I see. Say stop reading to end."
        was_continuous = self.mode == CONTINUOUS
        self._begin(ONCE, now)           # "read this" during reading mode: one fresh read, mode ends
        return ("Reading mode off. Reading once. Hold it steady." if was_continuous
                else "Reading. Hold it steady.")

    def _begin(self, mode, now):
        self.engine.cancel()
        self.mode = mode
        self.state = WAITING
        self.wait_started = now
        self.next_read_at = now
        self.best_blurry = None
        self.pending = None
        self.target = None
        self.best_candidate = None
        self.saw_ambiguous = False
        self.last_detect_at = -1e9
        if mode == CONTINUOUS:
            self.repeats.clear()

    def stop(self):
        self.engine.cancel()
        self.state = OFF
        self.mode = None
        self.best_blurry = None
        self.pending = None
        self.target = None
        self.best_candidate = None

    # ------------------------------------------------------------- per frame
    def feed(self, frame, now, camera_moving=False):
        """Offer the current frame. Cheap no-op unless OCR is waiting for a frame."""
        if self.state != WAITING or now < self.next_read_at:
            return
        if self.mode == ONCE and self.detector is not None:
            self._feed_target(frame, now, camera_moving)
            return
        waited = now - self.wait_started
        give_up = waited >= self.config.steady_wait_s
        if camera_moving and not give_up:
            return
        image, score = prepare(frame, self.config)
        if score < self.config.blur_threshold and not give_up:
            if self.best_blurry is None or score > self.best_blurry[1]:
                self.best_blurry = (image, score)
            return
        if score < self.config.blur_threshold:
            # never got a sharp frame: in ONCE mode try the sharpest one we saw
            if self.mode == ONCE:
                if self.best_blurry is not None and self.best_blurry[1] > score:
                    image, score = self.best_blurry
                self.log(f"📖 OCR: no sharp frame in {waited:.1f}s (sharpness {score:.0f}), trying anyway")
            else:
                self._schedule_next(now)
                return
        try:
            self.image, self.attempt = image, 0
            self.engine.start(image, self.config.psm_modes[0])
            self.state = READING
            self.best_blurry = None
        except Exception as e:
            self.log(f"📖 OCR could not start: {e}")
            self._finish(now, None, error=True)

    def poll(self, now, can_speak=True):
        """An OCRMessage to say now, or None. Never blocks."""
        if self.state == READING and self.target is not None:
            self._poll_target(now)
        elif self.state == READING:
            result = self.engine.poll()
            if result is not None:
                if result.ok:
                    text = extract_text(result.words, self.config)
                    psm = self.config.psm_modes[self.attempt]
                    self.log(f"📖 OCR {result.seconds:.2f}s (psm {psm}): {len(result.words)} words -> '{text}'")
                    if not text and self._try_next_mode():
                        return None
                    self._finish(now, text, unclear=bool(text) and is_unclear(result.words, self.config))
                else:
                    self.log(f"📖 OCR failed: {result.error}")
                    self._finish(now, None, error=True)
        if self.pending is None:
            return None
        if self.pending.is_answer and not can_speak:
            if now - self.pending_since > self.config.answer_max_wait_s:
                self.pending = None      # too late to be useful
            return None
        msg, self.pending = self.pending, None
        return msg

    def _try_next_mode(self):
        """Nothing found: read the same image again with the next layout mode."""
        if self.attempt + 1 >= len(self.config.psm_modes):
            return False
        self.attempt += 1
        try:
            self.engine.start(self.image, self.config.psm_modes[self.attempt])
            return True
        except Exception as e:
            self.log(f"📖 OCR could not start: {e}")
            return False

    def _schedule_next(self, now):
        self.state = WAITING
        self.wait_started = now + self.config.continuous_interval_s
        self.next_read_at = now + self.config.continuous_interval_s
        self.best_blurry = None

    def _finish(self, now, text, error=False, unclear=False):
        self.image = None
        if self.mode == ONCE:
            if error:
                reply = "Sorry, I could not read that."
            elif text:
                reply = _with_note(f"It says: {text}", unclear)
            elif self.detector is not None and getattr(self, "last_frame", None) is not None:
                reply = tips.no_text_tip(self.last_frame, self.objects, self.config)
                self.last_frame = None
            else:
                reply = ("I can't find any readable text. Hold the text about 30 centimetres "
                         "from the camera, steady and in good light.")
            self.pending, self.pending_since = OCRMessage(reply, True, "ocr:answer"), now
            self.state, self.mode = OFF, None
            return
        if self.mode == CONTINUOUS:
            if text and not self.repeats.is_repeat(text, now):
                self.repeats.remember(text, now)
                self.pending = OCRMessage(_with_note(f"Text: {text}", unclear), False, "ocr:" + text[:40].lower())
                self.pending_since = now
            self._schedule_next(now)

    # ------------------------------------------- "read this" with text detection
    # ONCE mode, step by step (all in the main loop, Tesseract in its own process):
    #   1. every detect_interval_s while waiting: detect text, group into blocks,
    #      choose the target (target_selector.py). Boxes go to the HUD.
    #   2. wait (up to steady_wait_s) for a frame whose CHOSEN BLOCK is sharp and
    #      the choice is clear; otherwise keep the best candidate so far.
    #   3. crop + straighten the block, plan up to max_attempts recipes, read them
    #      one after another, keep the best reading (quality.py).
    #   4. answer: the text (with honest doubts), or one camera tip. Never a guess.
    detect_interval_s = 0.8          # EAST costs 0.1-0.5 s of the main loop, so not every frame

    def set_objects(self, objects):
        """SENSE's confirmed YOLO boxes from the latest frame: [(name, (x1, y1, x2, y2))]."""
        self.objects = list(objects or [])

    def draw(self, frame, now):
        """Draw text boxes on the HUD frame (only for a few seconds after a read)."""
        if now < self.hud_until and (self.hud_blocks or self.hud_chosen is not None):
            hud.draw(frame, self.hud_blocks, self.hud_chosen, self.hud_label, self.hud_failed)

    def _show(self, now, blocks, chosen=None, label="TEXT", failed=False):
        self.hud_blocks, self.hud_chosen, self.hud_label, self.hud_failed = blocks, chosen, label, failed
        self.hud_until = now + self.config.show_boxes_s

    def _feed_target(self, frame, now, camera_moving):
        waited = now - self.wait_started
        give_up = waited >= self.config.steady_wait_s
        if camera_moving and not give_up:
            return
        if now - self.last_detect_at < self.detect_interval_s and not give_up:
            return
        self.last_detect_at = now
        try:
            boxes = self.detector.detect(frame)
        except Exception as e:
            self.log(f"📖 Text detection failed ({e}); reading the whole picture")
            boxes = []
        blocks = group_blocks(boxes, self.config, self.objects)
        choice = choose(blocks, frame, self.config)
        if choice is not None:
            gray = to_gray(frame)
            x, y, w, h = choice.block.rect
            sharp = sharpness(gray[y:y + h, x:x + w])
            ok = sharp >= self.config.blur_threshold and not choice.ambiguous
            quality = (0 if choice.ambiguous else 2) + min(1.0, sharp / 400.0)
            if self.best_candidate is None or quality > self.best_candidate[0]:
                self.best_candidate = (quality, frame.copy(), blocks, choice)
            self.saw_ambiguous = self.saw_ambiguous or choice.ambiguous
            self._show(now, blocks, None if choice.ambiguous else choice.block, "TEXT")
            if not ok and not give_up:
                return
        elif not give_up:
            if self.best_blurry is None:
                self.best_blurry = (frame.copy(), 0)
            return
        if self.best_candidate is None:
            # nothing that looks like text: one whole-frame read (the old way),
            # in case the detector missed it. A miss there gets a camera tip.
            self.log(f"📖 OCR: no text regions found in {waited:.1f}s ({self.detector.name}); "
                     "trying the whole picture")
            self.last_frame = frame.copy()
            image, _ = prepare(frame, self.config)
            try:
                self.image, self.attempt = image, 0
                self.engine.start(image, self.config.psm_modes[0])
                self.state = READING
            except Exception as e:
                self.log(f"📖 OCR could not start: {e}")
                self._finish(now, None, error=True)
            return
        _, frame, blocks, choice = self.best_candidate
        self.best_candidate = None
        if choice.ambiguous:
            self.log(f"📖 OCR: {len(choice.scores)} text blocks scored too close to choose "
                     f"({', '.join(f'{s:.2f}' for s in choice.scores[:3])}); asking to frame it")
            self._show(now, blocks, None)
            self._answer(now, tips.TIP_AMBIGUOUS)
            return
        self._start_target(frame, blocks, choice, now)

    def _start_target(self, frame, blocks, choice, now):
        cfg = self.config
        block = choice.block
        gray = to_gray(frame)
        crop, M, how = crop_block(gray, block, cfg)
        scaled, s = scale_to_letter_height(crop, block.letter_h, cfg.target_letter_px, cfg.max_crop_side)
        recipes = plan_recipes(image_stats(scaled))[:cfg.max_attempts]
        centres = [b.quad.mean(axis=0) for b in block.boxes]
        fh, fw = gray.shape[:2]
        self.target = {"frame": frame, "block": block, "blocks": blocks, "how": how,
                       "img": scaled, "recipes": recipes, "i": 0, "best": None, "all_words": [],
                       "centres": map_points(M, centres) * s, "letter": block.letter_h * s,
                       "cut": tips.touches_edge(block.rect, fw, fh, cfg.edge_margin),
                       "psm": 7 if block.lines == 1 and len(block.boxes) <= 6 else (6 if block.lines <= 3 else 3)}
        self.log(f"📖 OCR target {where_in_frame(block, fw, fh)}"
                 + (f" on the {block.obj}" if block.obj else "")
                 + f": {len(block.boxes)} words, letters {block.letter_h:.0f}px, tilt {block.angle:.0f}°, "
                 f"{how}, score {choice.score:.2f} of {len(choice.scores)} block(s), recipes {recipes}, "
                 f"{self.detector.name} detection {getattr(self.detector, 'last_seconds', 0) * 1000:.0f} ms")
        self._show(now, blocks, block, "READING")
        self._start_attempt()

    def _start_attempt(self):
        t = self.target
        name = t["recipes"][t["i"]]
        img, pad = apply_recipe(name, t["img"], t["letter"])
        t["pad"] = pad
        try:
            self.engine.start(img, t["psm"])
            self.state = READING
        except Exception as e:
            self.log(f"📖 OCR could not start: {e}")
            self.target = None
            self._finish(self.clock(), None, error=True)

    def _poll_target(self, now):
        result = self.engine.poll()
        if result is None:
            return
        t = self.target
        name = t["recipes"][t["i"]]
        if not result.ok:
            self.log(f"📖 OCR failed ({name}): {result.error}")
            reading = None
        else:
            centres = np.asarray(t["centres"]) + t["pad"]
            reading = evaluate(result.words, self.config, centres, cut=t["cut"])
            t["all_words"].append(result.words)
            self.log(f"📖 OCR {result.seconds:.2f}s ({name}, psm {t['psm']}): {reading.status}, "
                     f"score {reading.score:.2f}, conf {reading.conf:.0f}, coverage {reading.coverage:.2f} "
                     f"-> '{reading.text}'")
        t["best"] = better(t["best"], reading)
        done = (t["best"] is not None and t["best"].status in (CLEAR, UNCLEAR)
                and t["best"].score >= self.config.good_score)
        t["i"] += 1
        if not done and t["i"] < len(t["recipes"]):
            self._start_attempt()
            return
        if not done and len(t["all_words"]) > 1:      # no single recipe was good: vote word by word
            centres = np.asarray(t["centres"]) + t["pad"]
            voted = evaluate(combine(t["all_words"]), self.config, centres, cut=t["cut"])
            self.log(f"📖 OCR combined {len(t['all_words'])} readings: {voted.status}, score {voted.score:.2f} "
                     f"-> '{voted.text}'")
            t["best"] = better(t["best"], voted)
        self.target = None
        self._finish_target(now, t)

    def _finish_target(self, now, t):
        r, block = t["best"], t["block"]
        if r is None or r.status == UNREADABLE:
            tip = tips.failed_read_tip(t["frame"], block, self.config, perspective=t["how"] == "perspective")
            self._show(now, t["blocks"], block, "CAN'T READ", failed=True)
            self._answer(now, "I can see some text but I can't read it. " + tip)
            return
        if r.status == PARTIAL:
            tip = tips.TIP_CUT if r.cut else tips.failed_read_tip(t["frame"], block, self.config)
            if tip == tips.TIP_GENERIC:
                tip = "Hold it closer and steady to hear the rest."
            reply = f"I can only read part of it: {r.text}. {tip}"
        elif r.status == UNCLEAR:
            reply = f"It says: {r.text}"
            if r.unsure_numbers:
                nums = ", ".join(dict.fromkeys(r.unsure_numbers))
                reply = reply.rstrip(" .") + f". I'm not sure about {nums}, so please check it."
            else:
                reply = _with_note(reply, True)
        else:
            reply = f"It says: {r.text}"
        self._show(now, t["blocks"], block, "READ")
        self._answer(now, reply)

    def _answer(self, now, reply):
        self.pending, self.pending_since = OCRMessage(reply, True, "ocr:answer"), now
        self.state, self.mode = OFF, None
        self.image = None

    def close(self):
        self.stop()
        try:
            self.engine.close()
        except Exception:
            pass
