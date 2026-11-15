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

from .config import OCRConfig
from .image_preprocessing import prepare
from .ocr_engine import TesseractEngine
from .text_filter import RepeatFilter, extract_text

OCRMessage = namedtuple("OCRMessage", "text is_answer key")

OFF, WAITING, READING = "OFF", "WAITING", "READING"
ONCE, CONTINUOUS = "ONCE", "CONTINUOUS"

UNAVAILABLE_REPLY = ("Text reading is not set up. Install Tesseract with: brew install tesseract, "
                     "then restart SENSE.")


def _phrase_regex(phrases):
    # whole words only: "read it" must not match inside "bread item"
    return re.compile(r"\b(" + "|".join(re.escape(p) for p in phrases) + r")\b")


class OCRController:
    def __init__(self, config=None, engine=None, clock=time.time, log=print):
        self.config = config or OCRConfig()
        self.engine = engine if engine is not None else TesseractEngine(self.config, clock=clock)
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
        ctrl = cls(config=config, log=log)
        if ctrl.available:
            log(f"📖 Text reading ready (Tesseract: {ctrl.engine.cmd}). Say 'read this' or press R.")
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
        if mode == CONTINUOUS:
            self.repeats.clear()

    def stop(self):
        self.engine.cancel()
        self.state = OFF
        self.mode = None
        self.best_blurry = None
        self.pending = None

    # ------------------------------------------------------------- per frame
    def feed(self, frame, now, camera_moving=False):
        """Offer the current frame. Cheap no-op unless OCR is waiting for a frame."""
        if self.state != WAITING or now < self.next_read_at:
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
        if self.state == READING:
            result = self.engine.poll()
            if result is not None:
                if result.ok:
                    text = extract_text(result.words, self.config)
                    psm = self.config.psm_modes[self.attempt]
                    self.log(f"📖 OCR {result.seconds:.2f}s (psm {psm}): {len(result.words)} words -> '{text}'")
                    if not text and self._try_next_mode():
                        return None
                    self._finish(now, text)
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

    def _finish(self, now, text, error=False):
        self.image = None
        if self.mode == ONCE:
            if error:
                reply = "Sorry, I could not read that."
            elif text:
                reply = f"It says: {text}"
            else:
                reply = ("I can't find any readable text. Hold the text about 30 centimetres "
                         "from the camera, steady and in good light.")
            self.pending, self.pending_since = OCRMessage(reply, True, "ocr:answer"), now
            self.state, self.mode = OFF, None
            return
        if self.mode == CONTINUOUS:
            if text and not self.repeats.is_repeat(text, now):
                self.repeats.remember(text, now)
                self.pending = OCRMessage(f"Text: {text}", False, "ocr:" + text[:40].lower())
                self.pending_since = now
            self._schedule_next(now)

    def close(self):
        self.stop()
        try:
            self.engine.close()
        except Exception:
            pass
