"""Shared test helpers: synthetic camera-like images with known text, and a fake engine."""
import cv2
import numpy as np

from ocr.ocr_engine import EngineResult, Word


def scene(lines, w=640, h=480, scale=1.0, dark=False, gradient=False, blur=0, noise=8, seed=1):
    """A 640x480 'webcam' frame: textured background with a card of printed text on it."""
    rng = np.random.default_rng(seed)
    img = cv2.GaussianBlur(rng.integers(60, 140, (h, w, 3)).astype(np.uint8), (9, 9), 0)
    x0, y0, pw, ph = 140, 120, 360, 60 + 50 * len(lines)
    bg, fg = ((30, 30, 30), (240, 240, 240)) if dark else ((235, 235, 230), (20, 20, 20))
    cv2.rectangle(img, (x0, y0), (x0 + pw, y0 + ph), bg, -1)
    for i, t in enumerate(lines):
        cv2.putText(img, t, (x0 + 15, y0 + 55 + 50 * i), cv2.FONT_HERSHEY_SIMPLEX, scale, fg, 2, cv2.LINE_AA)
    if gradient:   # one side of the card in shadow
        img = np.clip(img * np.linspace(0.45, 1.1, w)[None, :, None], 0, 255).astype(np.uint8)
    img = np.clip(img.astype(int) + rng.normal(0, noise, img.shape), 0, 255).astype(np.uint8)
    if blur:
        img = cv2.GaussianBlur(img, (blur, blur), 0)
    return img


def words(*specs):
    """words(("EXIT", 95), ("x", 30), ...) -> Word list, one line per call argument group."""
    out = []
    for i, (text, conf, *line) in enumerate(specs):
        ln = line[0] if line else 1
        out.append(Word(text, conf, 1, 1, ln, i + 1, 10 * i, 0, 10, 10))
    return out


class FakeEngine:
    """Stands in for Tesseract: finishes after `delay` seconds of the fake clock."""

    def __init__(self, clock, result_words=None, delay=0.5, available=True, error=None):
        self.clock = clock
        self.result_words = result_words or []
        self.delay = delay
        self._available = available
        self.error = error
        self.started_at = None
        self.starts = 0
        self.cancels = 0
        self.closed = False
        self.cmd = "/fake/tesseract"

    @property
    def available(self):
        return self._available

    @property
    def busy(self):
        return self.started_at is not None

    def start(self, image, psm=None):
        assert image is not None and image.ndim == 2, "engine must get a preprocessed gray image"
        self.started_at = self.clock()
        self.starts += 1
        self.psms = getattr(self, "psms", []) + [psm]

    def poll(self):
        if self.started_at is None or self.clock() - self.started_at < self.delay:
            return None
        self.started_at = None
        if self.error:
            return EngineResult(error=self.error)
        return EngineResult(list(self.result_words), seconds=self.delay)

    def cancel(self):
        if self.started_at is not None:
            self.cancels += 1
        self.started_at = None

    def close(self):
        self.closed = True


class Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t
