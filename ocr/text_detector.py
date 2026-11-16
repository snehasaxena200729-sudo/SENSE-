"""
ocr/text_detector.py - find WHERE text is in a frame, without YOLO.

YOLO only knows 80 object classes, and SENSE only draws the few it trusts, so a
medicine strip, a page or a door sign often gets no box at all. Text detection
does not care what the object is: it looks for text-shaped patterns anywhere.

Two detectors, same output (a list of TextBox):

  EASTDetector   "EAST: An Efficient and Accurate Scene Text detector" (Zhou et
                 al., 2017). A small CNN looks at the image once and, for every
                 4x4 pixel cell, predicts (a) the chance that the cell is inside
                 a word and (b) the distances to that word's four edges plus its
                 rotation angle. Cells with a high score vote for a rotated box;
                 non-maximum suppression (NMS) keeps the best box per word. The
                 rotation angle is what lets us straighten a tilted label later.
                 Runs in OpenCV's dnn module (no new pip package). The 96 MB
                 model downloads once into models/ with a checksum check; after
                 that it is fully offline.

  ClassicDetector  fallback when the model file is missing. Classic computer
                 vision: the morphological gradient (dilate - erode) is bright
                 on letter strokes; Otsu turns it black/white; a wide "closing"
                 smears the letters of a word into one blob; blobs that are
                 wider than tall and mostly filled are kept. No model, a few ms,
                 but it also boxes some textures (keyboards, wood grain).

Neither detector runs every frame: OCRController calls detect() only after the
user asks to read, so YOLO's frame rate is not touched in normal use.
"""

import hashlib
import os
import time
import urllib.request
from collections import namedtuple

import cv2
import numpy as np

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")

# quad: 4x2 float32 corners; rect: axis-aligned (x, y, w, h) clipped to the
# frame; angle: slope of the text line in degrees (0 = level); conf: detector
# score; height: letter height in px (short side of the rotated box).
TextBox = namedtuple("TextBox", "quad rect angle conf height")


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_east_model(config, download=True, log=print):
    """Path of the EAST model if it is present and intact (downloading it once if
    allowed), else None. Never raises: no model just means the classic detector."""
    path = os.path.join(MODEL_DIR, config.east_model_file)
    try:
        if os.path.exists(path) and _sha256(path) == config.east_sha256:
            return path
        if not download:
            return None
        os.makedirs(MODEL_DIR, exist_ok=True)
        log(f"📖 Downloading the text detector {config.east_model_file} (96 MB, one time only)...")
        tmp = path + ".part"
        with urllib.request.urlopen(config.east_url, timeout=30) as r, open(tmp, "wb") as f:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
        if _sha256(tmp) != config.east_sha256:
            os.remove(tmp)
            log("📖 Text detector download rejected (checksum mismatch).")
            return None
        os.replace(tmp, path)
        return path
    except Exception as e:      # offline, disk full, ...: SENSE carries on without it
        log(f"📖 Text detector model not available ({e}).")
        return None


def make_box(rrect, conf, frame_w, frame_h):
    """OpenCV rotated rect ((cx, cy), (w, h), angle) -> TextBox."""
    (cx, cy), (w, h), angle = rrect
    quad = cv2.boxPoints(((cx, cy), (w, h), angle)).astype(np.float32)
    # OpenCV's angle depends on which side it calls "width"; make the long side
    # the reading direction, so the angle is the slope of the text line.
    if h > w:
        w, h = h, w
        angle = angle - 90 if angle > 0 else angle + 90
    while angle > 45:
        angle -= 90
    while angle < -45:
        angle += 90
    x0, y0 = np.clip(quad.min(axis=0), 0, [frame_w - 1, frame_h - 1])
    x1, y1 = np.clip(quad.max(axis=0), 0, [frame_w - 1, frame_h - 1])
    rect = (int(x0), int(y0), int(max(1, x1 - x0)), int(max(1, y1 - y0)))
    return TextBox(quad, rect, float(angle), float(conf), float(h))


class EASTDetector:
    name = "EAST"

    def __init__(self, model_path, config):
        self.config = config
        self.model = cv2.dnn_TextDetectionModel_EAST(model_path)
        self.model.setConfidenceThreshold(config.east_conf)
        self.model.setNMSThreshold(config.east_nms)
        self.model.setInputParams(1.0, tuple(config.east_input_size), (123.68, 116.78, 103.94), True)

    def detect(self, frame):
        h, w = frame.shape[:2]
        img = frame if frame.ndim == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        rrects, confs = self.model.detectTextRectangles(img)
        boxes = [make_box(rr, c, w, h) for rr, c in zip(rrects, confs)]
        return [b for b in boxes if b.height >= 4]


class ClassicDetector:
    name = "classic"

    def __init__(self, config):
        self.config = config

    def detect(self, frame):
        gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        scale = 960.0 / w if w > 960 else 1.0
        small = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA) \
            if scale < 1 else gray
        grad = cv2.morphologyEx(small, cv2.MORPH_GRADIENT,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        _, bw = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        joined = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1)))
        contours, _ = cv2.findContours(joined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        sh, sw = small.shape[:2]
        out = []
        for c in contours:
            x, y, cw, ch = cv2.boundingRect(c)
            if ch < 8 or ch > 0.25 * sh or cw < 1.2 * ch or cw > 0.95 * sw:
                continue
            if cv2.countNonZero(bw[y:y + ch, x:x + cw]) / float(cw * ch) < 0.35:
                continue
            (cx, cy), (rw, rh), a = cv2.minAreaRect(c)
            out.append(make_box(((cx / scale, cy / scale), (rw / scale, rh / scale), a), 0.5, w, h))
        return out


class TextDetector:
    """EAST when its model is there, otherwise the classic detector."""

    def __init__(self, config, download=True, log=print):
        self.config = config
        self.impl = None
        self.last_seconds = 0.0
        path = ensure_east_model(config, download=download, log=log) if config.use_east else None
        if path:
            try:
                self.impl = EASTDetector(path, config)
            except Exception as e:
                log(f"📖 EAST text detector could not load ({e}); using the classic detector.")
        if self.impl is None:
            self.impl = ClassicDetector(config)

    @property
    def name(self):
        return self.impl.name

    def detect(self, frame):
        # Ultralytics switches OpenCV to ONE internal thread (cv2.setNumThreads(0)),
        # which made EAST 3x slower inside SENSE (1.5 s instead of 0.5 s in the
        # cloud test). YOLO is not running at this moment (same main loop), so
        # the detector may use all cores for this one call; the setting is put
        # back afterwards. These are OpenCV's C++ worker threads, not Python
        # threads, so SENSE's two-thread rule is unaffected.
        t0 = time.perf_counter()
        previous = cv2.getNumThreads()
        try:
            cv2.setNumThreads(max(1, cv2.getNumberOfCPUs()))
            boxes = self.impl.detect(frame)
        finally:
            cv2.setNumThreads(previous)
        self.last_seconds = time.perf_counter() - t0
        return boxes
