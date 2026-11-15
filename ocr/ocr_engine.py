"""
ocr/ocr_engine.py - run Tesseract OCR WITHOUT blocking SENSE and WITHOUT a thread.

Why a separate process (and not pytesseract or a thread):
  * SENSE may use at most two Python threads (main loop + voice listener), and
    both are taken. A third "OCR thread" is not allowed.
  * Calling Tesseract directly in the main loop (that is what pytesseract does:
    it starts the same `tesseract` program and WAITS for it) would freeze the
    camera, tracking and hazard warnings for 0.3-1.5 s per read.
  * So we do what SENSE already does for speech with `say`: start the program
    with subprocess.Popen (returns immediately), then check proc.poll() once per
    frame. The operating system runs Tesseract on another core in parallel.

Output goes to a file, not a pipe: a pipe holds only ~64 KB, and a process that
nobody reads from while it writes a long TSV would hang forever.

TSV output = one row per recognised word with its box and a confidence 0-100,
which text_filter.py uses to drop unreliable words.
"""

import os
import shutil
import subprocess
import tempfile
import time
from collections import namedtuple

import cv2

Word = namedtuple("Word", "text conf block par line word_num left top width height")


def find_tesseract(config):
    """Full path of the tesseract program, or None if it is not installed."""
    found = shutil.which(config.tesseract_cmd)
    if found:
        return found
    for path in config.tesseract_search_paths:
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    return None


def parse_tsv(tsv_text):
    """Tesseract TSV -> list of Word (only real words: level 5, non-empty text)."""
    words = []
    lines = tsv_text.splitlines()
    if not lines:
        return words
    header = lines[0].split("\t")
    try:
        idx = {name: header.index(name) for name in
               ("level", "block_num", "par_num", "line_num", "word_num",
                "left", "top", "width", "height", "conf", "text")}
    except ValueError:
        return words
    for row in lines[1:]:
        cols = row.split("\t")
        if len(cols) <= idx["text"]:
            continue
        text = cols[idx["text"]].strip()
        if cols[idx["level"]] != "5" or not text:
            continue
        try:
            conf = float(cols[idx["conf"]])
            words.append(Word(text, conf,
                              int(cols[idx["block_num"]]), int(cols[idx["par_num"]]),
                              int(cols[idx["line_num"]]), int(cols[idx["word_num"]]),
                              int(cols[idx["left"]]), int(cols[idx["top"]]),
                              int(cols[idx["width"]]), int(cols[idx["height"]])))
        except ValueError:
            continue
    return words


class EngineResult:
    """What a finished read produced: words, or an error message for the log."""

    def __init__(self, words=None, error=None, seconds=0.0):
        self.words = words or []
        self.error = error
        self.seconds = seconds

    @property
    def ok(self):
        return self.error is None


class TesseractEngine:
    """One read at a time: start(image) -> poll() each frame -> EngineResult."""

    def __init__(self, config, clock=time.time):
        self.config = config
        self.clock = clock
        self.cmd = find_tesseract(config)
        self.proc = None
        self.started_at = 0.0
        self._workdir = None
        self._out_base = None

    @property
    def available(self):
        return self.cmd is not None

    @property
    def busy(self):
        return self.proc is not None

    def _dir(self):
        if self._workdir is None:
            self._workdir = tempfile.mkdtemp(prefix="sense_ocr_")
        return self._workdir

    def start(self, image, psm=None):
        """Write the image to a temp PNG and launch tesseract. Returns at once."""
        if not self.available:
            raise RuntimeError("tesseract is not installed")
        if self.busy:
            raise RuntimeError("a read is already running")
        img_path = os.path.join(self._dir(), "frame.png")
        self._out_base = os.path.join(self._dir(), "result")
        try:
            os.remove(self._out_base + ".tsv")     # never read a previous result
        except FileNotFoundError:
            pass
        if not cv2.imwrite(img_path, image):
            raise RuntimeError("could not write the OCR image")
        env = dict(os.environ, OMP_THREAD_LIMIT=str(self.config.cpu_threads))
        self.proc = subprocess.Popen(
            [self.cmd, img_path, self._out_base, "-l", self.config.language,
             "--psm", str(psm if psm is not None else self.config.psm_modes[0]), "tsv"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env)
        self.started_at = self.clock()

    def poll(self):
        """None while tesseract is still working, else an EngineResult."""
        if self.proc is None:
            return None
        elapsed = self.clock() - self.started_at
        code = self.proc.poll()
        if code is None:
            if elapsed > self.config.timeout_s:
                self.cancel()
                return EngineResult(error=f"timed out after {elapsed:.1f}s", seconds=elapsed)
            return None
        self.proc = None
        if code != 0:
            return EngineResult(error=f"tesseract exited with code {code}", seconds=elapsed)
        try:
            with open(self._out_base + ".tsv", encoding="utf-8", errors="replace") as f:
                return EngineResult(parse_tsv(f.read()), seconds=elapsed)
        except OSError as e:
            return EngineResult(error=f"no OCR output ({e})", seconds=elapsed)

    def read_blocking(self, image, psm=None, poll_interval_s=0.02):
        """Start + wait. For tests and tools only - SENSE itself never calls this."""
        self.start(image, psm)
        while True:
            result = self.poll()
            if result is not None:
                return result
            time.sleep(poll_interval_s)

    def cancel(self):
        if self.proc is not None:
            try:
                self.proc.kill()
                self.proc.wait(timeout=1)
            except Exception:
                pass
            self.proc = None

    def close(self):
        """Stop a running read and delete the temp folder (called when SENSE quits)."""
        self.cancel()
        if self._workdir is not None:
            shutil.rmtree(self._workdir, ignore_errors=True)
            self._workdir = None
