"""
heading_fusion.py - keep the turn estimate right when you turn quickly.

WHY
---
SENSE knows how far you have turned (the "heading", used for "turn about 90
degrees to your left" and "behind you") by tracking corner points between
frames with Lucas-Kanade optical flow (sense_master.py, SUB-SYSTEM A). LK is
precise for slow motion, but it searches only a small window around each
point. In a quick turn the picture jumps hundreds of pixels per frame, the
points are lost or matched wrongly, and the turn is under-counted. Measured on
a simulated 180 degree turn at 12 fps:

    turn takes   4 s    2 s    1.5 s   1 s
    LK says      180    175    144     136   degrees
    + this       180    179    181     179

On a plain wall LK may find no corners at all (fewer than 4 points), and then
the turn is not counted in that frame.

HOW
---
Phase correlation (cv2.phaseCorrelate) compares two whole frames in the
frequency domain (Fourier shift theorem: a shift in the image is a phase
change in its spectrum) and returns the single global shift, up to half the
image width, in one step. It is less precise than LK for small moves and can
be pulled by a large moving thing, so it is used only as a fallback:

    LK confident that the camera is still   -> LK   (a passer-by can't fake a turn)
    LK failed (too few points)              -> phase correlation, if its peak is clear
    big move and the two disagree           -> phase correlation, if its peak is clear
    otherwise                               -> LK   (exactly as before)

It also keeps the TOTAL turn without wrapping (turn right 270 degrees and it
says 270, not -90), and the total at the moment each thing was last seen, so
SENSE can say "You have turned about 270 degrees to your right since I last
saw Sneha". Directions themselves still come from spatial.py.

It runs on a small grey copy (320 px wide, upper half of the picture, the same
"background" region LK uses), about 1 ms per frame. No thread, no model.
"""

import cv2
import numpy as np


class TurnEstimator:
    SMALL_W = 320
    TOP_FRACTION = 0.5        # the upper part: walls, shelves - not people's legs or the floor
    MIN_RESPONSE = 0.08       # phase-correlation peak strength we trust (0..1)
    BIG_MOVE_PX = 30.0        # full-resolution pixels per frame
    STILL_PX = 2.0
    STILL_POINTS = 20
    MIN_LK_POINTS = 8
    DISAGREE = 0.25
    STRONG_RESPONSE = 0.2

    def __init__(self):
        self.prev = None
        self.window = None
        self.last_source = "lk"
        self.total_deg = 0.0      # unwrapped: +270 means turned right 3/4 of a circle
        self.seen_at_total = {}   # name -> total_deg when it was last seen (this run only)

    # ---- total turn, for "you have turned about 270 degrees" ------------
    def add_turn(self, deg):
        """Count a heading change (degrees, + = right). Returns it unchanged."""
        self.total_deg += deg
        return deg

    def note_seen(self, names):
        for n in names:
            self.seen_at_total[n.lower()] = self.total_deg

    def forget(self, name):
        self.seen_at_total.pop(name.lower(), None)

    def turned_since(self, name):
        """Degrees turned since `name` was last seen (+ right), or None."""
        t = self.seen_at_total.get((name or "").lower())
        return None if t is None else self.total_deg - t

    def since_sentence(self, name, spoken, min_deg=25.0):
        """'You have turned about 270 degrees to your right since I last saw Sneha. '
        or '' when unknown or hardly turned."""
        d = self.turned_since(name)
        if d is None or abs(d) < min_deg:
            return ""
        n = int(round(abs(d) / 10.0) * 10)
        side = "right" if d > 0 else "left"
        return f"You have turned about {n} degrees to your {side} since I last saw {spoken}. "

    def _small(self, gray):
        h, w = gray.shape[:2]
        top = gray[: int(h * self.TOP_FRACTION)]
        sh = max(16, int(round(top.shape[0] * self.SMALL_W / float(w))))
        return cv2.resize(top, (self.SMALL_W, sh), interpolation=cv2.INTER_AREA).astype(np.float32)

    def update(self, gray, lk_dx, lk_points):
        """gray: this frame (full size). lk_dx: LK's median x shift in pixels
        (0 if it failed). lk_points: how many points LK tracked (0 if it failed).
        Returns the x shift in full-resolution pixels to use for the heading."""
        small = self._small(gray)
        scale = gray.shape[1] / float(self.SMALL_W)
        prev, self.prev = self.prev, small
        if prev is None or prev.shape != small.shape:
            self.last_source = "lk"
            return float(lk_dx)
        if lk_points >= self.STILL_POINTS and abs(lk_dx) < self.STILL_PX:
            self.last_source = "lk"
            return float(lk_dx)
        if self.window is None or self.window.shape != small.shape:
            self.window = cv2.createHanningWindow((small.shape[1], small.shape[0]), cv2.CV_32F)
        (sx, _), response = cv2.phaseCorrelate(prev, small, self.window)
        pc_dx = sx * scale
        if response >= self.MIN_RESPONSE:
            if lk_points < self.MIN_LK_POINTS:
                self.last_source = "phase"
                return float(pc_dx)
            if abs(pc_dx) > self.BIG_MOVE_PX and abs(pc_dx - lk_dx) > max(20.0, self.DISAGREE * abs(pc_dx)):
                self.last_source = "phase"
                return float(pc_dx)
            # LK's points got stuck on something moving (a person walking past
            # a still camera) while the whole picture clearly did not move
            if response >= self.STRONG_RESPONSE and abs(pc_dx) < self.STILL_PX and abs(lk_dx) > 3 * self.STILL_PX:
                self.last_source = "phase"
                return float(pc_dx)
        self.last_source = "lk"
        return float(lk_dx)
