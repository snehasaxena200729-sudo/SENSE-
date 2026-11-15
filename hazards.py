"""
hazards.py - Sudden-obstacle warnings (field-test fix for "no alert for obstacles")

Two complementary detectors, both cheap enough for the main loop (no threads):

1) LoomingDetector - CLASS-AGNOSTIC, works for walls, doors, boxes, a hand,
   anything with some texture, even if YOLO has no name for it.

   When something comes straight at the camera, its image EXPANDS. In the
   optical-flow field that expansion shows up as positive DIVERGENCE
   (flow vectors pointing outward from the centre). For a surface approaching
   head-on, a classic result from vision research ("time to contact", Lee 1976):

        divergence = du/dx + dv/dy  ~=  2 / TTC        (TTC in frames)

   so   TTC_seconds ~= 2 / divergence * seconds_per_frame.

   We compute dense Farneback optical flow on a small grey crop of the centre
   of the image (fast: ~160x120 pixels), take the mean divergence, and warn
   when the time-to-contact stays below TTC_ALERT_S for a few frames in a row.
   Turning the head adds the SAME flow everywhere (a translation), and the
   divergence of a constant field is zero, so ordinary panning is ignored.

2) approach_warning() - for objects YOLO DOES recognise (any of its 80 classes,
   not only SENSE's whitelist): warn when a box ahead grows quickly (looming)
   or suddenly fills a large part of the view.

Limitations: blank walls with no texture give no flow; sudden lighting changes
can fake flow; it is an aid, never a safety guarantee.
"""

from collections import deque

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None


class LoomingDetector:
    CROP_W, CROP_H = 160, 120     # analysed size of the centre region
    CENTRE_FRACTION = 0.6         # middle 60% of width/height = "ahead"
    TTC_ALERT_S = 1.2             # warn when contact is closer than this
    FRAMES_NEEDED = 3             # consecutive frames (debounce)
    MIN_FLOW_PX = 0.3             # ignore when almost nothing moves (noise)
    MAX_DT = 0.25                 # skip if the loop stalled (dt unreliable)

    def __init__(self):
        self.prev = None
        self.streak = 0
        self.last_ttc = None

    def _crop(self, gray):
        h, w = gray.shape[:2]
        cw, ch = int(w * self.CENTRE_FRACTION), int(h * self.CENTRE_FRACTION)
        x0, y0 = (w - cw) // 2, (h - ch) // 2
        return cv2.resize(gray[y0:y0 + ch, x0:x0 + cw], (self.CROP_W, self.CROP_H))

    @staticmethod
    def divergence(flow):
        du_dx = np.gradient(flow[..., 0], axis=1)
        dv_dy = np.gradient(flow[..., 1], axis=0)
        # trim borders where Farneback is least reliable
        return float(np.mean((du_dx + dv_dy)[8:-8, 8:-8]))

    def update(self, gray, dt):
        """Feed one grey frame. Returns time-to-contact in seconds when an
        alert should fire, else None."""
        if cv2 is None:
            return None
        small = self._crop(gray)
        prev, self.prev = self.prev, small
        if prev is None or dt <= 0 or dt > self.MAX_DT:
            self.streak = 0
            return None
        flow = cv2.calcOpticalFlowFarneback(prev, small, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        magnitude = float(np.median(np.linalg.norm(flow, axis=2)))
        div = self.divergence(flow)
        ttc = (2.0 / div) * dt if div > 1e-4 else None
        self.last_ttc = ttc
        if ttc is not None and ttc < self.TTC_ALERT_S and magnitude > self.MIN_FLOW_PX:
            self.streak += 1
        else:
            self.streak = 0
        if self.streak >= self.FRAMES_NEEDED:
            self.streak = 0
            return ttc
        return None


class ApproachTracker:
    """Box-growth (looming) check for YOLO detections of ANY class."""
    WINDOW_S = 0.6
    MIN_AGE_S = 0.25
    GROWTH_ALERT = 1.15        # box 15% taller within ~0.3-0.6 s
    BIG_FRACTION = 0.55        # or the box suddenly fills >55% of frame height
    CENTRE_DEG = 18            # only objects roughly ahead

    def __init__(self):
        self.history = {}      # tid -> deque[(time, height_fraction)]

    def update(self, tid, now, box_h, frame_h, angle_deg, is_person=False):
        hist = self.history.setdefault(tid, deque())
        frac = box_h / float(frame_h)
        hist.append((now, frac))
        while hist and now - hist[0][0] > self.WINDOW_S:
            hist.popleft()
        if abs(angle_deg) > self.CENTRE_DEG:
            return False
        older = [f for t, f in hist if now - t >= self.MIN_AGE_S]
        if not older:
            return False
        growth = frac / max(1e-3, older[0])
        if growth > self.GROWTH_ALERT and (frac > 0.25 or growth > 1.4):
            return True
        # 'suddenly fills the view' rule is skipped for people: someone sitting at
        # the webcam always fills the view and must not trigger warnings (field test 2)
        return not is_person and frac > self.BIG_FRACTION and growth > 1.05

    def prune(self, now, max_age_s=3.0):
        for tid in [t for t, h in self.history.items() if not h or now - h[-1][0] > max_age_s]:
            del self.history[tid]
