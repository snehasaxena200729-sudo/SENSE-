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

import spatial

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


class SceneHazardDetector:
    """Doors and stairs - things YOLO's standard (COCO) model has no class for.

    Uses a second, small offline model trained on Google's Open Images V7
    dataset (yolov8n-oiv7.pt, 600 classes, among them "Door" and "Stairs").
    Ultralytics downloads it once on the first run, like the Whisper model.

    Cost control: it runs only every EVERY_N frames, on a smaller image, and
    only for the hazard classes, so the main loop stays fast. No thread.
    Identity, "door ahead" announcements and proximity warnings are handled by
    continuity.py (one identity per door while you walk towards it).
    """
    HAZARD_NAMES = ("door", "stairs")
    HANDLE = "door handle"  # audit fix 3: the most distinctive part of a plain white door
    EVERY_N = 5
    IMG_SIZE = 640          # field test 3: 480 px missed doors; the model was trained at 640
    MIN_CONF = 0.25         # a door or stairs box on its own must score at least this
    WEAK_DOOR_CONF = 0.10   # a weaker door box counts only if a handle sits inside it
    HANDLE_CONF = 0.30      # a handle on its own -> "door handle" (honest: that is what was seen)
    VISIBLE_S = 2.5         # how long a sighting counts as "visible now"

    def __init__(self, model):
        self.model = model
        self.class_ids = {i: n.lower() for i, n in model.names.items()
                          if n.lower() in self.HAZARD_NAMES + (self.HANDLE,)}
        self.boxes = []          # last result, for the HUD: (name, x1, y1, x2, y2)
        self.seen = {}           # name -> dict(time, angle, frac, conf, first)
        self._last_log = 0.0

    def visible(self, now):
        """{name: angle} for doors/stairs seen in the last VISIBLE_S seconds."""
        return {n: d["angle"] for n, d in self.seen.items() if now - d["time"] <= self.VISIBLE_S}

    @staticmethod
    def _inside(handle, door, slack=0.15):
        """Is the handle's centre inside the door box (with a little slack)?"""
        hx, hy = (handle[0] + handle[2]) / 2, (handle[1] + handle[3]) / 2
        dw, dh = door[2] - door[0], door[3] - door[1]
        return (door[0] - slack * dw <= hx <= door[2] + slack * dw
                and door[1] - slack * dh <= hy <= door[3] + slack * dh)

    def filter_detections(self, raw):
        """raw: [(name, conf, (x1, y1, x2, y2))] straight from the model.

        Returns accepted [(name, conf, box)]. Rules (audit fix 3):
          * door / stairs at MIN_CONF or more: accepted
          * a weaker door (WEAK_DOOR_CONF..MIN_CONF): accepted only if a door
            handle (HANDLE_CONF+) sits inside it - two independent clues agree
          * a confident handle with no accepted door around it: reported as a
            "door handle", because that is all the model actually saw
        """
        handles = [(c, b) for n, c, b in raw if n == self.HANDLE and c >= self.HANDLE_CONF]
        out, doors = [], []
        for name, conf, box in raw:
            if name == self.HANDLE:
                continue
            has_handle = name == "door" and any(self._inside(b, box) for _, b in handles)
            if conf >= self.MIN_CONF or (has_handle and conf >= self.WEAK_DOOR_CONF):
                out.append((name, conf, box))
                if name == "door":
                    doors.append(box)
        for conf, box in handles:
            if not any(self._inside(box, d) for d in doors):
                out.append((self.HANDLE, conf, box))
        return out

    def detect(self, frame, frame_count, now):
        """Run the door/stairs model on every EVERY_N-th frame.

        Returns None on frames where it did not run, else the accepted
        detections [(name, conf, (x1, y1, x2, y2))]. Announcements and
        proximity alerts are made by continuity.py, which keeps one identity
        per door while you walk towards it (foundation fix 2)."""
        if not self.class_ids or frame_count % self.EVERY_N:
            return None
        res = self.model.predict(frame, classes=list(self.class_ids), conf=self.WEAK_DOOR_CONF,
                                 imgsz=self.IMG_SIZE, verbose=False)
        frame_w = frame.shape[1]
        raw = []
        for box in (res[0].boxes if res and res[0].boxes is not None else []):
            name = self.class_ids.get(int(box.cls[0]))
            if name is None:
                continue
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
            conf = float(box.conf[0]) if getattr(box, "conf", None) is not None else 0.0
            raw.append((name, conf, (x1, y1, x2, y2)))
        accepted = self.filter_detections(raw)
        self.boxes = [(n, *b) for n, _, b in accepted]
        best = {}
        for name, conf, (x1, y1, x2, y2) in accepted:
            angle = spatial.pixel_to_angle((x1 + x2) / 2, frame_w)
            if name not in best or abs(angle) < abs(best[name][0]):
                best[name] = (angle, conf)
        for name, (angle, conf) in best.items():      # for "where is the door?"
            self.seen[name] = {"time": now, "angle": angle, "conf": conf}
        if raw and now - self._last_log >= 1.0:   # diagnostics, at most once a second
            self._last_log = now
            kept = {(n, b) for n, _, b in accepted}
            print("🚪 Candidates: " + ", ".join(
                f"{n} {c:.2f}{'' if (n, b) in kept else ' (rejected)'}" for n, c, b in raw))
        return accepted
