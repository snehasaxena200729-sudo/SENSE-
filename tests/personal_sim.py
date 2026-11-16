"""
personal_sim.py - a tiny simulated room for testing personal/ without a camera.

Objects sit at fixed world angles (directions in the room). The "camera" has a
heading and a 65 degree view, like spatial.CAMERA_FOV_DEG. Each frame, every
object inside the view becomes a ByteTrack-like track: it keeps its track id
while it stays in view, and gets a NEW id when it comes back (as ByteTrack
does after losing a track). Crops are rendered as small coloured images with
noise and lighting changes, and turned into the SAME colour signature that
sense_master.get_unified_chroma_signature computes (YCrCb Cr/Cb 16x20 hist).
"""

import itertools
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

try:
    import cv2
except ImportError:     # the tests that need it are skipped
    cv2 = None

import spatial  # noqa: E402

FOV = spatial.CAMERA_FOV_DEG


def chroma_signature(crop):
    """Copy of sense_master.get_unified_chroma_signature (sense_master can't be
    imported in a test: it opens the camera at import time)."""
    if crop is None or crop.size == 0:
        return None
    h, w, _ = crop.shape
    if h < 25 or w < 25:
        return None
    resized = cv2.resize(crop, (60, 80))
    ycrcb = cv2.cvtColor(resized, cv2.COLOR_BGR2YCrCb)
    hist = cv2.calcHist([ycrcb], [1, 2], None, [16, 20], [0, 256, 0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist.astype(np.float32)


# Looks: (body BGR, logo BGR). Two objects with the same look are IDENTICAL to
# the colour signature (the hardest case: two plain white cups).
LOOKS = {
    "white": ((235, 235, 235), (235, 235, 235)),
    "white_red_logo": ((235, 235, 235), (40, 40, 210)),
    "red": ((40, 40, 200), (40, 40, 200)),
    "blue": ((200, 90, 30), (200, 90, 30)),
    "green": ((60, 170, 60), (60, 170, 60)),
    "black": ((25, 25, 25), (25, 25, 25)),
    "yellow": ((40, 210, 230), (40, 210, 230)),
    "pink": ((180, 120, 240), (180, 120, 240)),
    "teal": ((160, 160, 30), (160, 160, 30)),
    "orange": ((30, 130, 240), (30, 130, 240)),
}


def render(look, rng, light=1.0, tint=(1.0, 1.0, 1.0), size=(80, 60)):
    body, logo = LOOKS[look]
    h, w = size
    img = np.zeros((h, w, 3), np.float32)
    img[:] = body
    img[h // 3: h // 2, w // 4: 3 * w // 4] = logo           # a logo band
    img[: h // 8] = (200, 200, 200)                           # a bit of background / rim
    img = img * light * np.array(tint, np.float32)
    img += rng.normal(0, 6, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


class SimObject:
    def __init__(self, key, category, look, angle):
        self.key = key            # ground truth identity: "cupA"
        self.category = category
        self.look = look
        self.angle = angle        # world angle, degrees
        self.present = True


class Room:
    """Drives a PersonalMatcher (or PersonalItems) like sense_master does."""

    _tids = itertools.count(1)

    def __init__(self, target, seed=0, heading_noise=0.0, light_jitter=0.08, pool=None):
        self.target = target              # PersonalMatcher or PersonalItems
        self.objects = []
        self.heading = 0.0
        self.now = 1000.0
        self.rng = np.random.default_rng(seed)
        self.heading_noise = heading_noise
        self.light_jitter = light_jitter
        self.track_of = {}                # object key -> current tid (while in view)
        self.object_of = {}               # tid -> object key (ground truth)
        self.labels = []                  # (time, tid, object key, label) for every labelled sighting
        self.drift = 0.0                  # heading estimate error
        self.pool = pool                  # look -> pre-rendered signatures (fast stress runs)

    def add(self, key, category, look, angle):
        o = SimObject(key, category, look, angle)
        self.objects.append(o)
        return o

    def visible(self):
        out = []
        for o in self.objects:
            rel = spatial.relative_angle(o.angle, self.heading)
            if o.present and abs(rel) < FOV / 2 - 2:
                out.append((o, rel))
        return out

    def _sig(self, o):
        if self.pool is not None:
            sigs = self.pool[o.look]
            return sigs[int(self.rng.integers(len(sigs)))]
        return chroma_signature(self._crop(o))

    def _crop(self, o):
        light = 1.0 + self.rng.uniform(-self.light_jitter, self.light_jitter)
        tint = tuple(1.0 + self.rng.uniform(-0.03, 0.03, 3))
        return render(o.look, self.rng, light, tint)

    def step(self, stable=True, dt=0.1, track_switch=None):
        """One YOLO frame. track_switch=(keyA, keyB): ByteTrack swaps their ids."""
        self.now += dt
        vis = self.visible()
        in_view = {o.key for o, _ in vis}
        for key in list(self.track_of):
            if key not in in_view:              # left the view: the track is lost
                del self.track_of[key]
        for o, _ in vis:
            if o.key not in self.track_of:
                tid = next(self._tids)
                self.track_of[o.key] = tid
                self.object_of[tid] = o.key
        if track_switch:
            a, b = track_switch
            if a in self.track_of and b in self.track_of:
                self.track_of[a], self.track_of[b] = self.track_of[b], self.track_of[a]
                self.object_of[self.track_of[a]] = a
                self.object_of[self.track_of[b]] = b
        out = {}
        est_heading = self.heading + self.drift
        for o, rel in vis:
            tid = self.track_of[o.key]
            sig = self._sig(o)
            measured = rel + (self.rng.normal(0, self.heading_noise) if self.heading_noise else 0.0)
            world = spatial.wrap_angle(est_heading + measured)
            label = self.target.label(tid, o.category, sig, world, self.now, stable)
            out[o.key] = label
            if label:
                self.labels.append((self.now, tid, o.key, label))
        self.target.end_frame(self.now, stable)
        return out

    def run(self, n, **kw):
        last = {}
        for _ in range(n):
            last = self.step(**kw)
        return last

    def turn_to(self, heading, steps=6):
        """Turn the camera (frames during the turn are 'CAM MOVING')."""
        start = self.heading
        delta = spatial.wrap_angle(heading - start)
        for k in range(1, steps + 1):
            self.heading = spatial.wrap_angle(start + delta * k / steps)
            self.step(stable=False, dt=0.05)
        self.now += 0.9          # sense_master waits 0.8 s before "stable" again

    def teach(self, name, key):
        """'This is my cup' while `key` is in view (largest = that one)."""
        o = next(x for x in self.objects if x.key == key)
        tid = self.track_of[o.key]
        rel = spatial.relative_angle(o.angle, self.heading)
        others = [(self._sig(x),
                   spatial.wrap_angle(self.heading + self.drift + r), self.track_of.get(x.key))
                  for x, r in self.visible() if x.key != key and x.category == o.category]
        return self.target.teach(name, o.category, self._sig(o), tid,
                                 spatial.wrap_angle(self.heading + self.drift + rel), self.now, others)

    def wrong_labels(self, name, true_key):
        """Every sighting where `name` was given to a different physical object."""
        return [(t, tid, k) for t, tid, k, lab in self.labels if lab == name and k != true_key]

    def label_count(self, name, key):
        return sum(1 for _, _, k, lab in self.labels if lab == name and k == key)
