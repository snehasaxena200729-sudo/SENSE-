"""
continuity.py - Keep ONE identity per physical object while you move, and warn
when you are really approaching it (foundation fix 2 + 3).

WHY BYTETRACK ALONE WAS NOT ENOUGH
----------------------------------
ByteTrack matches boxes between frames by overlap (IoU) with a constant-
velocity prediction. Walking towards a chair breaks both assumptions: the box
GROWS quickly and your head sway SHIFTS everything. Overlap drops, the match
fails, and the chair gets a new id. Everything keyed by that id (approach
history, alert keys) restarted, so alerts were missed or repeated. Doors were
worse: the door model ran without any tracking at all.

THE APPROACH (gated nearest-neighbour association with motion compensation)
---------------------------------------------------------------------------
For every remembered object we PREDICT where its box should be now:
  * shift it by the camera motion measured with optical flow (the whole
    image moves when you turn or sway), and
  * scale it by the object's own recent growth rate (walking towards it).
A new detection of the SAME class is matched to the object whose prediction it
fits best, if it passes gates on position and size. The ByteTrack id is used as
a strong hint (same id -> same object), never as the only evidence.

An object that is not detected for a moment is "coasting": it keeps its
identity for up to MAX_COAST_S, but it is NOT drawn and NOT reported as visible
(no stale boxes). If it is detected again in time, it continues with the same
id, so a door is not announced twice.

PROXIMITY: DISTANCE TREND, NOT BOX GROWTH
-----------------------------------------
Box size changes for many reasons (turning, tilting, partial occlusion, a box
cut off by the frame edge). Instead, each object keeps a history of distance
estimates from distance.py and fits a straight line to log(distance) over the
last few seconds:

    log d(t) ~= a + b t        b < 0  means the distance is shrinking

Using LOG distance has a useful property: an unknown size or focal-length
error multiplies every estimate by the same factor, which only shifts `a`, so
the TREND b is unaffected. We call it "approaching" only if the fitted change
is larger than the scatter of the estimates (noise), over at least
MIN_TREND_SPAN_S. One alert per approach: after warning, the object is
disarmed until it is clearly further away again (hysteresis), so the same
chair is never announced every frame.

Pure logic: numpy only, runs in the main loop, no thread.
"""

import math
from collections import deque

import numpy as np

import distance as dist_mod
import spatial

MAX_COAST_S = 2.0          # keep identity this long without a detection...
MIN_MISSES = 3             # ...and at least this many detector runs that missed it
                           # (a stalled loop - model loading, a slow frame - is not "unseen")
POS_GATE = 0.75            # max centre jump, as a fraction of the predicted box size
SCALE_GATE = 0.6           # max |log(size ratio)| vs the prediction
SCALE_RATE_ALPHA = 0.4     # smoothing of the per-object growth rate

TREND_WINDOW_S = 3.0
MAX_GAP_S = 0.6            # a longer gap between sightings starts a new trend
MIN_TREND_SAMPLES = 4
MIN_TREND_SPAN_S = 1.0
MIN_CLOSING_RATE = 0.08    # log-distance per second (about 8% closer per second)
NEAR_M = 1.5               # warn when an approaching object is within this...
LOOKAHEAD_S = 1.0          # ...or will be within it in about a second at the current pace
                           # (speech takes ~1.5 s, and detectors often lose an object
                           # in the last metre - the door model does - so warn early)
REARM_M = 2.8              # warn again only after it was beyond this
AHEAD_DEG = 20.0           # only objects roughly in front of you
RECENT_APPROACH_S = 2.0    # a cut-off box still counts if it was approaching just before


def _center(b):
    return (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0


def _size(b):
    return max(1.0, b[2] - b[0]), max(1.0, b[3] - b[1])


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


class TrackedObject:
    def __init__(self, oid, cls, box, now, bt_id=None):
        self.oid, self.cls = oid, cls
        self.box = tuple(float(v) for v in box)
        self.first_seen = self.last_seen = now
        self.last_update = now
        self.hits = 1
        self.misses = 0                     # detector runs since the last match
        self.bt_id = bt_id
        self.scale_rate = 0.0               # d log(size) / dt
        self.samples = deque(maxlen=60)     # (t, uncut axis, size px, distance m or None)
        self.anchor = None                  # absolute distance + box sizes when it was measured
        self.last_d = None
        self.filled = False
        self.armed = True                   # may give a proximity alert
        self.last_approach_t = -1e9
        self.closing_rate = 0.0             # last fitted closing rate (log-distance per second)
        self.announced = False

    # --- prediction ----------------------------------------------------
    def predicted_box(self, now, shift):
        dt = max(0.0, now - self.last_seen)
        k = math.exp(max(-1.0, min(1.0, self.scale_rate * dt)))
        cx, cy = _center(self.box)
        w, h = _size(self.box)
        cx, cy = cx + shift[0], cy + shift[1]
        return (cx - w * k / 2, cy - h * k / 2, cx + w * k / 2, cy + h * k / 2)

    # --- distance and approach -------------------------------------------
    def add_measure(self, now, box, frame_w, frame_h, est):
        """Record one sighting for the distance trend.

        Close up, objects on the floor are cut off at the bottom of the frame
        (a chest-height camera sees the floor only beyond ~3.5 m), so an
        absolute size-based distance is often unavailable. But the size along
        an axis that is NOT cut off still scales as 1 / distance, whatever the
        object's real size. So we
          * anchor: remember an absolute distance with the box sizes seen then,
          * propagate: distance_now = anchor_distance * anchor_size / size_now,
            along an uncut axis (switching axis re-anchors),
          * and fit the approach trend to log(size) along that axis.
        Turning the head moves a box but does not change its size, so it is
        not mistaken for approaching."""
        edges = dist_mod.clipped_edges(box, frame_w, frame_h)
        w, h = box[2] - box[0], box[3] - box[1]
        lateral = edges & {"left", "right"}
        # Cut off at ONE side edge: the object is entering or leaving the view
        # (usually you are turning). Then neither size means anything - the
        # visible part changes as it slides in - so nothing is measured.
        # Cut off at the top/bottom only (walking up to it): the width is fine.
        # Cut off at BOTH side edges: wider than the view, i.e. right in front.
        sizes = {"w": w if not lateral and w > 8 else None,
                 "h": h if not lateral and not edges & {"top", "bottom"} and h > 8 else None}
        axis = "w" if sizes["w"] else ("h" if sizes["h"] else None)
        d = None
        if est is not None and est.metres is not None:
            d = est.metres
            self.anchor = {"d": d, **sizes}
        elif axis and self.anchor is not None:
            if self.anchor.get(axis):
                d = self.anchor["d"] * self.anchor[axis] / sizes[axis]
            elif self.last_d is not None:            # axis changed: re-anchor on it
                self.anchor = {"d": self.last_d, **sizes}
                d = self.last_d
        self.filled = lateral == {"left", "right"}    # wider than the view
        if d is not None:
            self.last_d = d
        self.samples.append((now, axis, sizes[axis] if axis else None, d))

    def trend(self, now):
        """(closing rate per second, noise, span) or None.

        Fitted to log(size) along the current uncut axis: if d ~ 1/size, then
        d log(d)/dt = -d log(size)/dt, independent of the real size."""
        pts = []
        for t, axis, size, _ in reversed(self.samples):
            if now - t > TREND_WINDOW_S:
                break
            if axis is None or (pts and (axis != pts[-1][2] or pts[-1][0] - t > MAX_GAP_S)):
                break                    # one axis, one unbroken run of usable sizes per fit
                                         # (reappearing bigger after a gap is not a measured approach)
            pts.append((t, size, axis))
        if len(pts) < MIN_TREND_SAMPLES:
            return None
        pts.reverse()
        t = np.array([p[0] for p in pts]) - pts[0][0]
        if t[-1] < MIN_TREND_SPAN_S:
            return None
        y = np.log(np.array([p[1] for p in pts]))
        slope, intercept = np.polyfit(t, y, 1)
        noise = float(np.std(y - (slope * t + intercept)))
        return float(slope), noise, float(t[-1])

    def approaching(self, now):
        tr = self.trend(now)
        if tr is None:
            return False
        rate, noise, span = tr
        # the fitted change must clearly exceed the scatter of the measurements
        return rate >= MIN_CLOSING_RATE and rate * span > 2.0 * noise + 0.05

    def current_distance(self, now):
        """Best current guess in metres (None if never measurable)."""
        ds = [d for t, _, _, d in list(self.samples)[-3:] if d is not None and now - t <= 1.0]
        if ds:
            return float(np.median(ds))
        return self.last_d if self.filled and self.last_d is not None else None


class ObjectContinuity:
    """Persistent identities for one detection source (COCO model or door model)."""

    def __init__(self, classes, source="obj"):
        self.classes = set(classes)
        self.source = source
        self.objects = {}        # oid -> TrackedObject
        self._next = 1
        self.last_update = None

    def _new(self, cls, box, now, bt_id):
        obj = TrackedObject(self._next, cls, box, now, bt_id)
        self.objects[self._next] = obj
        self._next += 1
        return obj

    def update(self, detections, now, shift=(0.0, 0.0), frame_size=(1280, 720)):
        """detections: list of dicts with keys
              cls, box (x1, y1, x2, y2), conf, bt_id (or None),
              can_create (bool: confident enough to start a NEW object),
              est (distance.Estimate or None)
        shift: camera image motion (dx, dy) in px since the previous update.

        Returns (matched, created): lists of (TrackedObject, detection).
        """
        # forget objects unseen for too long BEFORE matching, so an old identity
        # is never revived by an unrelated detection after a long gap
        for oid in [o for o, obj in self.objects.items()
                    if now - obj.last_seen > MAX_COAST_S and obj.misses >= MIN_MISSES]:
            del self.objects[oid]
        dets = [d for d in detections if d["cls"] in self.classes]
        pairs = []
        for oid, obj in self.objects.items():
            pred = obj.predicted_box(now, shift)
            pw, ph = _size(pred)
            pcx, pcy = _center(pred)
            for j, d in enumerate(dets):
                if d["cls"] != obj.cls:
                    continue
                if d.get("bt_id") is not None and d["bt_id"] == obj.bt_id:
                    pairs.append((-1.0, oid, j))          # same ByteTrack id: strong hint
                    continue
                cx, cy = _center(d["box"])
                w, h = _size(d["box"])
                jump = math.hypot(cx - pcx, cy - pcy) / max(pw, ph)
                scale = abs(math.log(max(w, h) / max(pw, ph)))
                if jump > POS_GATE or scale > SCALE_GATE:
                    continue
                cost = jump + 0.5 * scale + 0.5 * (1.0 - iou(pred, d["box"]))
                pairs.append((cost, oid, j))
        pairs.sort()
        used_o, used_d, matched = set(), set(), []
        for cost, oid, j in pairs:
            if oid in used_o or j in used_d:
                continue
            used_o.add(oid)
            used_d.add(j)
            obj, d = self.objects[oid], dets[j]
            dt = max(1e-3, now - obj.last_seen)
            old_size = max(_size(obj.box))
            new_size = max(_size(d["box"]))
            rate = math.log(new_size / old_size) / dt
            obj.scale_rate = (1 - SCALE_RATE_ALPHA) * obj.scale_rate + SCALE_RATE_ALPHA * rate
            obj.box = tuple(float(v) for v in d["box"])
            obj.last_seen = now
            obj.hits += 1
            obj.misses = 0
            if d.get("bt_id") is not None:
                obj.bt_id = d["bt_id"]
            obj.add_measure(now, obj.box, frame_size[0], frame_size[1], d.get("est"))
            matched.append((obj, d))
        for oid, obj in self.objects.items():
            if oid not in used_o:
                obj.misses += 1
        created = []
        for j, d in enumerate(dets):
            if j in used_d or not d.get("can_create", True):
                continue
            obj = self._new(d["cls"], d["box"], now, d.get("bt_id"))
            obj.add_measure(now, obj.box, frame_size[0], frame_size[1], d.get("est"))
            created.append((obj, d))
        self.last_update = now
        return matched, created

    def visible(self, now, max_age_s=0.0):
        """Objects detected in the latest update (no coasting ghosts)."""
        return [o for o in self.objects.values() if o.last_seen >= (self.last_update or now) - max_age_s]

    def proximity_alerts(self, now, frame_w):
        """[(TrackedObject, distance words)] for objects you are walking into.

        Fires once per approach (then disarmed until the object is beyond
        REARM_M again, or forgotten)."""
        out = []
        for obj in self.objects.values():
            if obj.last_seen < (self.last_update or now):
                continue                                  # not seen right now
            d = obj.current_distance(now)
            if d is None:
                continue
            if not obj.armed:
                if d > REARM_M:
                    obj.armed = True
                continue
            cx, _ = _center(obj.box)
            if abs(spatial.pixel_to_angle(cx, frame_w)) > AHEAD_DEG:
                continue
            if obj.approaching(now):
                obj.last_approach_t = now
                obj.closing_rate = min(1.0, obj.trend(now)[0])   # cap: a noisy fit must not cry wolf from afar
            recently = now - obj.last_approach_t <= RECENT_APPROACH_S
            soon = d * math.exp(-obj.closing_rate * LOOKAHEAD_S) if recently else d
            if recently and (soon <= NEAR_M or obj.filled):
                obj.armed = False
                words = ("very close" if obj.filled or d <= dist_mod.ARM_REACH_M
                         else dist_mod.phrase(dist_mod.Estimate(d, d * 0.8, d * 1.2)))
                out.append((obj, words))
        return out
