"""
door_distance.py - "Door approximately 5 feet ahead": door distance that is
measured, smoothed and spoken honestly (door distance enhancement).

WHAT ALREADY EXISTED
--------------------
hazards.py finds doors (Open Images model), continuity.py keeps ONE identity
per door while you move, and distance.py has the pinhole geometry. But the
distance was never spoken, and nothing announced it again as you walked.
This module is the missing last step. It reads the doors continuity.py is
already tracking; it never runs a model and never touches the tracker.

1) ONE MEASUREMENT PER SIGHTING (pinhole camera, same as distance.py)
----------------------------------------------------------------------
        depth Z = f * S / s        f: focal length in px, S: real size (m),
                                   s: size of the box in px
  * HEIGHT cue when the door's top AND bottom are in view. Height does not
    change when you look at the door from an angle, so it wins when present.
  * WIDTH cue when the top or bottom is cut off but neither side is. With a
    laptop camera this is the usual case from about 2 to 9 feet: the camera
    sees ~39 degrees vertically, so a 2 m door only fits beyond ~3 m.
    Width shrinks by cos(angle) when you see the door obliquely, so this cue
    carries an extra +-10% uncertainty.
  * NOTHING when a side is cut off (the door is sliding into or out of view:
    its visible width means nothing), or when the box has a shape no door has
    (partly hidden door, door + cupboard merged): we do not invent a number.
  * Z is depth along the camera axis; the distance to a door off to the side
    is longer: range = Z / cos(horizontal angle).

CALIBRATION. The camera only ever sees the product f x S. Measuring that
product once at a known distance (tools/door_distance_test.py --calibrate)
cancels the focal-length guess, the door-size guess and the "YOLO box
includes the door frame" bias all together. It is stored resolution-
independently as   cal = box_px * Z / frame_width   (metres).

2) SMOOTHING: A KALMAN FILTER ON LOG(DISTANCE)
----------------------------------------------
State x = [log r, v], v = d(log r)/dt. Why log? Box jitter is a PERCENTAGE
error (5% of 2 ft or of 8 ft), and in log space a percentage is a constant,
so one noise setting fits every distance. And walking at a steady speed
towards a door is close to a straight line in log r over a second or two.
  predict:  log r += v * dt ;  uncertainty grows with dt
  update:   blend prediction and measurement by their uncertainties
Plausibility gate: a measurement implying movement faster than 2 m/s (fast
walking is ~1.4 m/s), or more than 3 sigma from the prediction, is rejected.
Three rejections in a row mean the scene really changed: restart the filter.

Alternatives considered: a moving average (simple, but lags while you walk,
so it says 6 ft when you are at 4), a median of the last N (robust, also
lags), a monocular depth network (Depth Anything / MiDaS: 100-300 ms per
frame on a laptop CPU, would slow down the obstacle warnings).

3) RELIABLE OR NOT
------------------
A number is spoken only when: 3+ consistent measurements since the last
restart (after more than 1 s without a usable measurement the filter restarts
and must earn this again), the newest no older than 1 s, the newest three
agreeing within ~20% of each other, the systematic uncertainty (size + focal
length, or the calibration) at most 25%, and the filter has settled.
In 2000 simulated random walks (door widths 0.75-1.0 m, sideways drift,
missed detections, 3% wrong boxes) no spoken number was off by more than
max(1.5 ft, 35%); median error 4% (tests/test_door_distance.py).
Otherwise: "Door detected ahead, but I cannot estimate its distance reliably."

4) WHAT IS SAID
---------------
  first sighting:  "Door detected approximately 8 feet ahead."
  while walking:   "The door is approximately 5 feet away."  only after a
                   real change (2 ft when further than 6 ft, else 1 ft), at
                   most every 2 s, only for a visible door within 25 degrees
                   of ahead, and never right after a "Careful, door" warning.
  wider than view: "very close, less than about 3 feet away" (an upper bound
                   from geometry, not a guess).
These are spoken as P1 because notifier.py only speaks P1 while the camera
moves - and walking is exactly when the updates matter.

Pure logic, no thread, no model, no I/O.
"""

import math
from collections import deque

import distance as dist_mod
import spatial

M_TO_FT = dist_mod.M_TO_FT

DOOR_HEIGHT = dist_mod.OBJECT_AXES["door"][0]     # (2.00 m, +-8%)
DOOR_WIDTH = dist_mod.OBJECT_AXES["door"][1]      # (0.85 m, +-15%) standard single door
CAL_UNC = 0.07              # calibrated: box jitter, door-to-door variation in one house
OBLIQUE_UNC = 0.10          # width cue only: you may be seeing the door at an angle
SHAPE_RANGE = (1.5, 5.0)    # height / width of a whole visible door (oblique -> narrower)
MIN_BOX_PX = 12

MEAS_SIGMA = 0.06           # per-sample jitter of log(distance) (box edges wobble ~5%)
ACCEL_SIGMA = 0.6           # how fast d(log r)/dt may change, per second
MAX_SPEED_MS = 2.0          # faster than this is not walking -> implausible jump
GATE_SIGMA = 3.0
MAX_REJECTS = 3

MIN_CONSISTENT = 3
FRESH_S = 1.0               # newest accepted measurement at most this old
MAX_SYS_UNC = 0.25          # systematic uncertainty allowed for speaking a number
MAX_FILTER_STD = 0.15       # filter's own uncertainty (log units, ~15%)
MAX_SPREAD = 0.2           # newest 3 accepted measurements must agree within ~20%
CARRY_S = 1.5               # coast the estimate this long without a measurement

FIRST_WAIT_S = 1.5          # wait this long for a reliable number before announcing without one
UPDATE_MIN_GAP_S = 2.0     # a spoken update takes ~1.8 s
UPDATE_STEP_FAR_FT = 2.0    # change needed beyond NEAR_STEP_LIMIT_FT
UPDATE_STEP_NEAR_FT = 1.0
NEAR_STEP_LIMIT_FT = 6.0
UPDATE_AHEAD_DEG = 25.0
AFTER_ALERT_S = 4.0         # no update right after a "Careful, door" warning
MAX_SPOKEN_M = 6.0          # ~20 ft: beyond this say "far ahead", no number
DEDUPE_S = 10.0             # same door flickering out and back: do not re-announce
DEDUPE_DEG = 25.0
VISIBLE_S = 2.5             # same as hazards.SceneHazardDetector.VISIBLE_S


def load_calibration(get_setting):
    """{"w": metres, "h": metres} from the settings table (missing keys = not calibrated)."""
    cal = {}
    for axis in ("w", "h"):
        try:
            v = float(get_setting(f"door_cal_{axis}", "0") or 0)
        except (TypeError, ValueError):
            v = 0.0
        if v > 0:
            cal[axis] = v
    return cal


def calibration_value(box_px, true_range_m, angle_deg, frame_w):
    """What --calibrate stores: box size x depth / frame width (metres)."""
    depth = true_range_m * math.cos(math.radians(angle_deg))
    return box_px * depth / float(frame_w)


def door_direction(angle_deg):
    """'ahead' / 'ahead on your right' / 'to your left' / 'behind you'."""
    a = abs(angle_deg)
    side = "right" if angle_deg > 0 else "left"
    if a < 11:                        # same centre band as spatial.direction_phrase
        return "ahead"
    if a <= UPDATE_AHEAD_DEG:
        return f"ahead on your {side}"
    if a > 135:
        return "behind you"
    return f"to your {side}"


def feet(m):
    return max(1, int(round(m * M_TO_FT)))


class Measure:
    __slots__ = ("metres", "unc", "cue", "reason")

    def __init__(self, metres=None, unc=None, cue="none", reason=""):
        self.metres, self.unc, self.cue, self.reason = metres, unc, cue, reason

    @property
    def ok(self):
        return self.metres is not None


def measure(box, frame_w, frame_h, f_px, focal_unc=dist_mod.FOCAL_UNCERTAINTY, cal=None):
    """One distance measurement (metres, range to the door) from one door box."""
    cal = cal or {}
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    edges = dist_mod.clipped_edges(box, frame_w, frame_h)
    if edges >= {"left", "right"}:
        return Measure(reason="wider than the view")
    if edges & {"left", "right"}:
        return Measure(reason="cut off at the side")
    if w < MIN_BOX_PX or h < MIN_BOX_PX:
        return Measure(reason="too small")
    if not edges & {"top", "bottom"}:
        if not SHAPE_RANGE[0] <= h / w <= SHAPE_RANGE[1]:
            return Measure(reason="not a door shape (partly hidden?)")
        if "h" in cal:
            depth, unc = cal["h"] * frame_w / h, CAL_UNC
        else:
            depth, unc = f_px * DOOR_HEIGHT[0] / h, math.hypot(DOOR_HEIGHT[1], focal_unc)
        cue = "height"
    else:
        if "w" in cal:
            depth, unc = cal["w"] * frame_w / w, math.hypot(CAL_UNC, OBLIQUE_UNC)
        else:
            depth = f_px * DOOR_WIDTH[0] / w
            unc = math.sqrt(DOOR_WIDTH[1] ** 2 + focal_unc ** 2 + OBLIQUE_UNC ** 2)
        cue = "width"
    angle = spatial.pixel_to_angle((x1 + x2) / 2.0, frame_w)
    return Measure(depth / math.cos(math.radians(angle)), unc, cue)


def wider_than_view_bound(frame_w, f_px, cal=None):
    """A door wider than the whole view is closer than this (metres)."""
    if cal and "w" in cal:
        return cal["w"] * (1 + CAL_UNC)
    return f_px * DOOR_WIDTH[0] * (1 + DOOR_WIDTH[1]) / frame_w


class LogRangeFilter:
    """2-state Kalman filter on [log r, d log r / dt] with a plausibility gate."""

    def __init__(self):
        self.x = None            # [log r, v]
        self.P = None            # 2x2 covariance as nested lists
        self.t = None
        self.consistent = 0
        self.rejects = 0
        self.last_accept_t = None

    def reset(self, logr, now):
        self.x = [logr, 0.0]
        self.P = [[MEAS_SIGMA ** 2, 0.0], [0.0, 0.3 ** 2]]
        self.t = now
        self.consistent = 1
        self.rejects = 0
        self.last_accept_t = now

    def predict(self, now):
        """(log r, variance) predicted for `now`, without changing the state."""
        dt = max(0.0, now - self.t)
        (p00, p01), (p10, p11) = self.P
        q = ACCEL_SIGMA ** 2
        logr = self.x[0] + self.x[1] * dt
        var = p00 + dt * (p01 + p10) + dt * dt * p11 + q * dt ** 3 / 3.0
        return logr, var

    def _advance(self, now):
        dt = max(0.0, now - self.t)
        (p00, p01), (p10, p11) = self.P
        q = ACCEL_SIGMA ** 2
        self.x = [self.x[0] + self.x[1] * dt, self.x[1]]
        self.P = [[p00 + dt * (p01 + p10) + dt * dt * p11 + q * dt ** 3 / 3.0, p01 + dt * p11 + q * dt * dt / 2.0],
                  [p10 + dt * p11 + q * dt * dt / 2.0, p11 + q * dt]]
        self.t = now

    def update(self, metres, now):
        """Feed one measurement. Returns True if accepted."""
        z = math.log(max(0.05, metres))
        if self.x is None or now - self.last_accept_t > FRESH_S:
            # nothing usable for a while (door out of view or cut off): the old
            # estimate is stale, so start again and re-earn "reliable" (3 samples)
            # instead of letting a wide, stale prediction swallow a bad box
            self.reset(z, now)
            return True
        dt = max(1e-3, now - self.t)
        pred_logr, pred_var = self.predict(now)
        s = pred_var + MEAS_SIGMA ** 2
        innov = z - pred_logr
        too_fast = abs(metres - math.exp(pred_logr)) > MAX_SPEED_MS * dt + 0.15
        if abs(innov) > GATE_SIGMA * math.sqrt(s) or too_fast:
            self.rejects += 1
            if self.rejects >= MAX_REJECTS:      # it really changed: start again
                self.reset(z, now)
                return True
            return False
        self._advance(now)
        (p00, p01), (p10, p11) = self.P
        k0, k1 = p00 / s, p10 / s
        self.x = [self.x[0] + k0 * innov, self.x[1] + k1 * innov]
        self.P = [[(1 - k0) * p00, (1 - k0) * p01], [p10 - k1 * p00, p11 - k1 * p01]]
        self.consistent += 1
        self.rejects = 0
        self.last_accept_t = now
        return True

    def metres(self, now):
        if self.x is None:
            return None
        return math.exp(self.predict(now)[0])

    def std(self, now):
        return math.sqrt(max(0.0, self.predict(now)[1])) if self.x is not None else float("inf")


class DoorState:
    def __init__(self, oid, now):
        self.oid = oid
        self.first_seen = now
        self.last_seen = now
        self.filter = LogRangeFilter()
        self.last_meas = Measure()
        self.recent = deque(maxlen=MIN_CONSISTENT)   # (time, log metres) of accepted measurements
        self.sys_unc = None
        self.angle = 0.0
        self.world_angle = None
        self.box = None
        self.filled = False
        self.announced = False
        self.last_spoken_m = None
        self.last_spoken_t = -1e9
        self.alert_t = -1e9
        self.n_spoken = 0

    def reliable(self, now):
        f = self.filter
        return (f.x is not None and f.consistent >= MIN_CONSISTENT
                and now - f.last_accept_t <= FRESH_S
                and self.sys_unc is not None and self.sys_unc <= MAX_SYS_UNC
                and f.std(now) <= MAX_FILTER_STD
                and self.agree())

    def agree(self):
        """The newest accepted measurements agree with each other. A single odd box
        that slipped through the filter's gate makes this False for a moment, so
        SENSE says nothing rather than something wrong."""
        if len(self.recent) < MIN_CONSISTENT:
            return False
        logs = [v for _, v in self.recent]
        return max(logs) - min(logs) <= MAX_SPREAD

    def distance(self, now):
        """Smoothed distance in metres, or None if it should not be spoken."""
        return self.filter.metres(now) if self.reliable(now) else None


class DoorRanger:
    """Per-door distance + narration on top of continuity.py's scene_objects."""

    def __init__(self, cal=None):
        self.cal = dict(cal or {})
        self.doors = {}              # continuity oid -> DoorState
        self.last_door = None        # the most recently seen door, kept after it vanishes
        self._last_log = 0.0

    # --- events from sense_master -------------------------------------------
    def on_created(self, obj, now, angle):
        """continuity.py made a new door identity. Announce it (soon) unless it
        is the same door flickering back."""
        st = self.doors.setdefault(obj.oid, DoorState(obj.oid, now))
        st.angle = angle
        # Same door as one seen moments ago at about the same place (it was
        # hidden for longer than continuity.py's 2 s coasting): keep quiet and
        # carry on from what was last said about it.
        last = self.last_door
        if last and last["oid"] != obj.oid and now - last["time"] < DEDUPE_S \
                and abs(angle - last["angle"]) < DEDUPE_DEG and last.get("announced"):
            st.announced = True
            st.last_spoken_m = last.get("spoken_m")
            st.last_spoken_t = now

    def note_alert(self, oid, now):
        """sense_master spoke "Careful, door ..." for this door."""
        st = self.doors.get(oid)
        if st:
            st.alert_t = now
            d = st.filter.metres(now)
            if d is not None:
                st.last_spoken_m = d

    def update(self, objects, now, frame_w, frame_h, f_px, heading=0.0,
               focal_unc=dist_mod.FOCAL_UNCERTAINTY, allow_updates=True):
        """Call after scene_objects.update(). `objects`: scene_objects.objects.
        Returns [(notifier key, sentence)] to push as P1."""
        out = []
        doors_now = {oid: o for oid, o in objects.items() if o.cls == "door"}
        for oid, obj in doors_now.items():
            st = self.doors.setdefault(oid, DoorState(oid, now))
            if obj.last_seen < now:          # coasting: not seen in this run
                continue
            st.last_seen = now
            st.box = obj.box
            st.angle = spatial.pixel_to_angle((obj.box[0] + obj.box[2]) / 2.0, frame_w)
            st.world_angle = spatial.wrap_angle(st.angle + heading)
            m = measure(obj.box, frame_w, frame_h, f_px, focal_unc, self.cal)
            st.last_meas = m
            st.filled = m.reason == "wider than the view"
            if m.ok and st.filter.update(m.metres, now):
                st.sys_unc = m.unc
                if st.filter.consistent == 1:            # the filter restarted
                    st.recent.clear()
                st.recent.append((now, math.log(max(0.05, m.metres))))
            self._remember(st, now)

        for oid, st in list(self.doors.items()):
            text = self._first_announcement(st, now, frame_w, f_px)
            if text:
                out.append(("seen:door", text))
                if self.last_door and self.last_door["oid"] == oid:
                    self.last_door.update(announced=True, spoken_m=st.last_spoken_m)
            elif allow_updates:
                text = self._distance_update(st, now, frame_w, f_px)
                if text:
                    st.n_spoken += 1
                    out.append((f"doordist:{oid}:{st.n_spoken}", text))
                    if self.last_door and self.last_door["oid"] == oid:
                        self.last_door["spoken_m"] = st.last_spoken_m
            if oid not in doors_now and st.announced:
                del self.doors[oid]          # continuity forgot it; last_door keeps the facts
        self._log(now)
        return out

    # --- narration -------------------------------------------------------
    def _first_announcement(self, st, now, frame_w, f_px):
        if st.announced:
            return None
        d = st.distance(now)
        where = door_direction(st.angle)
        if st.filled:
            text = (f"Door detected very close {where}, less than about "
                    f"{feet(wider_than_view_bound(frame_w, f_px, self.cal))} feet away")
        elif d is not None:
            st.last_spoken_m = d
            text = ("Door detected far " + where if d > MAX_SPOKEN_M
                    else f"Door detected approximately {feet(d)} feet {where}")
        elif now - st.first_seen >= FIRST_WAIT_S:
            text = f"Door detected {where}, but I cannot estimate its distance reliably"
        else:
            return None
        st.announced = True
        st.last_spoken_t = now
        return text

    def _distance_update(self, st, now, frame_w, f_px):
        if not st.announced or st.last_seen < now or abs(st.angle) > UPDATE_AHEAD_DEG:
            return None
        if now - st.last_spoken_t < UPDATE_MIN_GAP_S or now - st.alert_t < AFTER_ALERT_S:
            return None
        where = "" if abs(st.angle) < 11 else ", " + door_direction(st.angle)
        if st.filled:
            bound = wider_than_view_bound(frame_w, f_px, self.cal)
            if st.last_spoken_m is not None and st.last_spoken_m <= bound:
                return None
            st.last_spoken_m, st.last_spoken_t = bound, now
            return f"The door is very close, less than about {feet(bound)} feet away{where}"
        d = st.distance(now)
        if d is None or d > MAX_SPOKEN_M:
            return None
        if st.last_spoken_m is not None:
            step = UPDATE_STEP_FAR_FT if min(d, st.last_spoken_m) * M_TO_FT > NEAR_STEP_LIMIT_FT \
                else UPDATE_STEP_NEAR_FT
            if abs(d - st.last_spoken_m) * M_TO_FT < step or feet(d) == feet(st.last_spoken_m):
                return None
        st.last_spoken_m, st.last_spoken_t = d, now
        return f"The door is approximately {feet(d)} feet away{where}"

    # --- questions -----------------------------------------------------------
    def _remember(self, st, now):
        d = st.distance(now)
        prev = self.last_door if self.last_door and self.last_door["oid"] == st.oid else {}
        self.last_door = {"oid": st.oid, "time": now, "world_angle": st.world_angle,
                          "angle": st.angle,
                          "dist_m": d if d is not None else prev.get("dist_m"),
                          "dist_time": now if d is not None else prev.get("dist_time"),
                          "spoken_m": st.last_spoken_m, "announced": st.announced}

    def visible_doors(self, now):
        return [st for st in self.doors.values() if now - st.last_seen <= VISIBLE_S and st.box is not None]

    def describe_visible(self, st, now, frame_w=1280, f_px=None):
        """(distance words or None, direction words) for a visible door.
        e.g. ('approximately 5 feet ahead on your right', 'ahead on your right')."""
        where = door_direction(st.angle)
        if st.filled and f_px:
            bound = feet(wider_than_view_bound(frame_w, f_px, self.cal))
            return f"very close {where}, less than about {bound} feet away", where
        d = st.distance(now)
        if d is None and st.filter.last_accept_t is not None \
                and now - st.filter.last_accept_t <= CARRY_S \
                and st.filter.consistent >= MIN_CONSISTENT \
                and st.sys_unc is not None and st.sys_unc <= MAX_SYS_UNC and st.agree():
            d = st.filter.metres(now)        # briefly carried (e.g. side cut off for a moment)
        if d is None:
            return None, where
        if d > MAX_SPOKEN_M:
            return "far " + where, where
        return f"approximately {feet(d)} feet {where}", where

    def overview_items(self, now, frame_w=1280, f_px=None):
        """For "what do you see?": ['a door approximately 6 feet ahead'] (visible doors only)."""
        items = []
        for st in sorted(self.visible_doors(now), key=lambda s: abs(s.angle)):
            text, where = self.describe_visible(st, now, frame_w, f_px)
            items.append(f"a door {text or where}")
        return items

    def where_answer(self, now, heading=0.0, frame_w=1280, f_px=None):
        """Answer to "where is the door?", or None if no door was ever seen."""
        vis = sorted(self.visible_doors(now), key=lambda s: abs(s.angle))
        if vis:
            text, where = self.describe_visible(vis[0], now, frame_w, f_px)
            if text:
                return f"The door is {text}."
            return f"There is a door {where}, but I cannot estimate its distance reliably."
        last = self.last_door
        if not last:
            return None
        ago = format_ago(now - last["time"])
        rel = (spatial.relative_angle(last["world_angle"], heading)
               if last["world_angle"] is not None else last["angle"])
        where = door_direction(rel)
        d = last.get("dist_m")
        if d is not None and d <= MAX_SPOKEN_M:
            return (f"I can't see the door right now. When I last saw it, {ago}, it was "
                    f"approximately {feet(d)} feet away. From where you face now, it is {where}.")
        return f"I can't see the door right now. I last saw it {ago}. From where you face now, it is {where}."

    # --- HUD and diagnostics ------------------------------------------------
    def hud_labels(self, now):
        """[(box, text)] for the purple door boxes: '~5 ft (est)' or '? ft'."""
        out = []
        for st in self.visible_doors(now):
            d = st.distance(now)
            out.append((st.box, f"~{feet(d)} ft (est, {st.last_meas.cue})" if d is not None
                        else f"dist ? ({st.last_meas.reason or 'settling'})"))
        return out

    def _log(self, now):
        if now - self._last_log < 1.0 or not self.doors:
            return
        parts = []
        for st in self.doors.values():
            if now - st.last_seen > 0.5:
                continue
            m, d = st.last_meas, st.distance(now)
            raw = f"{m.metres * M_TO_FT:.1f} ft ({m.cue}, +-{m.unc * 100:.0f}%)" if m.ok else m.reason
            parts.append(f"door#{st.oid} raw {raw} -> " +
                         (f"{d * M_TO_FT:.1f} ft" if d is not None else "not reliable"))
        if parts:
            self._last_log = now
            print("🚪📏 " + "; ".join(parts))


def format_ago(seconds):
    seconds = int(max(0, seconds))
    if seconds < 60:
        return f"{seconds} seconds ago"
    minutes = seconds // 60
    return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
