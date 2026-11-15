"""
guidance.py - "Find my bottle": spoken left / right / straight guidance

WHAT IT DOES
------------
You say "find my bottle". SENSE then talks you onto it, like a parking sensor
with words:

    "Your bottle is a little to your left."      (you turn left a bit)
    "Straight ahead, a few steps away."          (you walk)
    "Straight ahead, close."
    "Your bottle should be within reach, straight ahead. Reach out slowly."

If the bottle is not in view, SENSE uses its memory instead:

    "I can't see your bottle right now. I last saw it 2 minutes ago, on your
     left, next to the laptop. Turn about 90 degrees to your left."

and keeps updating that turn instruction as you turn (it knows how far you
turned from the optical-flow heading), until the bottle comes into view.

HOW IT WORKS (a small state machine)
------------------------------------
    SEARCHING --target seen--> GUIDING --close enough--> done ("within reach")
        ^                         |
        +----not seen for 2 s-----+         (and a time limit on the whole search)

* DIRECTION = the horizontal angle of the target in the image (pixel -> degrees,
  spatial.pixel_to_angle). Three zones: straight (< 10 deg), "a little" left/right
  (10-25 deg), left/right (> 25 deg).
* HYSTERESIS: to LEAVE a zone the angle must go 4 deg past its border. Without
  it, a bottle sitting right on the 10 deg border would flip between "straight"
  and "a little left" every frame because of detector jitter.
* DEBOUNCE: a NEW instruction must be seen on 2 detections in a row before it
  is spoken (one bad box cannot make SENSE say "right" for a moment).
* RATE LIMIT: at least MIN_GAP_S between cues; the same cue is repeated only
  every REPEAT_S as reassurance. A cue is never spoken over other speech.
* SAFETY FIRST: the caller only lets guidance speak when no hazard alert is
  waiting or was just spoken (see can_speak in sense_master.py). Hazard alerts
  (P1) interrupt a guidance cue; a guidance cue never interrupts anything.
* "WITHIN REACH" needs the distance estimate to say so on 2 detections in a row
  AND the target to be roughly ahead. Distance is only a coarse estimate
  (distance.py), so the words stay coarse: far / a few steps / close / reach.
* WORLD ANGLE (dead reckoning, same as spatial.py): where the target was seen
  is stored as  image angle + heading at that moment. When it is out of view,
  "where is it now" = world angle - heading now. This is how the memory
  fallback can say "keep turning left".

Pure logic, no camera or audio here, so it is unit-tested with a fake clock
(tests/test_guidance.py). It runs inside the main loop: no extra thread.
"""

import spatial

# Direction zones (degrees from the image centre line, + = right)
STRAIGHT_DEG = 10.0
LITTLE_DEG = 25.0
HYSTERESIS_DEG = 4.0

# Distance words (metres). Coarse on purpose - see distance.py
REACH_M = 0.8            # same as distance.ARM_REACH_M
CLOSE_M = 1.5
STEPS_M = 3.0
REACH_MAX_ANGLE = 20.0   # "within reach" only if it is roughly ahead

# Timing (seconds)
MIN_GAP_S = 1.5          # between any two guidance cues
REACH_GAP_S = 0.8        # "within reach" may follow sooner: walking on would bump into it
REPEAT_S = 4.0           # repeat an unchanged cue as reassurance
LOST_S = 2.0             # not detected this long -> "I lost sight of it"
SEARCH_REPEAT_S = 6.0    # how often to repeat a turn instruction while searching
SEARCH_TIMEOUT_S = 45.0  # searching with nothing found -> give up
TOTAL_TIMEOUT_S = 120.0  # whole guidance session
CONFIRM_OBS = 2          # observations needed before a NEW cue is spoken
VIEW_MARGIN_DEG = 8.0    # "it should be in view now" = inside the FOV by this much

IDLE, SEARCHING, GUIDING = "idle", "searching", "guiding"


def direction_zone(angle, previous=None):
    """'straight', 'little_left', 'little_right', 'left' or 'right' with hysteresis.

    previous: the zone we were in, so small jitter around a border is ignored."""
    a = abs(angle)
    side = "right" if angle > 0 else "left"
    if previous == "straight" and a < STRAIGHT_DEG + HYSTERESIS_DEG:
        return "straight"
    if previous in ("little_left", "little_right") and previous.endswith(side) \
            and STRAIGHT_DEG - HYSTERESIS_DEG <= a < LITTLE_DEG + HYSTERESIS_DEG:
        return previous
    if previous in ("left", "right") and previous == side and a >= LITTLE_DEG - HYSTERESIS_DEG:
        return previous
    if a < STRAIGHT_DEG:
        return "straight"
    if a < LITTLE_DEG:
        return "little_" + side
    return side


def distance_band(dist_m, max_m=None):
    """'reach', 'close', 'steps', 'far' or None (unknown). max_m: the object is at
    most this far (a box cut off by the frame edge gives only an upper bound)."""
    if dist_m is None:
        if max_m is not None and max_m <= REACH_M + 0.2:
            return "reach"
        if max_m is not None and max_m <= CLOSE_M:
            return "close"
        return None
    if dist_m <= REACH_M:
        return "reach"
    if dist_m <= CLOSE_M:
        return "close"
    if dist_m <= STEPS_M:
        return "steps"
    return "far"


ZONE_WORDS = {"straight": "straight ahead", "little_left": "a little to your left",
              "little_right": "a little to your right", "left": "on your left", "right": "on your right"}
BAND_WORDS = {"close": "close", "steps": "a few steps away", "far": "more than a few steps away"}


class FindGuide:
    def __init__(self):
        self.state = IDLE
        self.label = None

    @property
    def active(self):
        return self.state != IDLE

    # ------------------------------------------------------------------
    def start(self, label, now, memory=None):
        """Begin guiding to `label` ('your bottle', 'the door', 'Sneha').

        memory (optional): the last remembered sighting, prepared by the caller:
            world_angle  float or None (only valid in the current session)
            age_text     '2 minutes ago'
            where        'on your left' (words already relative to the user)
            relation     'next to the laptop' or None
            turn         'turn about 90 degrees to your left' or None
        Returns the first sentence to say."""
        self.state = SEARCHING
        self.label = label
        self.started = now
        self.search_since = now
        self.last_seen = None
        self.world_angle = memory.get("world_angle") if memory else None
        self.zone = None            # zone last SPOKEN
        self.zone_now = None        # zone of the latest observation (with hysteresis)
        self.pending_zone = None
        self.pending_count = 0
        self.recent_dist = []
        self.filled = False
        self._lost_msg = None
        self.last_cue = None
        self.last_cue_at = -1e9
        self.last_search_hint = now
        self.search_side = None
        self.said_should_be_here = False
        self.reach_count = 0
        self.band = None
        self.cues_spoken = 0
        if memory:
            where = memory.get("where") or ""
            rel = memory.get("relation")
            sentence = (f"Looking for {label}. I can't see it right now. I last saw it "
                        f"{memory.get('age_text', 'earlier')}, {where}" + (f", {rel}" if rel else "") + ".")
            if memory.get("turn"):
                sentence += f" {memory['turn'][0].upper()}{memory['turn'][1:]}, slowly."
                self.search_side = "right" if "right" in memory["turn"] else "left"
            elif self.world_angle is not None:
                sentence += " It should be in front of you. Move the camera slowly."
            else:
                sentence += " Turn slowly and I will tell you when I see it."
            return sentence
        return (f"Looking for {label}. I can't see it right now. "
                f"Turn slowly in a circle and I will tell you when I see it.")

    def start_visible(self, label, now, obs, heading):
        """Begin when the target is already in view: the first cue straight away."""
        self.start(label, now)
        self.observe(now, [obs], heading)
        self.pending_count = CONFIRM_OBS        # first sighting: no need to wait
        return self.poll(now, heading, can_speak=True, force=True) or f"Looking for {label}."

    def stop(self, reason=None):
        label = self.label
        self.state = IDLE
        self.label = None
        if reason == "user":
            return f"Okay, I stopped looking for {label}."
        return None

    # ------------------------------------------------------------------
    def observe(self, now, candidates, heading):
        """Call on every frame the detector ran. candidates: observations of things
        that match the target, each {"angle": deg in image, "dist_m": float|None,
        "max_m": float|None, "filled": bool (optional, box spans the whole view width)}. Picks ONE: the one closest to where the target was
        (identity continuity), else the closest to straight ahead."""
        if not self.active or not candidates:
            return
        if self.world_angle is not None:
            pick = min(candidates, key=lambda c: abs(spatial.wrap_angle(c["angle"] + heading - self.world_angle)))
        else:
            pick = min(candidates, key=lambda c: abs(c["angle"]))
        first_sighting = self.state == SEARCHING
        self.state = GUIDING
        self._lost_msg = None
        self.last_seen = now
        self.world_angle = spatial.wrap_angle(pick["angle"] + heading)
        self.zone_now = direction_zone(pick["angle"], self.zone_now)
        # distance: median of the last 3 estimates, so one odd box can't change the words
        if pick.get("dist_m") is not None or pick.get("max_m") is not None:
            self.recent_dist.append((pick.get("dist_m"), pick.get("max_m")))
            del self.recent_dist[:-3]
        band = self._median_band()
        # A big thing (bed, door, bicycle) close up is wider than the view: its box is
        # cut off on both sides, so size only gives an upper bound. "Fills the view
        # from side to side" is then the honest "you are right at it" signal.
        self.filled = bool(pick.get("filled"))
        if self.filled:
            band = "reach"
        if band is not None:
            self.band = band
        if band == "reach" and (self.filled or distance_band(pick.get("dist_m"), pick.get("max_m")) == "reach") \
                and abs(pick["angle"]) <= REACH_MAX_ANGLE:
            self.reach_count += 1
        else:
            self.reach_count = 0
        if self.zone_now == self.pending_zone:
            self.pending_count += 1
        else:
            self.pending_zone, self.pending_count = self.zone_now, 1
        if first_sighting:
            self.said_should_be_here = False

    def _median_band(self):
        if not self.recent_dist:
            return None
        order = ["reach", "close", "steps", "far", None]
        bands = sorted((distance_band(d, m) for d, m in self.recent_dist), key=order.index)
        return bands[len(bands) // 2]

    # ------------------------------------------------------------------
    def poll(self, now, heading, can_speak=True, force=False):
        """Call every frame. Returns a sentence to say now, or None.

        can_speak=False while SENSE is talking, listening, or a hazard alert is
        waiting / was just said: guidance then simply waits."""
        if not self.active:
            return None
        # status messages (lost / give up) also keep the minimum gap after a cue
        ready = can_speak and now - self.last_cue_at >= MIN_GAP_S
        if now - self.started > TOTAL_TIMEOUT_S and ready:
            label = self.label
            self.stop()
            return f"I'm stopping the search for {label}. Ask me again if you need to."

        if self.state == GUIDING and now - self.last_seen > LOST_S:
            self.state = SEARCHING
            self.search_since = now
            self.last_search_hint = now
            self.reach_count = 0
            rel = spatial.relative_angle(self.world_angle, heading)
            turn = spatial.turn_instruction(rel)
            self.search_side = None if turn is None else ("right" if rel > 0 else "left")
            self._lost_msg = (f"I lost sight of {self.label}. It was {spatial.direction_phrase(rel)}"
                              + (f". {turn[0].upper()}{turn[1:]}, slowly." if turn else "."))
        if self._lost_msg:
            if not ready:
                return None
            msg, self._lost_msg = self._lost_msg, None
            return self._said(now, ("lost", None), msg)

        if self.state == SEARCHING:
            if now - self.search_since > SEARCH_TIMEOUT_S and ready:
                label = self.label
                self.stop()
                return (f"I couldn't find {label}. It may have been moved or be hidden. "
                        f"Stopping the search.")
            if not can_speak or now - self.last_search_hint < SEARCH_REPEAT_S \
                    or now - self.last_cue_at < MIN_GAP_S:
                return None
            self.last_search_hint = now
            if self.world_angle is None:
                return self._said(now, ("search", None), "Keep turning slowly.")
            rel = spatial.relative_angle(self.world_angle, heading)
            if spatial.is_in_view(self.world_angle, heading, margin=VIEW_MARGIN_DEG):
                if self.said_should_be_here:
                    return self._said(now, ("search", None), "Keep turning slowly.")
                self.said_should_be_here = True
                return self._said(now, ("search", "here"),
                                  f"{self.label[0].upper()}{self.label[1:]} should be in front of you, "
                                  f"but I can't see it. It may have been moved. Turn slowly.")
            turn = spatial.turn_instruction(rel) or (
                "turn a little to your " + ("right" if rel > 0 else "left"))
            side = "right" if rel > 0 else "left"
            text = ("Keep turning " + side) if side == self.search_side else (turn[0].upper() + turn[1:])
            self.search_side = side
            return self._said(now, ("search", side), text + ".")

        # GUIDING
        if self.reach_count >= CONFIRM_OBS:
            if not can_speak or now - self.last_cue_at < REACH_GAP_S:
                return None
            where = ZONE_WORDS[self.zone_now]
            label = self.label
            self.stop()
            if self.filled:
                return f"{label[0].upper()}{label[1:]} is right in front of you. Slow down."
            return f"{label[0].upper()}{label[1:]} should be within reach, {where}. Reach out slowly."
        if not can_speak and not force:
            return None
        if self.pending_count < CONFIRM_OBS:
            return None              # a new direction must be seen twice in a row
        since = now - self.last_cue_at
        if since < MIN_GAP_S and not force:
            return None
        cue = (self.pending_zone, self.band)
        last_zone, last_band = self.last_cue if self.last_cue else (None, None)
        order = ["reach", "close", "steps", "far"]
        closer = (cue[1] in order and (last_band not in order or order.index(cue[1]) < order.index(last_band)))
        if cue[0] == last_zone and not closer and since < REPEAT_S:
            return None              # nothing new: stay quiet, repeat only every REPEAT_S
        return self._said(now, cue, self._cue_text(cue, now))

    def _cue_text(self, cue, now):
        zone, band = cue
        words = ZONE_WORDS[zone]
        if self.cues_spoken == 0 or (self.last_cue and self.last_cue[0] in ("lost", "search")):
            # first cue after finding it: say what was found
            text = f"Found {self.label}, {words}"
        elif zone == "straight":
            text = "Straight ahead"
        else:
            text = words[0].upper() + words[1:]
        if band in BAND_WORDS:
            text += ", " + BAND_WORDS[band]
        return text + "."

    def _said(self, now, cue, text):
        self.last_cue, self.last_cue_at = cue, now
        if cue[0] not in ("lost", "search"):
            self.zone = cue[0]
        self.cues_spoken += 1
        return text
