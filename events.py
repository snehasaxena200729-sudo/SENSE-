"""
events.py - Debounced environmental events (Roadmap E2)

A small FINITE-STATE MACHINE per remembered item:

        +---------- seen again ----------+
        v                                |
    VISIBLE --unseen > GONE_AFTER_S--> GONE         (its spot was in view: it really left)
        |                                ^
        +--unseen, spot NOT in view--> OUT_OF_VIEW  (user turned away: we just can't see it)

Events produced:
    APPEARED     first sighting this session
    MOVED        position changed: 4 of the last 5 stable readings put it more than
                 MOVE_THRESHOLD_DEG away (in WORLD angle, so the user turning
                 their head is not mistaken for the object moving)
    GONE         disappeared while its spot was still in view
    OUT_OF_VIEW  not visible because the camera points elsewhere
    RETURNED     seen again after GONE / OUT_OF_VIEW

Why debounce? YOLO misses objects for single frames all the time. The old code
wrote MISSING after ONE missed frame. Waiting GONE_AFTER_S seconds (hysteresis)
removes those false alarms.

Pure logic: the caller passes the current time, so it is easy to unit-test with
a fake clock.
"""

from collections import deque

import spatial

VISIBLE, GONE, OUT_OF_VIEW = "VISIBLE", "GONE", "OUT_OF_VIEW"


class EventTracker:
    GONE_AFTER_S = 3.0
    ZONE_WINDOW = 5
    ZONE_VOTES_NEEDED = 4
    MOVE_THRESHOLD_DEG = 12.0   # about one third of the camera's view

    def __init__(self):
        self.items = {}  # name -> state dict

    def update(self, observations, now, camera_stable, heading):
        """observations: name -> dict(zone, world_angle, x, y, dist_feet)
        Returns a list of event dicts. Only call with stable frames' data;
        while the camera moves positions are unreliable, so we only age items.
        """
        events = []
        if not camera_stable:
            return events

        for name, obs in observations.items():
            st = self.items.get(name)
            if st is None:
                self.items[name] = {
                    "state": VISIBLE, "zone": obs["zone"], "obs": obs,
                    "last_seen": now, "votes": deque(maxlen=self.ZONE_WINDOW),
                    "anchor": obs,
                }
                events.append(self._event("APPEARED", name, obs, now))
                continue

            if st["state"] in (GONE, OUT_OF_VIEW):
                ev = self._event("RETURNED", name, obs, now)
                if st["state"] == GONE and self._moved(st["obs"], obs):
                    ev["from_zone"] = st["zone"]
                events.append(ev)
                st.update(state=VISIBLE, zone=obs["zone"], anchor=obs)
                st["votes"].clear()
            else:
                st["votes"].append(self._moved(st["anchor"], obs))
                if sum(st["votes"]) >= self.ZONE_VOTES_NEEDED:
                    ev = self._event("MOVED", name, obs, now)
                    ev["from_zone"] = st["zone"]
                    events.append(ev)
                    st["votes"].clear()
                    st["anchor"] = obs
                    st["zone"] = obs["zone"]
            if st["state"] == VISIBLE and "anchor" not in st:
                st["anchor"] = obs
            st["obs"] = obs
            st["last_seen"] = now

        for name, st in self.items.items():
            if st["state"] != VISIBLE or name in observations:
                continue
            if now - st["last_seen"] < self.GONE_AFTER_S:
                continue
            angle = st["obs"].get("world_angle")
            in_view = angle is None or spatial.is_in_view(angle, heading)
            st["state"] = GONE if in_view else OUT_OF_VIEW
            events.append(self._event(st["state"], name, st["obs"], st["last_seen"]))
        return events

    def _moved(self, old, new):
        a, b = old.get("world_angle"), new.get("world_angle")
        if a is None or b is None:
            return old.get("zone") != new.get("zone")
        return abs(spatial.wrap_angle(a - b)) > self.MOVE_THRESHOLD_DEG

    def state_of(self, name):
        st = self.items.get(name)
        return st["state"] if st else None

    @staticmethod
    def _event(kind, name, obs, ts):
        return {
            "type": kind, "name": name, "ts": ts,
            "zone": obs.get("zone"), "world_angle": obs.get("world_angle"),
            "dist_feet": obs.get("dist_feet"),
        }
