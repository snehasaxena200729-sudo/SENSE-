"""
teaching.py - learning a room while you turn round once.

After "this room is the kitchen", SENSE says "turn slowly all the way round".
While you turn, TeachSession keeps one view for every 20 degrees of turn
(18 bins for a full circle), using the heading SENSE already estimates
(heading_fusion.py). A view is only taken from a sharp frame (motion blur is
measured with the variance of the Laplacian) and at most a few per second.

Nothing is written to the database until the session finishes, so a
cancelled "relearn" never destroys the old views.

    feed(...)  -> None, or ("progress"|"nudge"|"done"|"failed", info)
"""

import math


def wrap(deg):
    return (deg + 180.0) % 360.0 - 180.0


class TeachSession:
    def __init__(self, room_id, room_name, heading, now, cfg, relearn=False):
        self.room_id, self.room_name, self.relearn, self.cfg = room_id, room_name, relearn, cfg
        self.started = now
        self.last_heading = heading
        self.turned = 0.0              # signed, unwrapped degrees since the start
        self.max_abs_turn = 0.0
        self.views = []                # dicts: heading_deg, fingerprint parts, landmarks
        self.bins = set()
        self.last_capture = -1e9
        self.nudged = False
        self.halfway_said = False

    @property
    def bins_total(self):
        return int(round(360.0 / self.cfg.view_step_deg))

    def _bin(self, angle):
        return int((angle % 360.0) // self.cfg.view_step_deg)

    def feed(self, heading, now, sharp, capture):
        """One frame. capture() -> view dict (or None); called only when a view is wanted."""
        self.turned += wrap(heading - self.last_heading)
        self.last_heading = heading
        self.max_abs_turn = max(self.max_abs_turn, abs(self.turned))
        b = self._bin(self.turned)
        wanted = (b not in self.bins and now - self.last_capture >= self.cfg.capture_min_interval_s
                  and sharp >= self.cfg.min_sharpness)
        if wanted:
            view = capture()
            if view is not None:
                view["heading_deg"] = round(self.turned % 360.0, 1)
                view["bin"] = b
                self.views.append(view)
                self.bins.add(b)
                self.last_capture = now
        elapsed = now - self.started
        if len(self.bins) >= self.cfg.done_bins or self.max_abs_turn >= 360.0 and len(self.views) >= self.cfg.min_views:
            return ("done", len(self.views))
        if elapsed >= self.cfg.max_teach_s:
            return ("done", len(self.views)) if len(self.views) >= self.cfg.min_views else ("failed", len(self.views))
        if not self.nudged and elapsed >= self.cfg.nudge_after_s and self.max_abs_turn < 20.0:
            self.nudged = True
            return ("nudge", None)
        if not self.halfway_said and len(self.bins) >= self.bins_total // 2:
            self.halfway_said = True
            return ("progress", len(self.bins))
        return None

    def stop(self):
        """User said stop: finish with what we have."""
        return ("done", len(self.views)) if len(self.views) >= self.cfg.min_stop_views else ("failed", len(self.views))

    def coverage_deg(self):
        return len(self.bins) * self.cfg.view_step_deg

    def __repr__(self):
        return (f"TeachSession({self.room_name}, views={len(self.views)}, "
                f"turned={self.turned:.0f}, {math.floor(self.coverage_deg())} deg covered)")
