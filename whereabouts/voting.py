"""
voting.py - one noisy frame must never change what SENSE tells you.

Each item keeps its last `window` (5) samples of (surface, landmark). A value
is DECIDED only when at least `needed` (3) of those samples agree - a simple
majority vote, the same idea as events.py's zone votes and sense_master's
class votes. "Nothing there" is a value too: 3 of 5 samples with no table
under the item decide "not on furniture".

If fewer than 3 agree the field is UNDECIDED and the stored anchor is kept.

Samples are taken at most every 0.4 s (frames 1/30 s apart are nearly the
same picture and would just repeat one error 5 times), and the window is
cleared when the item moves or has not been seen for a while, so votes from
two different places are never mixed.
"""

from collections import Counter, deque

UNDECIDED = object()


class AnchorVoter:
    def __init__(self, cfg):
        self.cfg = cfg
        self.items = {}   # name -> {"samples": deque, "last": t, "angle": deg}

    def add(self, name, now, surface, landmark, world_angle=None):
        """Record one sample. Returns True if it was taken (not throttled)."""
        st = self.items.get(name)
        if st is not None and (now - st["last"] > self.cfg.reset_gap_s or
                               _moved(st.get("angle"), world_angle, self.cfg.moved_deg)):
            st = None                          # stale or moved: start a fresh vote
        if st is None:
            st = {"samples": deque(maxlen=self.cfg.window), "last": -1e9, "angle": world_angle}
            self.items[name] = st
        if now - st["last"] < self.cfg.sample_interval_s:
            return False
        st["samples"].append((surface, landmark))
        st["last"] = now
        if world_angle is not None:
            st["angle"] = world_angle
        return True

    def decide(self, name):
        """(surface, landmark, surface_votes, landmark_votes); each value may be
        UNDECIDED. A decided None means "confirmed: nothing there"."""
        st = self.items.get(name)
        if st is None:
            return UNDECIDED, UNDECIDED, 0, 0
        s_val, s_n = self._winner([s for s, _ in st["samples"]])
        l_val, l_n = self._winner([lm for _, lm in st["samples"]])
        return s_val, l_val, s_n, l_n

    def _winner(self, values):
        if not values:
            return UNDECIDED, 0
        value, count = Counter(values).most_common(1)[0]
        return (value, count) if count >= self.cfg.needed else (UNDECIDED, count)

    def forget(self, name):
        self.items.pop(name, None)

    def prune(self, now, max_age_s=600.0):
        for name in [n for n, st in self.items.items() if now - st["last"] > max_age_s]:
            del self.items[name]


def _moved(old, new, limit):
    if old is None or new is None:
        return False
    d = (new - old + 180.0) % 360.0 - 180.0
    return abs(d) > limit
