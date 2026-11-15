"""
identity.py - Stable object identity for SENSE (Roadmap E1)

PROBLEM THIS SOLVES
-------------------
Before this module, a generic object's memory key was "<colour> <category>",
recomputed on EVERY frame. Perceived colour flickers with lighting, so one
physical phone became "red phone", "orange phone", "gray phone"... (36 DB rows
for ~6 objects). Answers then read whichever stale row matched.

THE IDEA
--------
Separate IDENTITY (which physical thing is this?) from ATTRIBUTES (what colour
does it look like right now?).

1. ByteTrack already gives a track id (tid) that is stable while an object stays
   in view. We map each tid to an "entity" (our long-lived identity).
2. When a NEW tid appears, we try to re-identify it against remembered entities
   of the same category using:
     - appearance: correlation of the 2D chroma histogram (same signature that
       sense_master already uses for personal items), and
     - position: how close it is to where that entity was last seen
       (only a tie-breaking bonus, and only if seen recently).
   If the best score passes a threshold we reuse that entity, otherwise we create
   a new one.  (This is "gated nearest-neighbour data association".)
3. Colour is VOTED over the last N frames of a track. The entity is named only
   once one colour wins a clear majority, and the name is then frozen. If the
   name is already used by another entity we add a number ("white cup 2").

Runs synchronously in the main loop: no threads, no new dependencies.
"""

from collections import Counter, deque
import math

try:
    import cv2
except ImportError:  # allows the pure-logic tests to run without OpenCV
    cv2 = None


def histogram_similarity(sig_a, sig_b):
    """Correlation between two normalised histograms (1.0 = identical)."""
    if sig_a is None or sig_b is None:
        return None
    if cv2 is not None:
        return float(cv2.compareHist(sig_a, sig_b, cv2.HISTCMP_CORREL))
    # Fallback (same maths as HISTCMP_CORREL) for environments without OpenCV
    a = sig_a.ravel().astype(float)
    b = sig_b.ravel().astype(float)
    a = a - a.mean()
    b = b - b.mean()
    denom = math.sqrt(float((a * a).sum()) * float((b * b).sum()))
    return float((a * b).sum() / denom) if denom else 0.0


class EntityResolver:
    # Tunable parameters (kept as class attributes so they are easy to explain)
    COLOR_WINDOW = 30          # frames of colour readings kept per track
    COLOR_MIN_VOTES = 8        # need at least this many readings to name
    COLOR_MAJORITY = 0.60      # winning colour must have >= 60% of readings
    MATCH_THRESHOLD = 0.50     # minimum appearance correlation to re-identify
    PROXIMITY_RADIUS_PX = 200  # position bonus fades to 0 at this distance
    PROXIMITY_WEIGHT = 0.30    # how much position can add to the score
    PROXIMITY_MAX_AGE_S = 60   # positions older than this give no bonus
    CLAIM_TIMEOUT_S = 1.0      # an entity seen by another track this recently is "taken"
    SIG_LEARNING_RATE = 0.10   # exponential moving average for appearance

    def __init__(self, entities=None):
        # entity_id -> dict(category, color, name, sig, last_x, last_y, last_seen)
        self.entities = dict(entities or {})
        self.track_to_entity = {}
        self.color_votes = {}
        self.entity_claimed_at = {}   # entity_id -> (tid, time)
        self.track_last_seen = {}     # tid -> time (for pruning old buffers)
        self.dirty = set()            # entity ids that changed since last save
        self._counter = 0
        for eid in self.entities:
            try:
                self._counter = max(self._counter, int(eid.rsplit("-", 1)[1]))
            except (IndexError, ValueError):
                pass

    # ------------------------------------------------------------------
    def _new_entity_id(self, category):
        self._counter += 1
        return f"{category}-{self._counter}"

    def _name_taken(self, name, except_eid=None):
        return any(e.get("name") == name and eid != except_eid
                   for eid, e in self.entities.items())

    def _unique_name(self, base, eid):
        if not self._name_taken(base, eid):
            return base
        n = 2
        while self._name_taken(f"{base} {n}", eid):
            n += 1
        return f"{base} {n}"

    def _score(self, entity, sig, center, now):
        sim = histogram_similarity(sig, entity.get("sig"))
        if sim is None:
            return None
        bonus = 0.0
        last_seen = entity.get("last_seen") or 0
        if center is not None and entity.get("last_x") is not None and now - last_seen < self.PROXIMITY_MAX_AGE_S:
            dist = math.hypot(center[0] - entity["last_x"], center[1] - entity["last_y"])
            bonus = max(0.0, 1.0 - dist / self.PROXIMITY_RADIUS_PX) * self.PROXIMITY_WEIGHT
        return sim, sim + bonus

    def _is_claimed(self, eid, tid, now):
        claim = self.entity_claimed_at.get(eid)
        return claim is not None and claim[0] != tid and now - claim[1] < self.CLAIM_TIMEOUT_S

    # ------------------------------------------------------------------
    def resolve(self, tid, category, sig, color, center, now, learn=True):
        """Return the entity id for this track (creating one if needed).

        Returns None while the crop is too small to fingerprint a NEW track.
        `learn` should be False while the camera is moving (blurry crops).
        """
        self.track_last_seen[tid] = now
        # Colour vote for this track
        votes = self.color_votes.setdefault(tid, deque(maxlen=self.COLOR_WINDOW))
        if color:
            votes.append(color)

        eid = self.track_to_entity.get(tid)
        if eid is not None and self.entities.get(eid, {}).get("category") != category:
            eid = None  # the detector changed its mind about the class: re-resolve

        if eid is None:
            if sig is None:
                return None
            best_eid, best_score = None, -1.0
            for cand_eid, ent in self.entities.items():
                if ent.get("category") != category or self._is_claimed(cand_eid, tid, now):
                    continue
                scored = self._score(ent, sig, center, now)
                if scored is None:
                    continue
                sim, total = scored
                if sim >= self.MATCH_THRESHOLD and total > best_score:
                    best_eid, best_score = cand_eid, total
            if best_eid is None:
                best_eid = self._new_entity_id(category)
                self.entities[best_eid] = {
                    "category": category, "color": None, "name": None,
                    "sig": sig.copy(), "last_x": None, "last_y": None,
                    "last_seen": now,
                }
                self.dirty.add(best_eid)
            eid = best_eid
            self.track_to_entity[tid] = eid

        ent = self.entities[eid]
        self.entity_claimed_at[eid] = (tid, now)
        ent["last_seen"] = now
        if center is not None:
            ent["last_x"], ent["last_y"] = center
        if learn and sig is not None and ent.get("sig") is not None:
            ent["sig"] = ((1 - self.SIG_LEARNING_RATE) * ent["sig"]
                          + self.SIG_LEARNING_RATE * sig).astype(ent["sig"].dtype)

        # Name the entity once its colour is clear (then the name is frozen)
        if ent.get("name") is None and len(votes) >= self.COLOR_MIN_VOTES:
            winner, count = Counter(votes).most_common(1)[0]
            if count / len(votes) >= self.COLOR_MAJORITY:
                ent["color"] = winner
                ent["name"] = self._unique_name(f"{winner} {category}", eid)
                self.dirty.add(eid)
        return eid

    def name_of(self, eid):
        """Stable memory name ('white cup'), or None if not settled yet."""
        ent = self.entities.get(eid)
        return ent.get("name") if ent else None

    def display_label(self, eid, category):
        name = self.name_of(eid)
        return name.title() if name else category.capitalize()

    def forget_track(self, tid):
        self.track_to_entity.pop(tid, None)
        self.color_votes.pop(tid, None)
        self.track_last_seen.pop(tid, None)

    def prune_tracks(self, now, max_age_s=10.0):
        """Drop per-track buffers for tracks not seen for a while (bounded memory)."""
        for tid, seen in list(self.track_last_seen.items()):
            if now - seen > max_age_s:
                self.forget_track(tid)

    def pop_dirty(self):
        dirty = [(eid, self.entities[eid]) for eid in self.dirty if eid in self.entities]
        self.dirty.clear()
        return dirty
