"""
tracker.py - WhereaboutsTracker, the only thing sense_master.py talks to.

    observe(frame_observations, now, frame_w, frame_h, camera_stable)
        once per frame. Uses only the stable, settled observations that
        sense_master already collected (with their box and category).
    describe(rec, spoken_name, age_s, where, same_session, turn_hint, now)
        the "where is my phone?" sentence, or None -> caller keeps its old one.
    relation_for(rec)
        "on the table, to the left of the laptop" for "find my phone", or None.
    forget(name) / rename(old, new)
        keep the table in step with SENSE's privacy and rename commands.

Every public method is wrapped so that an error here is printed once and
SENSE carries on exactly as it did without this package.
"""

import functools
import time

from . import phrasing
from .config import WhereaboutsConfig
from .furniture import surface_under
from .landmarks import nearest_landmark
from .store import AnchorStore
from .voting import UNDECIDED, AnchorVoter


def _safe(default=None):
    def wrap(fn):
        @functools.wraps(fn)
        def inner(self, *a, **kw):
            if not self.enabled:
                return default
            try:
                return fn(self, *a, **kw)
            except Exception as e:      # never let this feature break SENSE
                if fn.__name__ not in self._errors:
                    self._errors.add(fn.__name__)
                    print(f"📍 Whereabouts {fn.__name__} error (ignored): {e!r}")
                return default
        return inner
    return wrap


class WhereaboutsTracker:
    def __init__(self, conn, session_id, cfg=None, clock=time.time):
        self.cfg = cfg or WhereaboutsConfig()
        self.session_id = session_id
        self.clock = clock
        self.enabled = True
        self._errors = set()
        self.store = AnchorStore(conn)
        self.voter = AnchorVoter(self.cfg)
        self._last_write = {}    # name -> (time, content)

    # ------------------------------------------------------------------
    # Seeing
    # ------------------------------------------------------------------
    @_safe()
    def observe(self, observations, now, frame_w, frame_h, camera_stable=True):
        """observations: sense_master's frame_observations, name -> dict with
        'box', 'category', 'zone', 'world_angle'. Only call with a stable
        camera: while it moves, boxes and the heading are blurred."""
        if not camera_stable or not observations:
            return
        objs = [(name, o.get("category"), tuple(o["box"]), o)
                for name, o in observations.items() if o.get("box") and o.get("category")]
        furniture = [(cat, box) for _, cat, box, _ in objs if cat in self.cfg.surfaces]
        others = [(name, cat, box) for name, cat, box, _ in objs]
        for name, cat, box, obs in objs:
            if cat not in self.cfg.item_categories:
                continue
            surface = surface_under(box, furniture, frame_h, self.cfg)
            lm = nearest_landmark(name, box, others, frame_w, self.cfg)
            landmark = (lm[0].lower(), lm[1], lm[2]) if lm else None
            if self.voter.add(name.lower(), now, surface, landmark, obs.get("world_angle")):
                self._commit(name.lower(), cat, obs, now)
        if int(now) % 60 == 0:
            self.voter.prune(now)

    def _commit(self, name, category, obs, now):
        surface, landmark, s_votes, l_votes = self.voter.decide(name)
        if surface is UNDECIDED and landmark is UNDECIDED:
            return                       # nothing reliable yet: keep what is stored
        stored = self.store.get(name) or {}
        same_place = self._same_place(stored, obs)
        if surface is UNDECIDED:         # keep the stored surface only if the item has not moved
            surface, s_votes = (stored.get("surface"), stored.get("surface_votes")) if same_place else (None, 0)
        if landmark is UNDECIDED:
            if same_place and stored.get("landmark"):
                landmark = (stored["landmark"], stored.get("landmark_category"), stored.get("relation"))
                l_votes = stored.get("landmark_votes")
            else:
                landmark, l_votes = None, 0
        content = (surface, landmark)
        last_t, last_content = self._last_write.get(name, (-1e9, None))
        if content == last_content and now - last_t < self.cfg.refresh_s:
            return
        unchanged = stored and (stored.get("surface"), self._stored_landmark(stored)) == content
        self.store.save({
            "item_name": name, "item_category": category,
            "surface": surface, "surface_votes": s_votes,
            "landmark": landmark[0] if landmark else None,
            "landmark_category": landmark[1] if landmark else None,
            "relation": landmark[2] if landmark else None,
            "landmark_votes": l_votes if landmark else None,
            "zone": obs.get("zone"), "world_angle": obs.get("world_angle"),
            "session_id": self.session_id, "seen_at": now,
            "first_seen_at": stored.get("first_seen_at") if unchanged else now,
        })
        self._last_write[name] = (now, content)
        if not unchanged:
            print(f"📍 {name}: {phrasing.relation_text(self.store.get(name)) or 'no reliable anchor'}")

    @staticmethod
    def _stored_landmark(stored):
        if not stored.get("landmark"):
            return None
        return (stored["landmark"], stored.get("landmark_category"), stored.get("relation"))

    def _same_place(self, stored, obs):
        if not stored or stored.get("session_id") != self.session_id:
            return False
        a, b = stored.get("world_angle"), obs.get("world_angle")
        if a is None or b is None:
            return False
        return abs((b - a + 180.0) % 360.0 - 180.0) <= self.cfg.moved_deg

    # ------------------------------------------------------------------
    # Answering
    # ------------------------------------------------------------------
    def _anchor_for(self, rec):
        """(place words, anchor, is_fresh) for a memory record, or None."""
        if not rec or not rec.get("name"):
            return None
        anchor = self.store.get(rec["name"])
        place = phrasing.relation_text(anchor)
        if not place or anchor.get("seen_at") is None or rec.get("last_seen") is None:
            return None
        fresh = anchor["seen_at"] >= rec["last_seen"] - self.cfg.stale_gap_s
        return place, anchor, fresh

    @_safe()
    def describe(self, rec, spoken_name, age_s, where, same_session, turn_hint=None, now=None):
        """Full "where is ..." sentence using the anchor, or None (no anchor,
        or a stale one: the caller keeps its own sentence and may add
        stale_note())."""
        found = self._anchor_for(rec)
        if not found or not found[2]:
            return None
        place, _, _ = found
        return phrasing.describe(spoken_name, place, age_s, rec.get("status"),
                                 where, same_session, turn_hint)

    @_safe(default="")
    def stale_note(self, rec, now=None):
        """' Earlier, 15 minutes ago, it was on the table, ...' when the
        anchor is older than the last sighting; '' otherwise."""
        found = self._anchor_for(rec)
        if not found or found[2]:
            return ""
        place, anchor, _ = found
        now = self.clock() if now is None else now
        return phrasing.earlier_note(place, now - anchor["seen_at"])

    @_safe()
    def relation_for(self, rec):
        """Place words for find mode, only when the anchor is fresh."""
        found = self._anchor_for(rec)
        return found[0] if found and found[2] else None

    @_safe()
    def forget(self, name):
        self.store.forget(name)
        self.voter.forget(name.lower())
        self._last_write.pop(name.lower(), None)

    @_safe()
    def rename(self, old, new):
        self.store.rename(old, new)
        self.voter.forget(old.lower())
        self._last_write.pop(old.lower(), None)


class _Disabled:
    """Stands in when the tracker can't start: every call does nothing."""
    enabled = False

    def observe(self, *a, **kw):
        return None

    describe = relation_for = forget = rename = observe

    def stale_note(self, *a, **kw):
        return ""


def create(session_id, conn=None):
    """Make the tracker on memory_db's connection; never raises."""
    try:
        if conn is None:
            import memory_db
            conn = memory_db._get_conn()
        tracker = WhereaboutsTracker(conn, session_id)
        print("📍 Item whereabouts on (remembers what your things are on and next to).")
        return tracker
    except Exception as e:
        print(f"📍 Item whereabouts off ({e!r})")
        return _Disabled()
