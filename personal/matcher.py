"""
matcher.py - which track (if any) is YOUR cup? Pure logic: no camera, no SQLite.

THE PROBLEM
-----------
"cup" is a category; "my cup" is one physical cup. The old code named ANY new
cup track "my cup" when its colours correlated above 0.55 with the taught cup,
and nothing stopped two tracks from both being "my cup". Turning 45 degrees to
a second white cup was enough to move "my cup" in memory.

THE IDEA: evidence + a gate, per object instance
-----------------------------------------------
Each personal item is an INSTANCE with:
    taught look      the colour signature from "this is my cup"
    adapted look     slowly updated while we are sure we see it
    position         world angle (direction in the room) of the last sure
                     sighting, valid only inside the same run (session)
    look-alikes      colour signatures (and positions) of OTHER objects of the
                     same kind that were in the SAME picture as it - provably
                     different objects

A ByteTrack id is only a short-term handle. While the same track stays in
view, it keeps the label ("track continuity"). A NEW track gets the personal
label only if every check passes (a "gated association"):

    1. appearance    similarity >= match_min (0.75; 0.80 after a restart)
    2. position      same run: within near_deg (25) of where your cup was
    3. margin        it looks clearly more like your cup than like any known
                     look-alike (>= +0.10). This is waived only when no
                     look-alike was ever seen near that spot - then the position
                     decides, which is what makes two IDENTICAL cups work.
    After a RESTART the heading starts again from wherever the camera faced,
    so positions from the last run can't be trusted: a cup that looks like
    yours is reported as "looks like yours" until you say "this is my cup"
    again (PersonalConfig.confirm_after_restart switches on a riskier mode).
    4. uniqueness    your cup is not already on another visible track, and no
                     second candidate passes in the same frame
    5. persistence   5 of the last 8 checks of this track agree

If the appearance is close but a check fails, the track is remembered as
"looks like yours" (for honest answers) and keeps its generic name.

This is the same pattern as multi-object tracking re-identification ("gated
nearest-neighbour data association"), tuned to prefer "I'm not sure" over a
wrong name: for a blind user a confident wrong answer is the worst outcome.

Everything runs synchronously in the main loop: no threads.
"""

from collections import deque

import numpy as np

from identity import histogram_similarity

from .config import PersonalConfig


def is_neutral(sig, threshold):
    """True for white / grey / black objects: almost all of the colour
    histogram sits in the bins around Cr = Cb = 128 (no colour)."""
    if sig is None:
        return True
    h = np.asarray(sig, np.float64)
    total = float(h.sum())
    if total <= 0:
        return True
    core = float(h[6:10, 8:12].sum())       # Cr 96..160, Cb 102..154
    return core / total >= threshold


def angle_diff(a, b):
    return abs((float(a) - float(b) + 180.0) % 360.0 - 180.0)


class Item:
    """One personal object instance."""

    def __init__(self, name, category, taught_sig, instance_id, created_at,
                 adapted_sig=None, lookalikes=None, world_angle=None, session_id=None,
                 last_seen=None):
        self.name = name                     # "my cup"
        self.category = category             # "cup"
        self.instance_id = instance_id       # "cup-1731772800" - never reused
        self.taught_sig = taught_sig
        self.adapted_sig = adapted_sig if adapted_sig is not None else (
            taught_sig.copy() if taught_sig is not None else None)
        self.lookalikes = list(lookalikes or [])   # [Lookalike]
        self.world_angle = world_angle
        self.session_id = session_id
        self.last_seen = last_seen
        self.created_at = created_at
        self.adapt_n = 1                     # samples averaged into adapted_sig so far
        self.ambiguous_at = None             # last time two tracks both looked like it
        self.last_lookalike_add = -1e9

    def similarity(self, sig):
        # adapted_sig starts as the taught look and becomes the average of the
        # sightings we were sure about: one frame's colours are noisy (a white
        # cup's correlation with itself jumps between ~0.5 and ~0.98)
        return histogram_similarity(sig, self.adapted_sig if self.adapted_sig is not None else self.taught_sig)

    def decoy_similarity(self, sig):
        sims = [histogram_similarity(sig, l.sig) for l in self.lookalikes]
        sims = [s for s in sims if s is not None]
        return max(sims) if sims else -1.0


class Lookalike:
    """Another object of the same kind, seen in the same picture as the item."""

    def __init__(self, sig, world_angle=None, session_id=None):
        self.sig = sig
        self.world_angle = world_angle
        self.session_id = session_id


class TrackState:
    def __init__(self, tid, category, now):
        self.tid = tid
        self.category = category
        self.first_seen = now
        self.last_seen = now
        self.locked = None                   # item name while this track IS the item
        self.votes = deque()                 # recent checks: item name or None
        self.doubt = 0
        self.moved_since_check = False       # camera turned since the last stable check
        self.lookalike_of = set()            # items it looks like but could not be confirmed as
        self.world_angle = None
        self.sigs = deque(maxlen=12)         # recent sharp (camera still) colour signatures
        self.resume_ref = None               # item's place before the camera turned (id-switch check)

    def mean_sig(self, min_samples=1):
        """Average look of this track: far less noisy than one frame."""
        if len(self.sigs) < min_samples:
            return None
        return np.mean(np.stack(self.sigs), axis=0).astype(np.float32)


class PersonalMatcher:
    def __init__(self, session_id, cfg=None):
        self.cfg = cfg or PersonalConfig()
        self.session_id = session_id
        self.items = {}                      # name -> Item
        self.tracks = {}                     # tid -> TrackState
        self._frame = []                     # this frame's evidence (tid, category, sig, angle)
        self._last_angles = {}               # category -> {tid: world angle} of the previous frame
        self.dirty = set()                   # items to persist

    # ------------------------------------------------------------------
    # Items
    # ------------------------------------------------------------------
    def categories(self):
        return {it.category for it in self.items.values()}

    def add_item(self, item):
        self.items[item.name] = item

    def teach(self, name, category, sig, tid, world_angle, now, other_sigs=()):
        """'This is my cup': bind the name to the instance on track `tid`.

        other_sigs: (signature, world angle) of the other objects of the same
        kind in the same picture. They are different physical objects by
        definition, so they become the item's first look-alikes."""
        old = self.items.get(name)
        ts_now = self.tracks.get(tid)
        if ts_now is not None and ts_now.category == category and len(ts_now.sigs) >= self.cfg.min_samples:
            sig = ts_now.mean_sig()
        item = Item(name, category, sig.copy(), instance_id=f"{category}-{int(now * 1000)}",
                    created_at=now, world_angle=world_angle,
                    session_id=self.session_id if world_angle is not None else None,
                    last_seen=now)
        if old is not None and old.category == category:
            # re-taught ("yes, this is my cup" after a restart): the same
            # personal identity, with a fresh look and place
            item.created_at, item.instance_id = old.created_at, old.instance_id
            # keep what we knew about other cups, except anything that looks
            # like the cup being taught now (it might be this very cup)
            item.lookalikes = [l for l in old.lookalikes
                               if (histogram_similarity(l.sig, item.taught_sig) or 0) < self.cfg.match_min]
        for other in other_sigs:
            sig_o, angle_o, tid_o = (tuple(other) + (None, None))[:3] if isinstance(other, tuple) \
                else (other, None, None)
            ts_o = self.tracks.get(tid_o)
            if ts_o is not None and len(ts_o.sigs) >= self.cfg.min_samples:
                sig_o = ts_o.mean_sig()           # its average look, if we have one
            self._add_lookalike(item, sig_o, now, angle_o, force=True)
        self.items[name] = item
        # the name now belongs to this track only
        for ts in self.tracks.values():
            if ts.locked == name:
                ts.locked = None
                ts.votes.clear()
            ts.lookalike_of.discard(name)
        ts = self.tracks.get(tid)
        if ts is None or ts.category != category:
            ts = self.tracks[tid] = TrackState(tid, category, now)
        ts.locked, ts.last_seen, ts.doubt, ts.world_angle = name, now, 0, world_angle
        ts.votes.clear()
        self.dirty.add(name)
        return item

    def forget(self, name):
        self.items.pop(name, None)
        for ts in self.tracks.values():
            if ts.locked == name:
                ts.locked = None
            ts.lookalike_of.discard(name)
            ts.votes = deque(v for v in ts.votes if v != name)

    def rename(self, old, new):
        item = self.items.pop(old, None)
        if item is None:
            return
        item.name = new
        self.items[new] = item
        for ts in self.tracks.values():
            if ts.locked == old:
                ts.locked = new
            if old in ts.lookalike_of:
                ts.lookalike_of.discard(old)
                ts.lookalike_of.add(new)
            ts.votes = deque(new if v == old else v for v in ts.votes)
        self.dirty.add(new)

    # ------------------------------------------------------------------
    # Seeing: label() per confirmed track, end_frame() once per frame
    # ------------------------------------------------------------------
    def label(self, tid, category, sig, world_angle, now, stable):
        """The personal name for this track RIGHT NOW, or None.

        Records this sighting as evidence; new associations are decided in
        end_frame(), so they appear one frame later (~50 ms)."""
        if category not in self.categories():
            self.tracks.pop(tid, None)   # e.g. the detector now calls it a bottle
            return None
        ts = self.tracks.get(tid)
        if ts is not None and (ts.category != category or now - ts.last_seen > self.cfg.lock_gap_s):
            ts = None                    # class changed, or the id came back after a gap
        if ts is None:
            ts = self.tracks[tid] = TrackState(tid, category, now)
        ts.last_seen = now
        ts.world_angle = world_angle
        if not stable:
            ts.moved_since_check = True
        elif sig is not None:
            ts.sigs.append(np.asarray(sig, np.float32))
        mean = ts.mean_sig(self.cfg.min_samples)
        self._frame.append((tid, category, mean, world_angle, stable))

        if ts.locked is None:
            return None
        item = self.items.get(ts.locked)
        if item is None or item.category != category:
            ts.locked = None
            return None
        if not self._still_same(ts, item, mean, world_angle, stable):
            ts.locked = None
            ts.votes.clear()
            return None
        if stable:
            sure = ts.doubt == 0 and mean is not None and (item.similarity(mean) or 0) >= self.cfg.match_min
            self._refresh(item, sig if sure else None, world_angle, now)
        else:
            item.last_seen = now
        return item.name

    def _still_same(self, ts, item, sig, world_angle, stable=True):
        """Guards against ByteTrack id switches while a track is locked.

        While the camera turns (and on the first sharp frame after), the
        object must stay where the item was (its place is only updated from
        sharp frames), and no other track of the same kind may be clearly
        closer to that place. Then, with a sharp picture, the look is checked."""
        turning = (not stable or ts.moved_since_check) and item.world_angle is not None \
            and item.session_id == self.session_id and world_angle is not None
        if turning:
            mine = angle_diff(world_angle, item.world_angle)
            closer = any(t != ts.tid and a is not None
                         and angle_diff(a, item.world_angle) + self.cfg.resume_slack_deg < mine
                         for t, a in self._last_angles.get(ts.category, {}).items())
            if mine > self.cfg.jump_deg or closer:
                print(f"🔓 '{item.name}': track {ts.tid} may have switched objects during the turn; "
                      f"label removed")
                return False
            if stable:
                ts.resume_ref = item.world_angle  # end_frame checks this frame's other tracks too
        if not stable:
            return True
        ts.moved_since_check = False
        sim = item.similarity(sig) if sig is not None else None
        if sim is None:
            return True
        if sim < self.cfg.doubt_below or sim < item.decoy_similarity(sig) - self.cfg.margin:
            ts.doubt += 1
        else:
            ts.doubt = 0
        return ts.doubt < self.cfg.doubt_frames

    def _refresh(self, item, sig, world_angle, now):
        item.last_seen = now
        if world_angle is not None:
            item.world_angle = world_angle
            item.session_id = self.session_id
        if sig is not None and item.adapted_sig is not None:
            # running average for the first sightings, then a slow moving average
            r = max(self.cfg.adapt_rate, 1.0 / (item.adapt_n + 1))
            item.adapted_sig = ((1 - r) * item.adapted_sig + r * np.asarray(sig, np.float32)).astype(np.float32)
            item.adapt_n += 1
        self.dirty.add(item.name)

    def held(self, name, now, except_tid=None):
        """Is this item on a track that is visible right now?"""
        return any(ts.locked == name and ts.tid != except_tid and now - ts.last_seen <= self.cfg.held_s
                   for ts in self.tracks.values())

    def _verdict(self, item, sig, world_angle):
        """('match' | 'lookalike' | None, appearance score) for one candidate."""
        sim = item.similarity(sig) if sig is not None else None
        if sim is None or sim < self.cfg.lookalike_min:
            return None, sim
        decoy = item.decoy_similarity(sig)
        margin_ok = sim - decoy >= self.cfg.margin
        # after a restart the look must be even closer
        look_ok = sim >= (self.cfg.match_min if item.session_id == self.session_id
                          else self.cfg.restart_match_min)
        same_run = item.session_id == self.session_id
        colourful = not is_neutral(item.adapted_sig if item.adapted_sig is not None else item.taught_sig,
                                   self.cfg.neutral_mass)
        if same_run and item.world_angle is not None:
            # Same run: the heading is tracked, so the place is reliable.
            place_ok = world_angle is not None and angle_diff(world_angle, item.world_angle) <= self.cfg.near_deg
            if not margin_ok and place_ok:
                # identical-looking objects: the position decides, unless a
                # look-alike was also seen around this same spot
                margin_ok = not any(
                    l.session_id == self.session_id and l.world_angle is not None
                    and angle_diff(world_angle, l.world_angle) <= self.cfg.near_deg
                    for l in item.lookalikes)
        elif self.cfg.confirm_after_restart:
            # After a restart the heading starts again from wherever SENSE
            # faced at start-up, so the old place is only a hint. Opt-in only:
            # an identical cup that was never seen together with yours, standing
            # where yours used to be in the new heading, would be taken for it.
            place_ok = colourful and (item.world_angle is None or (
                world_angle is not None and angle_diff(world_angle, item.world_angle) <= self.cfg.near_deg))
        else:
            # Default after a restart: never confirm by itself. The answer says
            # "a cup that looks like yours"; "this is my cup" confirms it again.
            place_ok = False
        if look_ok and margin_ok and place_ok:
            return "match", sim
        return "lookalike", sim

    def end_frame(self, now, stable):
        """Decide new associations from this frame's evidence."""
        frame, self._frame = self._frame, []
        self._last_angles = {}
        for tid, category, _, angle, _ in frame:
            self._last_angles.setdefault(category, {})[tid] = angle
        if not frame:
            self._prune(now)
            return
        by_cat = {}
        for ev in frame:
            by_cat.setdefault(ev[1], []).append(ev)
        for category, evs in by_cat.items():
            items = [it for it in self.items.values() if it.category == category]
            if not items:
                continue
            if stable:
                self._check_resumed(evs)
                self._collect_lookalikes(evs, now)
            candidates = [ev for ev in evs if self.tracks[ev[0]].locked is None]
            if not candidates or not stable:
                continue          # blurry crops and a turning heading: don't decide anything
            # score every candidate against every free item
            proposals = {}        # tid -> (item name, score)
            for tid, _, sig, angle, _ in candidates:
                ts = self.tracks[tid]
                ts.lookalike_of = set()
                scored = []
                for item in items:
                    if self.held(item.name, now, except_tid=tid):
                        verdict, sim = self._verdict(item, sig, angle)
                        if verdict is not None:
                            ts.lookalike_of.add(item.name)   # yours is elsewhere: just similar
                        continue
                    verdict, sim = self._verdict(item, sig, angle)
                    if verdict == "match":
                        scored.append((sim, item.name))
                    elif verdict == "lookalike":
                        ts.lookalike_of.add(item.name)
                scored.sort(reverse=True)
                if len(scored) >= 2 and scored[0][0] - scored[1][0] < self.cfg.item_margin:
                    for _, n in scored:           # can't tell which of your cups this is
                        ts.lookalike_of.add(n)
                    scored = []
                if scored:
                    proposals[tid] = (scored[0][1], scored[0][0])
            # uniqueness: an item proposed by two tracks in one frame is ambiguous
            claims = {}
            for tid, (name, _) in proposals.items():
                claims.setdefault(name, []).append(tid)
            for name, tids in claims.items():
                if len(tids) > 1:
                    self.items[name].ambiguous_at = now
                    for tid in tids:
                        proposals.pop(tid)
                        self.tracks[tid].lookalike_of.add(name)
                        self.tracks[tid].votes.clear()
            # vote, and lock when enough recent checks agree
            for tid, _, _, _, _ in candidates:
                ts = self.tracks[tid]
                vote = proposals.get(tid, (None, 0))[0]
                ts.votes.append(vote)
                while len(ts.votes) > self.cfg.confirm_window:
                    ts.votes.popleft()
                if vote is None:
                    continue
                if sum(1 for v in ts.votes if v == vote) >= self.cfg.confirm_needed \
                        and not self.held(vote, now, except_tid=tid):
                    ts.locked = vote
                    ts.votes.clear()
                    ts.lookalike_of.discard(vote)
                    ts.moved_since_check = False
                    item = self.items[vote]
                    self._refresh(item, next(e[2] for e in candidates if e[0] == tid), ts.world_angle, now)
                    print(f"🔒 Re-identified '{vote}' on track {tid}")
        self._prune(now)

    def _check_resumed(self, evs):
        """After a turn, ByteTrack can hand a locked id to the cup next to it.
        If another track of the same kind is now clearly closer to where the
        item was before the turn, the lock is not trusted any more."""
        for tid, _, _, angle, _ in evs:
            ts = self.tracks[tid]
            ref, ts.resume_ref = ts.resume_ref, None
            if ref is None or ts.locked is None or angle is None:
                continue
            mine = angle_diff(angle, ref)
            for other_tid, _, _, other_angle, _ in evs:
                if other_tid != tid and other_angle is not None \
                        and angle_diff(other_angle, ref) + self.cfg.resume_slack_deg < mine:
                    item = self.items.get(ts.locked)
                    print(f"🔓 '{ts.locked}': track {tid} may have switched objects during the turn; label removed")
                    if item is not None:
                        item.world_angle = ref      # its place is still the one before the turn
                    ts.locked = None
                    ts.votes.clear()
                    break

    def _collect_lookalikes(self, evs, now):
        """Objects of the same kind in the SAME picture as a locked item are
        different physical objects: remember their look as look-alikes."""
        for tid, _, _, _, _ in evs:
            ts = self.tracks[tid]
            if ts.locked is None or ts.locked not in self.items:
                continue
            item = self.items[ts.locked]
            for other_tid, _, other_sig, other_angle, _ in evs:
                if other_tid != tid and other_sig is not None:
                    self._add_lookalike(item, other_sig, now, other_angle)

    def _add_lookalike(self, item, sig, now, world_angle=None, force=False):
        if sig is None:
            return
        if not force and now - item.last_lookalike_add < self.cfg.lookalike_every_s:
            return
        item.last_lookalike_add = now
        sig = np.asarray(sig, dtype=np.float32)
        session = self.session_id if world_angle is not None else None
        for l in item.lookalikes:
            s = histogram_similarity(sig, l.sig)
            if s is not None and s >= self.cfg.lookalike_merge:
                l.sig = (0.7 * l.sig + 0.3 * sig).astype(np.float32)
                if world_angle is not None:      # the same look-alike, seen again (maybe moved)
                    l.world_angle, l.session_id = world_angle, session
                self.dirty.add(item.name)
                return
        item.lookalikes.append(Lookalike(sig.copy(), world_angle, session))
        if len(item.lookalikes) > self.cfg.max_lookalikes:
            item.lookalikes.pop(0)
        self.dirty.add(item.name)

    def _prune(self, now):
        for tid in [t for t, ts in self.tracks.items() if now - ts.last_seen > 10.0]:
            del self.tracks[tid]

    # ------------------------------------------------------------------
    # Answering
    # ------------------------------------------------------------------
    def status(self, name, now):
        """What is visible right now that matters for a question about `name`."""
        item = self.items.get(name)
        if item is None:
            return None
        recent = [ts for ts in self.tracks.values()
                  if ts.category == item.category and now - ts.last_seen <= self.cfg.visible_s]
        mine = [ts for ts in recent if ts.locked == name]
        lookalikes = [ts for ts in recent if ts.locked != name and name in ts.lookalike_of]
        others = [ts for ts in recent if ts.locked != name and name not in ts.lookalike_of]
        return {
            "item": item,
            "visible": bool(mine),
            "lookalikes": lookalikes,
            "others": others,
            "ambiguous": item.ambiguous_at is not None and now - item.ambiguous_at <= self.cfg.ambiguous_s,
        }
