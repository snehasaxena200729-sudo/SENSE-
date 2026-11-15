"""
people.py - What SENSE is allowed to CALL a person (audit fix 1).

THE PROBLEM THIS FIXES
----------------------
Three different things were mixed up:

  1. Person DETECTION  (YOLO):      "there is a person-shaped thing in this box"
  2. Person TRACKING   (ByteTrack): "this box is the same box as last frame" -> track id 7
  3. Person IDENTITY   (face recognition): "this is Sneha"

YOLO and ByteTrack only give you 1 and 2. A track id says nothing about WHO it
is: if Sneha walks out and back in she gets a new id, and two people crossing
can even swap ids. The old code filled in 3 with a colour histogram of the
person's SHIRT, so anybody in similar clothes was announced as "Sneha".

THE RULE NOW
------------
A person is "a person" unless a real identity check (face recognition, see
face_id.py) has VERIFIED a name on this track. This class is the single gate
every spoken or displayed person name has to pass through.

Voting (why not trust one frame?): a single face match can be wrong (blur, a
side view). We keep the last WINDOW checks per track and lock a name only when
NEEDED of them agree on the same name and none of them disagrees. A clear face
that matches nobody, or a different name, counts against the lock; frames with
no usable face (turned away) count neither way, so the name stays on the track
while you look away, and is dropped when the track ends.
"""

from collections import deque

NO_FACE = None          # observation: no usable face this time (no information)
UNKNOWN = "?unknown"    # observation: a clear face that matches nobody enrolled

GENERIC = "a person"


class PersonIdentityGate:
    WINDOW = 5
    NEEDED = 3

    def __init__(self):
        self.history = {}   # tid -> deque of recent verified results (names or UNKNOWN)
        self.locked = {}    # tid -> verified name
        self.last_seen = {} # tid -> time the track was last reported by ByteTrack

    def observe(self, tid, result):
        """Record one identity check for a track.

        result: NO_FACE (ignored), UNKNOWN, or an enrolled name that the face
        recogniser has already accepted (similarity >= threshold AND margin).

        Lock:    >= NEEDED of the last WINDOW checks say this name, and none
                 says a DIFFERENT enrolled name. (An earlier rule demanded all
                 WINDOW agree; one borderline frame then blocked the name.)
        Release: >= NEEDED of the last WINDOW checks are clear faces that are
                 NOT this name (someone else, or ByteTrack swapped ids).
        """
        if result is NO_FACE:
            return self.label(tid)
        hist = self.history.setdefault(tid, deque(maxlen=self.WINDOW))
        hist.append(result)
        name = self.locked.get(tid)
        if name is not None:
            if sum(1 for r in hist if r != name) >= self.NEEDED:
                del self.locked[tid]
                name = None
        if name is None:
            names = {r for r in hist if r != UNKNOWN}
            if len(names) == 1:
                candidate = names.pop()
                if sum(1 for r in hist if r == candidate) >= self.NEEDED:
                    self.locked[tid] = candidate
        return self.label(tid)

    def reset_track(self, tid):
        """Forget the identity checks of one track (e.g. right after enrolment,
        so 'unknown' results from before do not count against the new name)."""
        self.history.pop(tid, None)
        self.locked.pop(tid, None)

    def label(self, tid):
        """The verified name for this track, or None."""
        return self.locked.get(tid)

    def spoken(self, tid):
        """What to SAY for this track: the verified name or 'a person'."""
        return self.label(tid) or GENERIC

    def tracks_named(self, name):
        return [t for t, n in self.locked.items() if n.lower() == name.lower()]

    def rename(self, old, new):
        for t, n in list(self.locked.items()):
            if n.lower() == old.lower():
                self.locked[t] = new
        for hist in self.history.values():
            for i, r in enumerate(hist):
                if isinstance(r, str) and r.lower() == old.lower():
                    hist[i] = new

    def forget_name(self, name):
        for t in self.tracks_named(name):
            del self.locked[t]
        for hist in self.history.values():
            kept = [r for r in hist if not (isinstance(r, str) and r.lower() == name.lower())]
            hist.clear()
            hist.extend(kept)

    def touch(self, tid, now):
        """Call every frame a person track is reported."""
        self.last_seen[tid] = now

    def drop_track(self, tid):
        self.history.pop(tid, None)
        self.locked.pop(tid, None)
        self.last_seen.pop(tid, None)

    def prune(self, now, max_age_s=3.0):
        """Forget tracks ByteTrack stopped reporting. A name never jumps to a
        new track id: a returning person must be verified again."""
        for t in [t for t, ts in self.last_seen.items() if now - ts > max_age_s]:
            self.drop_track(t)


def count_phrase(n):
    """'a person' / '2 people' for the overview sentence."""
    return GENERIC if n == 1 else f"{n} people"
