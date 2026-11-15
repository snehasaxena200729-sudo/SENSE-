"""
scene_diff.py - "What changed since I left?" (scene change summary)

IDEA: take a SNAPSHOT of the room (every remembered item and where it was),
and later compare it with what SENSE has seen since. Each item in the
snapshot ends up in exactly one bucket:

    moved       seen again, more than MOVE_DEG away from where it was
    same        seen again at (about) the same place
    gone        not at its spot any more, and SENSE has really LOOKED at that spot
                (said as "no longer where it was": it may have been moved to a
                place SENSE has not looked at yet)
    unchecked   not seen again, but SENSE has not looked at its spot yet
                (or cannot tell where its spot is) - we say so honestly
and anything seen since that was not in the snapshot is
    appeared    new in the room

Two problems make this harder than a set difference:

1) THE CAMERA MAY HAVE MOVED BETWEEN RUNS.
   World angles are measured from the heading the camera had when SENSE
   started. Within one run they are comparable. Across runs they are only
   comparable if the laptop was put back pointing the same way - which we
   cannot assume. So align_sessions() ESTIMATES the rotation between the two
   runs from the items themselves: every item seen in both gives one vote
   "the camera turned by d degrees". Most things in a room do not move, so the
   largest group of votes that agree (within ALIGN_TOL_DEG) is the true
   rotation and the items outside that group are the ones that moved.
   This is consensus voting, a 1-D version of RANSAC. It needs at least two
   agreeing items: with only one we could not tell "the camera turned" from
   "that item moved", so then we make NO position claims at all.

2) "NOT SEEN" IS NOT "GONE".
   If the user never pointed the camera at the bottle's spot, the bottle may
   well still be there. ViewCoverage remembers how many seconds of steady
   view each direction got. Only an item whose spot was in steady view for
   CHECK_S seconds (and which was not seen) is reported as gone.

Pure logic: no camera, no database, no clock. sense_master.py feeds it rows
from SQLite and the heading, which keeps it easy to unit-test.
"""

import spatial

MOVE_DEG = 12.0          # same threshold as events.py MOVED (about 1/5 of the view)
ALIGN_TOL_DEG = 8.0      # votes this close count as "the same camera rotation"
CHECK_S = 3.0            # steady view of a spot needed before we call an item gone
EDGE_MARGIN_DEG = 8.0    # items near the frame edge are often missed: don't count them
COVERAGE_BIN_DEG = 5.0
ALIGNED_EXTRA_MARGIN_DEG = 5.0   # across runs the rotation is only estimated: be stricter


def speakable(name):
    """'yellow dog 2' -> 'yellow dog' (identity.py numbers look-alike objects)."""
    words = name.split()
    while len(words) > 1 and words[-1].isdigit():
        words.pop()
    return " ".join(words)


def base_noun(name):
    """'white cup' -> 'cup', 'my water bottle' -> 'bottle', 'yellow dog 2' -> 'dog'."""
    words = [w for w in name.lower().split() if not w.isdigit()]
    return words[-1] if words else ""


# ----------------------------------------------------------------------
# Which directions has SENSE actually looked at?
# ----------------------------------------------------------------------
class ViewCoverage:
    """Seconds of steady view per heading bin (72 bins of 5 degrees).

    add() is called once per steady frame with the frame's duration, so the
    cost is O(1) per frame and the memory is 72 numbers, however long it runs.
    """

    def __init__(self):
        self.bins = [0.0] * int(360 / COVERAGE_BIN_DEG)

    def reset(self):
        self.bins = [0.0] * len(self.bins)

    def add(self, heading, dt):
        if dt <= 0 or dt > 1.0:          # ignore stalls (e.g. the app was paused)
            return
        k = int(((heading % 360.0) // COVERAGE_BIN_DEG)) % len(self.bins)
        self.bins[k] += dt

    def seconds_on(self, world_deg, fov=spatial.CAMERA_FOV_DEG, margin=EDGE_MARGIN_DEG):
        """How long was world_deg comfortably inside the view?

        A bin stands for headings up to half a bin away from its centre, so
        half a bin is subtracted too: rounding must never make us claim we
        looked at a spot we did not."""
        half = fov / 2.0 - margin - COVERAGE_BIN_DEG / 2.0
        total = 0.0
        for k, s in enumerate(self.bins):
            if s and abs(spatial.wrap_angle(world_deg - (k + 0.5) * COVERAGE_BIN_DEG)) < half:
                total += s
        return total

    def checked(self, world_deg, extra_margin=0.0):
        return self.seconds_on(world_deg, margin=EDGE_MARGIN_DEG + extra_margin) >= CHECK_S


# ----------------------------------------------------------------------
# Rotation between two runs (consensus voting)
# ----------------------------------------------------------------------
def align_sessions(pairs, tol=ALIGN_TOL_DEG):
    """pairs: [(name, old_angle, new_angle)] for items seen in both runs.

    Returns (offset, inlier_names) with new = old + offset, or (None, set())
    when the evidence is too weak (fewer than 2 agreeing items, or no clear
    majority).
    """
    if len(pairs) < 2:
        return None, set()
    votes = [(n, spatial.wrap_angle(b - a)) for n, a, b in pairs]
    best = None
    for _, d in votes:      # try every vote as the candidate rotation
        inliers = [(n, v) for n, v in votes if abs(spatial.wrap_angle(v - d)) <= tol]
        spread = sum(abs(spatial.wrap_angle(v - d)) for _, v in inliers)
        if best is None or (len(inliers), -spread) > (len(best), -best_spread):
            best, best_spread = inliers, spread
    if len(best) * 2 <= len(votes) or len(best) < min(3, len(votes)) or len(best) < 2:
        # Need a strict majority (2 vs 2 is a tie between two explanations),
        # and 3 agreeing items - or, with only 2 items seen, both agreeing.
        # 2 out of 3 is not enough: two moved items can agree by chance
        # (the randomised test found exactly that), and then the one item
        # that did NOT move would be reported as moved.
        return None, set()
    ref = best[0][1]
    offset = spatial.wrap_angle(ref + sum(spatial.wrap_angle(v - ref) for _, v in best) / len(best))
    return offset, {n for n, _ in best}


# ----------------------------------------------------------------------
# The comparison
# ----------------------------------------------------------------------
def compare(ref_items, now_rows, coverage=None, offset=None, offset_estimated=False):
    """Compare a snapshot with what has been seen since.

    ref_items: [{name, zone, world_angle}] - the snapshot
    now_rows:  {name: {zone, world_angle, status, comparable}} - items seen
               AFTER the snapshot. comparable=True means its world_angle is
               in the snapshot's frame once `offset` is added to the old angle
               (same run: offset 0; another run: the aligned offset).
    coverage:  ViewCoverage for the current run, or None (no spot checks)
    offset:    rotation from snapshot frame to the current frame, or None if
               unknown (then nothing can be called moved/gone by position)
    offset_estimated: True when the offset came from align_sessions() (another
               run): spot checks then use a wider safety margin

    Returns dict of lists: moved, same, gone, unchecked, appeared.
    moved items carry old_angle/new_angle (current frame) when known.
    """
    out = {"moved": [], "same": [], "gone": [], "unchecked": [], "appeared": []}
    ref_names = set()
    missing = []            # snapshot items not seen since, or seen leaving

    for item in ref_items:
        name = item["name"]
        ref_names.add(name)
        row = now_rows.get(name)
        if (item.get("status") or "").upper() == "GONE":
            # in the snapshot only as "seen shortly before" (it was flickering or
            # had just left): it can't be reported gone, but seeing it now is not
            # news either - it was in the room when the user left
            continue
        if row is None:
            missing.append(item)
        elif (row.get("status") or "").upper() == "GONE":
            missing.append(dict(item, watched=True))     # the event tracker saw it leave
        else:
            _place(out, item, row, offset)

    new_rows = {n: r for n, r in now_rows.items()
                if n not in ref_names and (r.get("status") or "").upper() != "GONE"}

    # Same object, new name: 'white cup' then 'cup' (the colour vote changed), or
    # 'yellow dog' then 'yellow dog 2' (identity.py starts a new entity when an
    # object shows up far from where it was - which is exactly what a MOVE looks
    # like). A missing item and a new item with the same noun are most likely
    # the same object. Greedy pairing by angle; a perfect matching (Hungarian
    # algorithm) would be overkill for the handful of items in one room.
    # Known limit: "your cup was taken and another cup put elsewhere" is
    # reported as the cup being moved.
    for item in missing:
        same_noun = [n for n in new_rows if base_noun(n) == base_noun(item["name"])]
        if same_noun:
            def gap(n):
                a, b = item.get("world_angle"), new_rows[n].get("world_angle")
                if a is None or b is None or offset is None:
                    return 0.0
                return abs(spatial.wrap_angle(b - (a + offset)))
            match = min(same_noun, key=gap)
            row = new_rows.pop(match)
            item = {k: v for k, v in item.items() if k != "watched"}
            _place(out, dict(item, now_name=match), row, offset)
            continue
        if item.get("watched"):
            out["gone"].append(dict(item, how="watched"))
            continue
        expected = None
        if offset is not None and item.get("world_angle") is not None:
            expected = spatial.wrap_angle(item["world_angle"] + offset)
        margin = ALIGNED_EXTRA_MARGIN_DEG if offset_estimated else 0.0
        if coverage is not None and expected is not None and coverage.checked(expected, margin):
            out["gone"].append(dict(item, how="looked"))
        else:
            # not_looked: we know where its spot is, the camera has not been there
            # unknown_spot: across runs without alignment we don't know where to look
            out["unchecked"].append(dict(item, why="not_looked" if expected is not None else "unknown_spot"))

    for n, r in new_rows.items():
        out["appeared"].append(dict(r, name=n))
    return out


def _place(out, item, row, offset):
    a, b = item.get("world_angle"), row.get("world_angle")
    if offset is None or a is None or b is None or not row.get("comparable", True):
        out["same"].append(dict(item, now=row, unsure_position=True))
        return
    old_now_frame = spatial.wrap_angle(a + offset)
    if abs(spatial.wrap_angle(b - old_now_frame)) > MOVE_DEG:
        out["moved"].append(dict(item, now=row, old_angle=old_now_frame, new_angle=b))
    else:
        out["same"].append(dict(item, now=row))


# ----------------------------------------------------------------------
# Spoken summary
# ----------------------------------------------------------------------
def _join(parts):
    if len(parts) <= 1:
        return "".join(parts)
    return ", ".join(parts[:-1]) + " and " + parts[-1]


def _cap(parts, limit):
    if len(parts) > limit:
        return parts[:limit] + [f"{len(parts) - limit} more"]
    return parts


def summarise(diff, heading, label, since_words, people=None, positions_known=True, limit=3):
    """Turn compare()'s result into one short spoken answer.

    label(name) -> spoken name ('your bottle', 'the white cup').
    heading: current heading, to say where things are relative to the user NOW.
    people: names of known people who came by, already spoken form.
    Order = what the user most likely needs: moved, gone, new, people, then
    what we could not check. Capped so the answer stays under ~20 seconds.
    """
    def where(angle):
        return spatial.direction_phrase(spatial.relative_angle(angle, heading))

    sentences = []
    moved = []
    for m in diff["moved"]:
        name = label(m["name"])
        if m["old_angle"] is not None and m["new_angle"] is not None:
            old_w, new_w = where(m["old_angle"]), where(m["new_angle"])
            if old_w != new_w:
                moved.append(f"{name} was moved from {old_w} to {new_w}")
            else:
                moved.append(f"{name} was moved a little, it is still {new_w}")
        else:
            moved.append(f"{name} was moved")
    if moved:
        sentences.append(_join(_cap(moved, limit)))

    # "no longer where it was", not "gone": the spot is empty, but the item may
    # have been moved somewhere SENSE has not looked yet
    gone = [label(g["name"]) for g in diff["gone"]]
    if gone:
        tail_words = "is no longer where it was" if len(gone) == 1 else "are no longer where they were"
        sentences.append(f"{_join(_cap(gone, limit))} {tail_words}")

    new = []
    for a in diff["appeared"]:
        if a.get("comparable", True) and a.get("world_angle") is not None and positions_known:
            new.append(f"{label(a['name'])} {where(a['world_angle'])}")
        else:
            new.append(label(a["name"]))
    if new:
        sentences.append("new: " + _join(_cap(new, limit)))

    if people:
        verb = "was" if len(people) == 1 else "were"
        sentences.append(f"{_join(people)} {verb} here")

    changed = bool(sentences)
    if not changed:
        if diff["same"] and not positions_known:
            body = "Everything I have seen again is still here"
        elif diff["same"]:
            body = "Nothing has changed that I can see"
        else:
            body = "I can't see anything I remember yet"
    else:
        body = ". ".join(s[0].upper() + s[1:] for s in sentences)

    tail = []
    not_looked = [label(u["name"]) for u in diff["unchecked"] if u.get("why") != "unknown_spot"]
    unknown = [label(u["name"]) for u in diff["unchecked"] if u.get("why") == "unknown_spot"]
    if not_looked:
        tail.append(f"I haven't looked where {_join(_cap(not_looked, limit))} "
                    f"{'was' if len(not_looked) == 1 else 'were'} yet")
    if unknown:
        tail.append(f"I haven't seen {_join(_cap(unknown, limit))} since then")
    if not positions_known and (diff["same"] or diff["moved"]):
        tail.append("I can't tell if anything was moved, because the camera may have been turned since then")
    elif changed and diff["same"]:
        tail.append("everything else is where it was")

    text = f"Since {since_words}: {body}."
    for t in tail:
        text += " " + t[0].upper() + t[1:] + "."
    return text


# ----------------------------------------------------------------------
# Choosing the reference snapshot, and putting rows into one frame
# ----------------------------------------------------------------------
def choose_reference(snapshots, current_session, last_answer_ts=0.0):
    """Which snapshot does "since I left" mean? snapshots: newest first.

    - a 'mark' ("I'm leaving") is the user's own reference point
    - an 'auto' snapshot of the CURRENT run is never used (that's not "before")
    - an 'auto' snapshot of a run that also has an unanswered mark is skipped:
      SENSE kept watching after the user left, so its end state already
      contains the changes the user wants to hear about
    """
    open_marks = {s["session_id"] for s in snapshots
                  if s["kind"] == "mark" and s["ts"] > (last_answer_ts or 0.0)}
    for s in snapshots:
        if s["kind"] == "auto" and (s["session_id"] == current_session or s["session_id"] in open_marks):
            continue
        return s
    return None


def prepare(ref, ref_items, rows, current_session, is_person=lambda n: False):
    """Turn memory rows into compare() input in ONE frame (the current run's).

    rows: environmental_memory rows (name, world_angle, session_id, status, last_seen, zone)
    Returns (ref_items, now_rows, offset, positions_known).
    """
    ref_items = [i for i in ref_items if not is_person(i["name"])]
    after = [r for r in rows if r["last_seen"] is not None and r["last_seen"] > ref["ts"]
             and not is_person(r["name"])]

    if ref["session_id"] == current_session:
        offset = 0.0                     # same run: the heading is shared
    else:
        old = {i["name"]: i.get("world_angle") for i in ref_items
               if (i.get("status") or "").upper() != "GONE"}
        pairs = [(r["name"], old[r["name"]], r["world_angle"]) for r in after
                 if r.get("session_id") == current_session and r.get("world_angle") is not None
                 and old.get(r["name"]) is not None and (r.get("status") or "").upper() != "GONE"]
        offset, _ = align_sessions(pairs)

    now_rows = {}
    for r in after:
        angle = r.get("world_angle")
        if r.get("session_id") == current_session:
            comparable = angle is not None
        elif r.get("session_id") == ref["session_id"] and offset is not None and angle is not None:
            angle = spatial.wrap_angle(angle + offset)   # old run's frame -> current frame
            comparable = True
        else:
            comparable = False
        now_rows[r["name"]] = {"zone": r.get("zone"), "world_angle": angle if comparable else None,
                               "status": r.get("status"), "comparable": comparable,
                               "last_seen": r["last_seen"]}
    return ref_items, now_rows, offset, offset is not None
