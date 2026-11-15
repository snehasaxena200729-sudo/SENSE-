# "What changed since I left?": how the scene change summary works

Before you go, say **"I'm leaving"**:

> "Okay, I remembered 3 things: your bottle, the white cup and the chair. When you're back, ask me
> what changed."

When you're back, say **"What changed since I left?"**, **"I'm back"** or just **"What changed?"**:

> "Since you left 12 minutes ago: Your bottle was moved from in front of you to on your right. The
> white cup is no longer where it was. New: the laptop on your left. Sneha was here. Everything else
> is where it was."

If you forgot to say "I'm leaving", SENSE compares with **how the room was when it last ran** (it
saves a snapshot every minute and when you quit), so "what changed since I left?" also works after
a restart.

Code: `scene_diff.py` (the logic, no camera or database), `mark_scene()` / `answer_since_left()` in
`sense_master.py` (the wiring), snapshot tables in `memory_db.py`. Tests: `tests/test_scene_diff.py`
and the `scene_change.mp4` / `scene_after.mp4` scenarios in `tools/run_checks.py`.

## The idea: snapshot, then compare

```
"I'm leaving" ──► snapshot = every item remembered in this run, with its world angle
                    (SQLite: scene_snapshots + snapshot_items)

 ... SENSE keeps running (or is restarted). The usual memory keeps updating:
     environmental_memory = latest place of each item, events = what happened ...

"What changed?" ──► for each item in the snapshot:
                       seen again?     ── same place (within 12°) ──► same
                                       └─ elsewhere ─────────────────► moved
                       not seen again? ── its spot was looked at ────► no longer where it was
                                       └─ never looked there ────────► "I haven't looked where … was yet"
                    items seen since that are not in the snapshot ───► new
                    people recognised by face since then ────────────► "Sneha was here"
```

Every item gets exactly one answer, and the order of the spoken answer is what you most likely need:
moved, missing, new, visitors, then what SENSE could not check.

## Three things that make it harder than a set difference

**1. "Not seen" is not "gone".** If you never point the camera at the bottle's spot, the bottle may
still be there. `ViewCoverage` keeps 72 numbers: how many seconds of *steady* view each 5° direction
got since the snapshot. An item is reported missing only if its spot was well inside the view (not
at the edges, where detections are often missed) for at least 3 seconds. Otherwise SENSE says
"I haven't looked where your bottle was yet". The wording is "no longer where it was" instead of
"gone", because the item may simply have been moved somewhere SENSE has not looked.

*Alternative:* store a full image of the room and compare pixels. Cheaper to explain, but a change
of light or a shadow would look like a change, and it says nothing about *which* object changed.

**2. The camera may have been turned between runs.** Directions are stored as world angles measured
from where the camera pointed when SENSE started. Within one run they are comparable. After a
restart they are only comparable if the laptop is in exactly the same position. So SENSE
*estimates* the rotation: every item seen in both runs votes "the camera turned by d degrees". Most
things in a room stay put, so the biggest group of agreeing votes is the rotation, and the items
outside that group are the ones that moved. This is **consensus voting**, a one-dimensional form of
RANSAC (the algorithm used to stitch panoramas).

Rules, chosen with the randomised test: at least 3 agreeing items and a majority (or, with only 2
items, both agreeing). The test found that 2 out of 3 is not safe: two moved items can agree by
chance and out-vote the one that stayed. Without enough agreement SENSE makes no claims about
position and says "I can't tell if anything was moved, because the camera may have been turned
since then".

**3. The same object can come back under a new name.** SENSE names objects by colour and class
("yellow dog"). When an object reappears far from where it was, `identity.py` may treat it as a new
one ("yellow dog 2"), which is exactly what a move looks like. So a missing item and a new item with
the same noun are paired and reported as one move. Known limit: "your cup was taken away and a
different cup was put somewhere else" is reported as the cup having moved.

Flickering detections (something YOLO sees only now and then) would otherwise be reported as "new"
every time. Items that disappeared in the two minutes before the snapshot are kept in it, marked as
"seen shortly before": they are never called new or missing.

## Constraints

- **Offline, no new packages:** plain Python + SQLite.
- **No new thread:** the snapshot is a few SQL rows written on the main thread when you speak, plus
  one automatic snapshot per minute (a single small transaction). Coverage is updated with one
  addition per frame.
- **Privacy:** "Forget my bottle" and "Rename …" also update the snapshots. People are never stored
  in snapshots; visitors are taken from the face-verified event history.
- **Database:** the newest 20 snapshots are kept; older ones are deleted.

## Limits to know for the demo

- Moves are measured as a **direction** (angle), not a distance: an item moved straight towards or
  away from the camera is not detected as moved. 12° is about a fifth of the view.
- Only things SENSE can detect and remembers (the classes in `ALLOWED_ITEMS` and taught items) are
  compared. Keys, papers, or a door left open are not.
- Across a restart with fewer than 3 items in view, SENSE can say what is still there and what it
  has not seen, but not what moved (see point 2).
