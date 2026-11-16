# "Where did I leave it?" (whereabouts/)

Ask **"where is my phone?"** after the phone has left the camera's view, even
after SENSE has been restarted, and SENSE answers from memory:

> "Your phone was on the table, to the left of the laptop, 10 minutes ago.
> It is probably still there, on your left. To face it, turn about 60 degrees to your left."

"Find my phone" starts its guidance with the same place words.

## What it adds to what SENSE already had

SENSE already remembered *when* and *in which direction* it last saw an item.
Two things were missing:

1. **Where in the room, in words that survive.** The old "next to the laptop"
   (`memory_relation()` in sense_master.py) compares positions saved at
   *different* times: it needs both items' last sightings within 30 s of each
   other. A laptop that is still on the desk keeps being seen, so its "last
   seen" is *now* and it was skipped. It also depends on the turn estimate
   (heading), which restarts at zero every run, so after a restart there was
   no relation at all.
2. **"On the table".** There was no support relation.

## How it works

Everything is decided from **one picture**, at the moment the item is seen
with the camera held still (`camera_is_stable`). Both boxes come from the same
frame, so the result does not depend on the heading and stays true after a
restart.

**Furniture** (`whereabouts/furniture.py`). Something resting on a table
touches it at the bottom of its box. If the item's **bottom-centre** lies in the
**upper part** of a table/desk/bed/couch box (upper 60% for a table or desk,
75% for a bed, 80% for a couch, whose box includes the backrest), not on the
box's edges, and the item is clearly smaller, it is "on" it. If the bottom is
in the lower part (legs, floor), the item is *in front of* the table and
nothing is claimed. If two different kinds of furniture qualify, nothing is
claimed either.

**Landmark** (`whereabouts/landmarks.py`). For every other object in the same
frame: skip people and pets (they move) and the item itself; skip it if the
boxes overlap a lot (in front? on top? a picture can't tell); skip it if the
bottoms are at very different heights (not on the same level); skip it if the
side gap is too big. The nearest remaining one wins, with a penalty for things
that move often (a cup is a worse landmark than a laptop). Overlapping
side-by-side boxes are "next to", otherwise "to the left of" / "to the right of"
(the camera's left is the user's left).

**Voting** (`whereabouts/voting.py`). One sample every 0.4 s; the last 5 are
kept, and a value counts only when **3 of 5 agree**. "Nothing there" is a value
too. If fewer than 3 agree, the stored anchor is kept. Votes restart when the
item moves (more than 12 degrees, the same threshold as events.py) or has not
been seen for 10 s.

**Storage** (`whereabouts/store.py`). One row per item in a new table
`item_anchors` inside `sense_memory.db`, keyed by the item's SENSE name ("my
bottle", "white cup"), never by ByteTrack ids (they restart every run).
Created with `CREATE TABLE IF NOT EXISTS` plus add-missing-columns; nothing
existing is dropped, reset or rewritten. "Forget my bottle" deletes its anchor
and its use as a landmark; "rename" carries it over.

**Answering** (`whereabouts/phrasing.py`, `tracker.py`). The anchor is used
only if it was confirmed around the item's last sighting (within 30 s). If the
item was seen later somewhere without a reliable anchor, the old answer is
given and SENSE adds: "Earlier, 15 minutes ago, it was on the table, ..., but
it may have moved since." The confidence tiers and the honest wording ("it
disappeared while I was watching, so it may have been moved", "I have not
looked there since", "before I was restarted") come from answers.py, as before.
With no anchor, the answer is exactly the old one.

## Alternatives considered

| Approach | Why not (for now) |
|---|---|
| Depth model (MiDaS) to know what is really on top | Extra model, heavy on a laptop CPU, every frame |
| Segmentation (yolov8-seg) for exact outlines | Second model; boxes are enough for "on / next to" |
| Relations from saved world angles (the old way) | Breaks when the landmark is still in view, and on restart |
| A local language model for the sentence | GBs of RAM, slow, can invent places |

A 2D box test is an approximation, not proof of contact. That is why
ambiguous geometry says nothing and why every relation is voted.

## Cost and limits

* No new threads (still main loop + voice listener), no network, no new model.
  Per frame: a few box comparisons for the handful of visible items; SQLite is
  written only when an anchor changes or every 5 s per item.
* Items: phone, remote, cup, bottle, book, laptop, mouse, and items you taught
  ("my bottle"). **Keys and wallets cannot be detected** (no class in the YOLO
  models), so don't demo with them.
* Surfaces: COCO's "dining table" (said as "table"), bed, couch. COCO has no
  "desk": a desk is usually detected as a table and said as "table". "Desk" is
  supported in the code for a detector that reports it.
* The camera must be held still for about 2 seconds for an anchor to form.
* An item in front of a table edge, seen from very low, can still look "on"
  it; voting reduces this but can't remove it.

## Tests

```
python -m pytest whereabouts/tests          # 52 tests: geometry, voting, storage, restart, answers
python tools/whereabouts_check.py PHOTO bottle bed --zoom 1.4 --centre 0.625 0.54
```
`whereabouts_check.py` runs the real sense_master.py loop twice on clips made
from a photo (see, turn away, ask; then restart with the same database and ask
again).

## Mac test (5 minutes)

Run `python sense_master.py`. Hold the camera still for 3 s at each step and
watch the terminal for a `📍 phone: on the table, to the left of the laptop` line.

1. Phone on a table beside a laptop (phone on the left). Turn away, ask
   "where is my phone?" → "on the table, to the left of the laptop".
2. Remove the laptop, put the phone back alone on the table. Turn away, ask → "on the table".
3. Phone next to a bottle. Turn away, ask → "...the bottle".
4. Move the phone somewhere else, hold still, turn away, ask → the new place, not the old one.
5. Turn away and back with nothing moved → the answer stays the same.
6. Quit (q), start again, ask before showing the phone → the same place words,
   with "before I was restarted" or "I have not looked there since".
7. "Find my phone" → guidance starts with the same place words.
