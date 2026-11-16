# "My cup" is one cup: personal item identity

## What was wrong

When you said "this is my cup", SENSE saved just two things: the category
(cup) and one colour histogram. From then on, `match_custom_object()` called
**any** cup "my cup" if its colours correlated above **0.55** with that
histogram. Two white cups pass that easily, because the colour signature
ignores brightness and white, grey and black all look alike to it. Nothing
stopped two cups from both being "my cup", and both wrote to the same memory
row. So after you turned 45 degrees to a second cup, the saved place of "my
cup" moved to the wrong cup.

Measured with the real app loop (`tools/personal_e2e.py`), with two white cups
45 degrees apart:

| | main (before) | this branch |
|---|---|---|
| Turned to Cup B, "where is my cup?" | "I see your cup in front of you" (that is Cup B) | "...I can't see it right now. There is a cup in front of you that looks like yours, but I can't be sure it is yours." |
| Restarted facing Cup B | "I see your cup in front of you" | same honest answer |
| Turned back to Cup A | "I see your cup" | "I see your cup" |

## The idea: an instance, and a gate

A **category** ("cup") is what YOLO tells us. An **instance** ("the cup I
showed you") has to be tracked by SENSE itself. The new `personal/` package
stores each personal item as an instance with:

- **its look**: the taught colour signature, refined over time by averaging
  frames it is sure about. A single frame is noisy: a white cup's correlation
  with *itself* jumps between about 0.5 and 0.98.
- **its place**: the world angle, which is the direction in the room, like
  "40 degrees to the left of where you started".
- **its look-alikes**: other objects of the same kind that were in the same
  picture as it, so they are provably different objects.

A ByteTrack id is only a short-term handle. While the same track stays in view,
it keeps the label. A *new* track only becomes "my cup" if every check of a
gate passes. This is the standard re-identification pattern, gated
nearest-neighbour data association:

1. **Look**: similarity of at least 0.75, up from 0.55, using the track's
   average look.
2. **Place**: within 25 degrees of where your cup was last seen in this run.
3. **Margin**: the track looks clearly more like your cup than like any known
   look-alike. When the two look identical, position decides, unless a
   look-alike was also seen at that spot.
4. **Unique**: your cup isn't on another visible track, and no second cup
   passes in the same frame. If two do, neither is named.
5. **Persistent**: 5 of the last 8 checks of the track agree.

If the look is close but a check fails, the cup stays "a cup" and is
remembered as "looks like yours". The answers say so.

**Guards against ByteTrack mistakes:**
- A track that disappears for more than 1.5 s loses its label, because ids
  can come back on a different object.
- During a turn, a locked track that jumps more than 25 degrees from the
  item's place loses its label. So does one with another cup clearly closer
  to that place. This catches id switches.
- A track that keeps looking unlike the item for 6 sharp frames loses its
  label.

**After a restart**, the heading starts again from wherever the camera faces,
so places from the last run can't be trusted. By default SENSE then never
confirms an item by itself. It says "there is a cup that looks like yours. If
it is, say: this is my cup." Saying that re-binds the same identity: the
instance id and look-alikes are kept. `PersonalConfig.confirm_after_restart =
True` turns on a riskier mode that recognises colourful items at their old
place.

## Questions

- **"Where is my cup?"**, "have you seen my bag?", "when did you last see my
  laptop?", "where's my mug?": SENSE answers about *that* instance. If it can't
  see it, it adds what else is in view: "I can see another cup on your right,
  but it doesn't look like yours."
- **Two personal bottles, and you say "my bottle"**: SENSE asks "Do you mean
  your medicine bottle or your water bottle?"
- **"My phone" was never taught**: SENSE says "I don't know which phone is
  yours yet...". It does not answer about whichever phone it saw last.
- **Teaching with two cups of about the same size in view**: SENSE says "hold
  yours closer" instead of guessing.

## What it can't do

These limits are honest for any camera-only system:

- If your cup is taken away without SENSE seeing it go, and an identical cup is
  put in exactly the same spot, it will be taken for yours.
- If someone moves your cup to another direction, SENSE will call it "a cup
  that looks like yours" until you say "this is my cup" again. That's the safe
  direction to be wrong in.

## Files

| file | what |
|---|---|
| `personal/config.py` | every threshold, with the reason for its value |
| `personal/matcher.py` | the gate (pure logic, no camera or database) |
| `personal/store.py` | the `personal_instances` table (additive; no existing table changed) |
| `personal/phrasing.py` | "my <kind>" in questions, and the honest extra sentence |
| `personal/manager.py` | `PersonalItems`, the only class `sense_master.py` talks to; errors are contained, and if it can't start, the old matching is used |
| `tests/test_personal_items.py` | 28 tests: TEST 1-6 from the request, plus id switches, re-teach, restart and questions |
| `tools/stress_personal.py` | 10,000 random rooms (in `run_checks.py`) |
| `tools/personal_e2e.py` | the real app loop on drawn cup videos (about 4 minutes) |

There are no new threads, no new dependencies and no network calls. YOLO,
ByteTrack, memory_db, face, distance and audio code are unchanged.
