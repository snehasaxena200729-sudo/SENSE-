# "Find my bottle": how guidance mode works

Say **"Find my bottle"** and SENSE talks you onto it:

> "Found your bottle, a little to your left, a few steps away."
> *(you turn)* "Straight ahead, a few steps away."
> *(you walk)* "Straight ahead, close."
> "Your bottle should be within reach, straight ahead. Reach out slowly."

If the bottle is out of view, SENSE uses its memory:

> "Looking for your bottle. I can't see it right now. I last saw it 2 minutes ago, on your left,
> next to the laptop. Turn about 90 degrees to your left, slowly."
> *(you turn)* "Keep turning left." … "Found your bottle, a little to your right."

Other phrasings: "help me find my phone", "take me to the door", "guide me to the chair", "look for
the cup". Stop with "stop looking", "cancel" or "I found it".

Code: `guidance.py` (the logic, no camera or audio) and `start_find()` / the main loop in
`sense_master.py` (the wiring). Tests: `tests/test_guidance.py`, `tools/stress_guidance.py`, and the
`find_item.mp4` scenario in `tools/run_checks.py`.

## The pipeline

```
 microphone ──► Whisper ──► "find my bottle" ──► intent FIND ──► start_find()
                                                                   │
                       which bottle? (fuzzy name match, taught item, person or door)
                                                                   │
   every camera frame:                                             ▼
   YOLO + ByteTrack ──► confirmed sightings ──► matching ones ──► guide.observe()
   door model       ──► continuity.py doors ──┘                     │
   optical flow ─────► heading (how far you turned) ───────────────┤
                                                                   ▼
   notifier (hazards first) ──► is it safe to talk? ──► guide.poll() ──► `say`
```

Nothing new runs in parallel: guidance is a few function calls inside the existing main loop. SENSE
still has exactly **two** threads (main loop + voice listener), stays offline, and needs no new
package.

## The ideas, one at a time

**1. Direction = angle in the image.** A camera is a pinhole: a pixel column maps to an angle,
`angle = (x / width − 0.5) × 65°` (`spatial.pixel_to_angle`). Three zones: under 10° is "straight
ahead", 10 to 25° is "a little to your left/right", beyond that "on your left/right".

**2. Hysteresis (stops flickering).** A bottle sitting at exactly 10° would flip between "straight"
and "a little left" every frame because detector boxes jitter by a few pixels. So *entering* a zone
uses the border, but *leaving* it needs 4° more. Thermostats do the same: heat on at 19°, off at 21°.

**3. Debounce (one bad box can't speak).** A new instruction must be seen on 2 detections in a row
before it is said. One wrong box from YOLO is ignored.

**4. Median of 3 for distance.** The words "a few steps / close / within reach" come from the median
of the last 3 distance estimates. A median ignores one outlier; an average would be pulled by it.

**5. Rate limiting.** At least 1.5 s between cues. The same cue is repeated at most every 4 s, as
reassurance. A new cue is said only when something changed: a new direction, or you got closer.
"Within reach" may come after 0.8 s, because walking on would mean bumping into it.

**6. Safety first (priority arbitration).** Hazard warnings go through `notifier.py` with priority
P1. Guidance asks `notifier.safety_clear()` before every cue and only speaks when no warning is
waiting and none was said in the last 2 s. It also never speaks while SENSE is talking or listening.
So a guidance cue can never drown out "Careful, a chair in front of you", but a warning can cut off a
guidance cue. While guiding, non-urgent news ("your phone was moved") waits.

**7. Memory fallback with dead reckoning.** When SENSE sees something it stores a *world angle*:
`image angle + heading`. The heading is how far you have turned since start, measured with optical
flow (`calcOpticalFlowPyrLK` on background points). Later, `world angle − heading now` says where it
is relative to you *now*, so "turn about 90 degrees to your left" stays correct as you turn ("keep
turning left"). If you overshoot, the side flips to right.

**8. A small state machine.**

```
 SEARCHING ──target seen──► GUIDING ──close enough, twice──► done ("within reach")
     ▲                         │
     └──── not seen for 2 s ───┘ ("I lost sight of it. It was on your left.")

 Limits: 45 s searching without a sighting → "I couldn't find it". 2 minutes in total.
```

**9. Honest "within reach".** It needs the distance estimate to say under 0.8 m on two detections
in a row *and* the item to be roughly ahead (within 20°). If SENSE cannot estimate the distance it
never says "within reach". A big thing (bed, door) close up is wider than the view, so its size
only gives an upper bound; "it fills the view from side to side" is then used instead, and the words
change to "is right in front of you. Slow down."

**10. Which bottle?** If two bottles are visible, SENSE keeps guiding to the one nearest the world
angle it was guiding to (or remembered), so it doesn't jump between them. Taught items ("my bottle")
only match that item; "a bottle" matches any. People are matched only by verified face.

## Alternatives considered

| Option | Why not (for now) |
| --- | --- |
| Beeps that speed up as you get closer (like a parking sensor) | Fast and language-free, but needs a tone generator and teaching the user what the tones mean. A good next step for noisy places (roadmap: haptic or tone cues) |
| Stereo panning (sound comes from the left) | Needs headphones and an audio library: a new dependency |
| A local language model for the wording | Seconds of latency on a laptop CPU, and it can invent things. Fixed sentences are fast and predictable |
| Kalman filter on the target's angle | More accurate smoothing, but the hysteresis + debounce + median is enough for spoken words and easier to explain and test |
| Speak on every frame | Floods the user; 15 sentences a second can't be followed |

## Known limitations (be upfront about these in the demo)

- **Heading drifts.** Optical-flow heading adds up small errors; after a lot of turning the memory
  angle can be off by tens of degrees. That is why the memory sentence says "about" and "slowly".
  It also resets each run, so memory from an earlier run only gives the old left/front/right.
- **Distance is coarse** until calibrated (`tools/validate_distance.py`).
- **Only what YOLO knows.** Bottle, cup, phone, remote, book, laptop, mouse, chair, bed, table,
  door, stairs, people known by face, and items you taught. Not keys or wallets.
- **Small items far away** are hard for YOLO; start within about 3 m.
- **Height is ignored.** "Straight ahead" says nothing about whether the bottle is on the floor or
  on a shelf.
- Guidance is an aid, not a safety guarantee. The tests on video are synthetic (a photo zoomed and
  panned). Real walking is tested with `docs/PHYSICAL_TESTS.md` section 4b.

## Demo script (about 60 seconds)

1. Bottle on a table, in view. "What do you see?" → it lists the bottle.
2. Turn away about 90°. "Find my bottle." → memory sentence with the turn instruction.
3. Turn slowly → "Keep turning…", then "Found your bottle…".
4. Walk up → "Straight ahead, close", then "within reach". Pick it up.
5. Mention: no internet, no new threads, hazards always win (point at the chair warning).
