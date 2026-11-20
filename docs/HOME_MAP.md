# Home map (mapping/): Phases 0 and 1

SENSE is learning the rooms of your home. Phase 0 added four voice commands
and four database tables. Phase 1 teaches SENSE to recognise a room **by
sight**. Nothing SENSE already did has changed.

## Phase 1 in one minute

1. Stand in the middle of the room and say **"This room is the kitchen."**
   SENSE answers "Okay, the kitchen. Now turn slowly all the way round."
2. Turn slowly once, about 20 seconds for a full circle. SENSE says
   "Halfway there", then "Done. I've learned the kitchen from 17 views."
   If you don't start turning, it reminds you once. If it can't see enough,
   it asks you to try again. You can say "stop" to finish early (it keeps
   what it saw) or "cancel" (it saves nothing new).
3. Later, anywhere in the house, ask **"Where am I?"**
   - "You are in the kitchen." means it is sure: in at least 3 of the last 4
     checks (about 3 seconds) the room matched clearly.
   - "I think you are in the kitchen, but I'm not completely sure." means
     the match is likely but not clear yet.
   - "This doesn't look like any room you've taught me." means you are
     somewhere you haven't taught it.
   - "I'm not sure which room this is." is the honest answer when it can't
     tell. Hold the camera steady for a moment and ask again.
4. **"Relearn this room"** (or "relearn the kitchen") teaches it again, for
   example after you move furniture. The old views are replaced only when the
   new turn has finished, so a cancelled relearn loses nothing.

Rooms are remembered after a restart. Saying "this room is the kitchen" in a
room it already knows just tells it where you are.

### What is stored (no photos)

SENSE saves **numbers only**, never a picture of your home. For each of
about 18 directions it keeps:

- 1,280 numbers from a small image network (MobileNetV2), describing what the
  view looks like.
- Up to 400 corner patterns (32 bytes each) with their positions.
- The names of large things YOLO saw there (refrigerator, bed, ...).

That is about 20 KB per view and 350 KB per room, stored in `room_photos`
(the `image_path` column stays empty). A photo cannot be rebuilt from it.

### The image model (one download)

On the first run SENSE downloads
`image_classification_mobilenetv2_2022apr.onnx` (14 MB) from OpenCV's model
zoo into `models/`, like the face models, and checks its SHA-256. After that
it runs fully offline. It runs in OpenCV's DNN module, so there is no new pip
package. It works on OpenCV 4 and 5, which give identical numbers (tested on
4.10 and 5.1). Without the file, SENSE still works with corner features and
landmarks, but it is more often "not sure".

### How it decides (for the curious)

- **Image match (k-nearest-neighbours).** The live view is compared with every
  saved view by cosine similarity. With 2 or more rooms, the home's average
  view is subtracted first ("centering"). Every indoor picture shares walls,
  ceilings and light, and what is left is what makes one room different. On
  real indoor photos this moved the median similarity between different rooms
  from 0.59 to 0.11.
- **Corner check (ORB + RANSAC).** For the two most likely rooms, corners are
  matched and only matches that agree on one camera geometry (a fundamental
  matrix) count. Many of them mean the exact same spot.
- **Landmarks (Naive Bayes).** Seeing a refrigerator makes the rooms where you
  saw one while teaching more likely. This is clipped, so it can never decide
  on its own.
- **Bayes filter.** SENSE keeps a probability for every room, plus "a place I
  don't know". Each check multiplies in the new evidence. Between checks you
  most likely stay where you are. One check can move the odds only so far, so
  one odd frame never flips the answer.
- It checks on every 10th frame, only while the camera is steady. A check
  takes about 10 ms (measured), and a view saved while teaching costs about
  the same, at most 4 a second. No thread is added.

### Measured results

- **Real indoor photos** (`tools/mapping_eval.py`): 12 taught rooms (11 from
  ScanNet plus one from a real 360-degree panorama) and 4 untaught places.
  - Panorama room, views between the taught ones, also darker and blurred:
    27 of 27 right and sure.
  - ScanNet rooms seen from a very different viewpoint (a hard case: each was
    taught from one spot only): 8 of 22 right and sure, the other 14 "not sure".
  - Untaught places: 7 of 8 not named, 1 "I think" wrong.
  - **0 confident wrong answers** out of 57.
- **Real SENSE loop** (`tools/mapping_e2e.py`): teaching the kitchen and the
  bedroom by turning, then restarting SENSE. "Where am I?" was right in the
  kitchen (darker light, a view between the taught ones), right in the
  bedroom, and "doesn't look like any room you've taught me" in an untaught
  place. Three seconds after walking back into the kitchen it said "I think
  you are in the kitchen". It says "You are in" only after about 3 seconds of
  clear matches.

Test it at home: record a slow turn in each room, then a few other frames.
Put them in folders (`teach/kitchen/*.jpg`, `query/kitchen/*.jpg`, optionally
`query/_unknown/`) and run `python tools/mapping_eval.py teach query`.

## What you can say now

| Say | SENSE answers |
| --- | --- |
| "This room is the kitchen." | Saves the room and starts teaching it (Phase 1, above). In a room it already knows: "Okay, you are in the kitchen." |
| "Where am I?" | Phase 1: from what the camera sees (above). If no room has been learned by sight yet: "The last room you told me was the kitchen, 2 minutes ago." Only rooms named in this run and in the last 30 minutes count. If there is none: "I don't know which room you are in." |
| "Take me to the bedroom." | If it doesn't know the room: "I don't know the bedroom yet." If it does: "I know the bedroom, but I can't guide you between rooms yet." |
| "Which rooms do you know?" | "No rooms are registered yet." or "I know 2 rooms: the kitchen and the bedroom." |

Other wordings work too, for example "call this room the study", "which room
am I in", "how do I get to the bedroom" and "list the rooms". Whisper's
"bed room" is read as "bedroom".

**The word "room" is needed to teach a room.** "This is the kitchen" still
goes to the existing naming command, just like "this is Sneha" and "this is my
bottle". So say "this room is the kitchen".

"Take me to …" is a room command only when the place is a room word
(kitchen, bedroom, hall, pooja room, …), a room you registered, or
"<someone>'s room". "Take me to my phone", "take me to the door" and "take
me to Sneha" go to the existing find command, as before. A person or object
SENSE knows by a name always wins over a room with that name.

## The four tables (added to sense_memory.db)

| Table | One row per | Filled from |
| --- | --- | --- |
| `rooms` | room you named | Phase 0 ("this room is …") |
| `room_photos` | saved view of a room (direction, numbers only, landmarks) | Phase 1 |
| `door_links` | pair of rooms joined by a door, with the door's direction in each | Phase 3 |
| `object_rooms` | object, with the room it was last seen in | Phase 5 |

Safety rules:

- The tables are created with `CREATE TABLE IF NOT EXISTS`. Nothing existing
  is dropped, renamed or altered, and no existing table is read or written.
  It is safe to run any number of times.
- If a table with one of these names already exists with different columns,
  SENSE prints `🏠 Home map off (SchemaConflict ...)`, leaves that table alone
  and runs without room commands.
- `room_photos`, `door_links` and `object_rooms` declare foreign keys to
  `rooms(id)`. The database connection is shared with the rest of SENSE, so
  SQLite's foreign-key switch is left as it was (off). When a room is deleted,
  the package removes its child rows itself.
- `object_rooms.object_name` uses the same names as `environmental_memory`,
  but it is not linked to that table, so nothing in your existing memory can be
  blocked or deleted because of it.

## How it connects to sense_master.py

There are three small additions and no other changes:

1. Startup, after the interactive-questions block: `home_map = mapping.create(SESSION_ID)`,
   inside try/except. If it fails, SENSE prints "🏠 Home map off" and runs as
   before.
2. In `answer_voice_query`, after the OCR and interactive checks and just
   before `answers.classify_intent`, `home_map.handle(...)` is called. If it
   returns a sentence, SENSE speaks it and stops there. If it returns `None`
   (any sentence that is not a room command), the existing code runs exactly
   as before. Phase 1 also passes the current heading here, so a new room's
   views start from the direction you face.
3. Phase 1: at the end of each frame, after the existing whereabouts and
   personal-item updates, `home_map.observe(raw_frame, boxes, model.names,
   heading, camera_is_stable, ...)` is called. It returns a teaching message
   ("Done. I've learned the kitchen") only when SENSE is neither listening
   nor already talking, and SENSE speaks it.

The home map adds no threads and opens no camera. Its only network call is
the one-time model download.

## Tests

    python -m unittest discover -s mapping/tests -t .     # 46 tests
    python tools/stress_mapping.py                         # 10,000 cases
    python tools/mapping_eval.py TEACH_DIR QUERY_DIR       # your own rooms
    python tools/mapping_e2e.py --pano P.jpg --strip A.jpg B.jpg --unknown U.jpg   # real loop

The stress test mixes three kinds of case. Room commands in noisy Whisper
wordings must be recognised. Existing commands ("this is Sneha", "take me to
my phone", "what's in this room", …) must be left alone. Random conversations
are checked against a reference model. Since Phase 1 it also covers random
head-turns while teaching (any speed, either direction, pauses, blur, stop),
and random walks through homes of 2 to 8 rooms. Once the user has arrived in
a room, "You are in X" must never name the wrong room.

## Next phases

1. ~~Teach and recognise rooms by sight.~~ Done.
2. Optional printed door markers.
3. Learn how rooms connect.
4. "Take me to the bedroom" guidance.
5. "Your phone is in the bedroom".

See the roadmap in the project thread.
