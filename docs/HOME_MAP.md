# Home map (mapping/): Phase 0

SENSE is learning the rooms of your home. Phase 0 is the groundwork. It adds
four voice commands and four database tables. Nothing SENSE already did has
changed.

## What you can say now

| Say | SENSE answers |
| --- | --- |
| "This room is the kitchen." | "Okay, I saved the kitchen as a room." (Says it again later: "Okay, you are in the kitchen.") |
| "Where am I?" | "The last room you told me was the kitchen, 2 minutes ago. I can't recognise rooms by sight yet." Only rooms named in this run and in the last 30 minutes count. If there is none: "I don't know which room you are in." |
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
| `room_photos` | saved view of a room (direction, image fingerprint, landmarks) | Phase 1 |
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

There are two small additions and no other changes:

1. Startup, after the interactive-questions block: `home_map = mapping.create(SESSION_ID)`,
   inside try/except. If it fails, SENSE prints "🏠 Home map off" and runs as
   before.
2. In `answer_voice_query`, after the OCR and interactive checks and just
   before `answers.classify_intent`, `home_map.handle(...)` is called. If it
   returns a sentence, SENSE speaks it and stops there. If it returns `None`
   (any sentence that is not a room command), the existing code runs exactly
   as before.

The home map adds no threads, opens no camera, loads no model and makes no
network calls.

## Tests

    python -m unittest discover -s mapping/tests -t .     # 20 tests
    python tools/stress_mapping.py                         # 10,000 cases

The stress test mixes three kinds of case. Room commands in noisy Whisper
wordings must be recognised. Existing commands ("this is Sneha", "take me to
my phone", "what's in this room", …) must be left alone. Random conversations
are checked against a reference model.

## Next phases

1. Teach and recognise rooms by sight.
2. Optional printed door markers.
3. Learn how rooms connect.
4. "Take me to the bedroom" guidance.
5. "Your phone is in the bedroom".

See the roadmap in the project thread.
