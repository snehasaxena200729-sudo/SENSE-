# SENSE: Context-Aware Environmental Memory for the Visually Impaired

SENSE is an offline assistive companion. A camera watches the user's surroundings, and SENSE
**remembers** what it saw, **notices** what changed, and **answers questions** about it by voice.
It says how sure it is, and it only speaks up when something matters.

Everything runs locally: YOLOv8 + ByteTrack, OpenCV, SQLite, Whisper and macOS `say`. The app uses
at most **two Python threads**: the main loop, plus a short-lived voice listener.

## Run it

```bash
brew install portaudio            # macOS, needed by PyAudio
pip install -r requirements.txt
python sense_master.py            # webcam 0
python sense_master.py --source clip.mp4   # replay a recorded video (repeatable tests)
```

Press **space** to talk and **q** to quit. SENSE beeps; start speaking after the beep. You have
about 5 seconds to start and 5 seconds to speak. The terminal prints how long each step took.

The first run needs internet once, to download the Whisper `tiny` model, the door/stairs model (and `lap` if it's
missing). After that SENSE works fully offline.

## What you can say

| Say | What SENSE does |
| --- | --- |
| "What do you see?" | Lists what is visible right now, with distance |
| "Where is my phone?" | Answers live if it's visible. Otherwise gives the last place it was seen, the direction from where you face *now* ("turn about 90 degrees to your right"), a nearby object ("next to the laptop"), and how sure it is |
| "What changed?" | Summarises what moved, disappeared or appeared since you last asked (max 30 minutes back) |
| "Where was my bottle before?" | Uses the event history to say where it was before it moved |
| "When did you last see my keys?" | Gives the time of the last sighting |
| "Who was here today?" | Lists people verified by face recognition today (until face recognition is set up, it says it cannot tell) |
| "What did you see?" | Lists things seen in the last 10 minutes that are not visible now |
| "This is my water bottle" | Teaches a personal item |
| "This is Sneha" | Needs face recognition. Until it is set up, SENSE refuses and calls everyone "a person" |
| "Rename my bottle to my medicine" / "Change my bottle to medicine" | Renames a taught item or person; it keeps its look, place and history |
| "Forget my bottle" / "Forget Sneha" | Deletes the profile, memory and history (privacy) |
| "Be quiet" / "Normal mode" / "Talk more" | Changes the alert level (saved between runs) |
| "Speak slower" / "Speak faster" | Changes the speech rate (saved between runs) |

SENSE also speaks **on its own**:

- **Always:** a warning when something is ahead and getting closer fast. Two checks run on every
  frame: any object YOLO knows growing quickly in the middle of the view, and a camera-only
  "looming" check (`hazards.py`) that works even for things YOLO has no name for, such as walls,
  doors or boxes, by measuring how fast the image expands (time to contact).
- **Normal mode:** your named items moving or disappearing, and known people arriving or leaving.
- **"Talk more" mode:** every change it notices.

## Architecture

| File | Role |
| --- | --- |
| `sense_master.py` | Main loop: camera → optical-flow heading → YOLOv8 + ByteTrack → identity → memory, events and alerts → HUD. Answers voice questions on the main thread |
| `identity.py` | **Stable object identity.** Colour is voted over time and the name is frozen, so one physical object = one memory. Re-identifies objects by appearance and position |
| `events.py` | **Event state machine:** APPEARED, MOVED, GONE, OUT_OF_VIEW, RETURNED. Debounced, and it tells "gone" apart from "you turned away" |
| `notifier.py` | **What to say, and when:** priorities, cooldowns, no talking over itself, quiet/normal/verbose |
| `spatial.py` | **World-anchored directions** (stored angle + heading), turn instructions, nearby-object relations |
| `answers.py` | **Intents, fuzzy name matching, confidence scoring** (confirmed / likely / uncertain) and wording |
| `hazards.py` | **Sudden-obstacle warnings:** time-to-contact from optical-flow divergence (any textured object) and box-growth checks for every YOLO class |
| `people.py` | **Person naming gate.** Detection (YOLO) and tracking (ByteTrack) never say who someone is; a name is spoken only after a verified identity check agrees 3 of the last 5 times |
| `memory_db.py` | SQLite tables: `environmental_memory` (latest state), `events` (history), `entities`, people and personal-item profiles, `settings` |
| `tests/` | Unit tests: `python -m unittest discover -s tests -v` |

The other `.py` files in the root are earlier learning prototypes and are not used by the app.
Note that `voice_sense.py` and `test_mic.py` use Google's *online* speech API.

### Threads

| Thread | Purpose |
| --- | --- |
| Main thread | Camera loop, all memory, database writes, answers and alerts |
| Voice listener | Started by the space key; records and transcribes only, then hands the text over through a `queue.Queue` |

Speech output is a separate `say` *process*, not a thread. Threads that libraries create
internally (PyTorch, OpenCV, PortAudio) are not managed by the app.

## Known limitations

- **People are "a person" unless verified.** Earlier versions named people from the colour of
  their clothes, which matched different people (8 of 15 pairs in a test). That is removed.

- The heading comes from optical flow. It drifts over time and resets every run, so turn
  directions are only given for things seen in the current run.
- Distance is a rough single-camera estimate.
- Two identical objects (same class and colour) that are never seen together may be treated as one.
- Doors and stairs come from a second small model (`yolov8n-oiv7.pt`, Open Images, about 7 MB),
  downloaded once on the first run. It runs every 5th frame to keep SENSE fast, so a door warning can
  come up to a few tenths of a second later than other warnings. Without the model SENSE still runs,
  just without door/stairs warnings.
- Obstacle warnings need some texture or a known object: blank walls, glass and stairs going down
  can be missed, and sudden lighting changes can cause a false warning. They are an aid, not a
  safety guarantee.
- People are recognised by clothing colour, which is not reliable across days.
