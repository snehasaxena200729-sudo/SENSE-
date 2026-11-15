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
python sense_master.py                  # the camera you used last time (built-in Mac camera at first)
python sense_master.py --camera camo    # Android phone through Camo Studio (remembered)
python sense_master.py --camera builtin # back to the Mac camera
python sense_master.py --source clip.mp4   # replay a recorded video (repeatable tests)
python tools/find_cameras.py            # once: see which camera is which, pick with B / P
```

Press **space** to talk, **c** to switch between the Mac camera and Camo, and **q** to quit. The keys
work in the video window **and** in the terminal SENSE was started from (no need to click the window). SENSE beeps; start speaking after the beep. You have
about 5 seconds to start and 5 seconds to speak. The terminal prints how long each step took.

The first run needs internet once, to download the Whisper `tiny` model, the door/stairs model, the two face
models (YuNet 0.2 MB + SFace 37 MB, checksum-verified, into `models/`) (and `lap` if it's missing). After that SENSE works fully offline.

## What you can say

| Say | What SENSE does |
| --- | --- |
| "What do you see?" | Lists what is visible right now, with distance |
| "Where is my phone?" | Answers live if it's visible. Otherwise gives the last place it was seen, the direction from where you face *now* ("turn about 90 degrees to your right"), a nearby object ("next to the laptop"), and how sure it is |
| "Find my bottle" / "Help me find my phone" / "Take me to the door" | **Guides you to it** with short cues as you turn and walk: "a little to your left", "straight ahead, a few steps away", then "your bottle should be within reach". If it is not in view, it uses memory ("I last saw it 2 minutes ago, on your left, next to the laptop. Turn about 90 degrees to your left") and updates the turn as you turn. Hazard warnings always come first. See `docs/FIND_MODE.md` |
| "Stop looking" / "Cancel" / "I found it" | Ends the guidance |
| "I'm leaving" / "Remember the room" | Takes a **snapshot of the room**: every item it remembers and where it is ("Okay, I remembered 4 things: …") |
| "What changed since I left?" / "I'm back" / "What happened while I was away?" | **Compares the room with that snapshot** (or, if you didn't say "I'm leaving", with how the room was when SENSE last ran): "Since you left 12 minutes ago: your bottle was moved from in front of you to on your right. The white cup is no longer where it was. New: a laptop on your left. Sneha was here. Everything else is where it was." It only says an item is no longer there after it has really looked at its spot, and says so when it hasn't. See `docs/SCENE_CHANGES.md` |
| "What changed?" | Right after "I'm leaving" (or in the first 10 minutes after a restart): the snapshot comparison above. Otherwise: what moved, disappeared or appeared since you last asked (max 30 minutes back) |
| "What changed in the last 10 minutes?" | The same, for that time window ("last hour", "past five minutes", "half an hour" also work) |
| "Where was my bottle before?" | Uses the event history to say where it was before it moved |
| "When did you last see my keys?" | Gives the time of the last sighting |
| "Who was here today?" | Lists people recognised **by face** today |
| "What did you see?" | Lists things seen in the last 10 minutes that are not visible now |
| "This is my water bottle" | Teaches a personal item |
| "This is Sneha" | Learns Sneha's **face** (look at the camera for about 2 seconds). Everyone else stays "a person" |
| "Rename my bottle to my medicine" / "Change my bottle to medicine" | Renames a taught item or person; it keeps its look, place and history |
| "Forget my bottle" / "Forget Sneha" | Deletes the profile, memory and history (privacy) |
| "Be quiet" / "Normal mode" / "Talk more" | Changes the alert level (saved between runs) |
| "Speak slower" / "Speak faster" | Changes the speech rate (saved between runs) |

SENSE also speaks **on its own**:

- **Always:** a warning when you are walking into something ahead.
  - Doors, stairs, chairs, beds, tables, couches, people and the other walk-into classes keep **one identity** while
    you move (`continuity.py`), and you get **one** warning per approach, about 1.5 to 2 m before
    contact: "Careful, a chair in front of you, about 5 to 7 feet away". It is said again only after
    you have clearly moved away (beyond about 2.8 m) and come back. Standing still near something,
    turning your head past it, or someone reappearing closer does not trigger it.
  - A low object (a bed) that slides out of the bottom of the view while you walk towards it still
    gets a warning ("Careful, a bed in front of you, close ahead").
  - Other YOLO classes: a warning when the box grows quickly in the middle of the view.
  - A camera-only "looming" check (`hazards.py`) works even for things YOLO has no name for, such as
    walls or boxes, by measuring how fast the image expands (time to contact).
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
| `scene_diff.py` | **"What changed since I left?"** Compares a room snapshot with what was seen since: moved / no longer there / new / not checked yet. Estimates how far the camera was turned between runs by consensus voting, and tracks which directions were actually looked at |
| `hazards.py` | **Sudden-obstacle warnings:** time-to-contact from optical-flow divergence (any textured object) and box-growth checks for every YOLO class |
| `continuity.py` | **Object continuity + proximity.** Keeps one identity per object while you walk (camera-motion-compensated matching, ByteTrack id as a hint, no stale boxes), measures approach from the size trend along an axis that is not cut off, one alert per approach |
| `distance.py` | **Monocular distance with error ranges.** Focal length from resolution (or calibration), cut-off box detection, coarse spoken bands |
| `face_id.py` | **Face identity.** OpenCV YuNet finds the face, SFace turns it into 128 numbers; a name needs cosine similarity >= 0.363, a margin over everyone else, and a good frontal face. Enrolment collects 5 samples without blocking |
| `people.py` | **Person naming gate.** Detection (YOLO) and tracking (ByteTrack) never say who someone is; a name is spoken only after a verified identity check agrees 3 of the last 5 times |
| `memory_db.py` | SQLite tables: `environmental_memory` (latest state), `events` (history), `entities`, people and personal-item profiles, `settings`, `scene_snapshots` + `snapshot_items` (room snapshots) |
| `camera_source.py` | **Which camera.** Finds the built-in camera or Camo by name (OpenCV numbers cameras by sorted uniqueID, so index 0 is not "the Mac camera"), normalises frames for YOLO (BGR, at most 1280 wide), notices a frozen picture, reconnects after unplugging and falls back to the Mac camera |
| `voice_input.py` | **SPACE → beep → record → Whisper.** The listener thread; loads Whisper once, rejects Whisper's hallucinations (hint prompt echoed back, "no speech"), logs every press |
| `key_input.py` | Keys from the terminal as well as the video window, without a thread (cbreak + `select`) |
| `tests/` | Unit tests: `python -m unittest discover -s tests -v` |
| `tools/run_checks.py` | **All automated checks in one command**: unit + regression tests, and full-app scenarios on clips made by `tools/make_test_videos.py` (walk towards objects, stand still, turn the head, walk to a door, introduce someone / leave / come back / strangers) |
| `tools/smoke_run.py` | Runs the real app headless on a video and prints what it would have said (`--loop --stats-every 250 --json out.json` for long stress runs) |
| `tools/stress_voice.py` | 10,000 SPACE presses through the real listener with random errors, silences, Whisper echoes and restarts |
| `tools/camera_drill.py` | The real app with two fake live cameras: Camo 1080p hand-held, switching, unplugging, fallback |
| `tools/find_cameras.py` | Lists and shows every camera; saves which one is the Mac camera and which is Camo |
| `tools/stress_logic.py` | 10,000 randomised iterations of the continuity, naming, notifier and database logic with invariant checks |

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

- **Distances are coarse on purpose.** One camera cannot measure depth; SENSE infers it from an
  assumed size (eye spacing, body height, typical object size), so it speaks a range such as
  "about 4 to 6 feet away", "within arm's reach", "closer than about 7 feet" (when you are cut
  off by the frame edge) or "distance unclear". Calibrate your camera once and check the error
  with a tape measure: `python tools/validate_distance.py --live --person-height 1.70 --save`.
- **Doors are the weakest detection.** The main model (COCO) has no door class. The Open Images model
  does, but in synthetic tests a flush white door with a faint frame scored under 0.15, and a door cut
  off at the top scored near 0. A door handle inside a weak door box now counts as evidence. Measure
  your own door with `python tools/check_doors.py photos/ --models yolov8n-oiv7.pt yolov8s-oiv7.pt`;
  if the small model does better, start SENSE with `--scene-model yolov8s-oiv7.pt`.
- **People are "a person" unless their face is verified.** Earlier versions named people from the
  colour of their clothes, which matched different people (8 of 15 pairs in a test). Now only face
  recognition can name someone, and it needs a clear, roughly frontal face; turned away, in the dark
  or masked, SENSE says "a person". People taught with the old clothing method must be taught again.

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
- Proximity warnings rely on the detector seeing the object. Close up, detectors lose objects (the
  door model drops most doors within about 1.5 m), which is why the warning comes early. Objects YOLO
  does not know, glass, and blank walls rely on the looming check only.
