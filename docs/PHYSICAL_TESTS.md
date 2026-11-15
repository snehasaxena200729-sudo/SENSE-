# SENSE physical test guide

These tests need **you, a Mac, the webcam, the microphone and the speakers**. They cannot be run by
the automated checks, which use recorded or synthetic video and fake the microphone and `say`.
Write down what actually happens next to each expected result. A test passes only if what you hear
matches.

Before you start:

```bash
git pull
pip install -r requirements.txt          # opencv-python must be 4.8 or newer
python tools/run_checks.py               # automated part: must end with "0 failed"
```

Keep the terminal visible: SENSE prints what it decided (🧭 intent, 🔊 what it said, 🚪 door candidates).

---

## 1. Start-up and offline

| Step | Expected |
| --- | --- |
| 1.1 Run `python sense_master.py` with Wi-Fi **on** (first run only: models download) | Window opens. You hear "Sense Master Online…". If you taught people with the old clothing method you also hear "…please introduce them again by face". Terminal shows `🙂 Face recognition on` |
| 1.2 Quit (`q`). Turn Wi-Fi **off**. Start again | Same as 1.1, no download messages, no errors. Everything below must work with Wi-Fi off |
| 1.3 If the terminal says face recognition is **off** | It prints the reason (usually OpenCV older than 4.8). Run `pip install -U "opencv-python>=4.8"` and retry. Names cannot work until this line says "on" |

## 2. Microphone and voice

| Step | Expected |
| --- | --- |
| 2.1 Press **space**, wait for the beep, say "What do you see?" | Within about 3 s: "Currently, I see …" with directions and distance bands, or "I do not see any familiar objects or people right now." Terminal shows `🧭 Intent: OVERVIEW` |
| 2.2 Say "Speak slower", then "Speak faster" | The voice changes speed both times |
| 2.3 Press space and say nothing | No crash; SENSE carries on |

## 3. Names: "This is Sneha" end to end

Use a real person (Sneha) and a second person who has **not** been introduced.

| Step | Expected |
| --- | --- |
| 3.1 Sneha sits facing the camera, about 1 m away, good light. Press space, say "This is Sneha" | "Okay. Sneha, please look at the camera for a moment." Within about 6 s: "Got it. I will recognise Sneha by face." |
| 3.2 Stay in view, ask "What do you see?" | The answer contains "Sneha" |
| 3.3 Sneha leaves the view for 30 s, then comes back | Within a few seconds: "Sneha is here, …". If she comes back side-on or in poor light she may be "a person" until she faces the camera. That is intended |
| 3.4 Ask "Where is Sneha?" while she is out of view | "I last recognised Sneha … ago, … I can't see them now." |
| 3.5 The second person sits in front of the camera, Sneha not in view | Always "a person", **never** "Sneha" or any other name. If you hear a wrong name, note the light and angle. That is a real failure |
| 3.6 Quit, restart, Sneha sits down again | Recognised again after restart ("Sneha is here") |
| 3.7 Ask "Who was here today?" | Lists Sneha |
| 3.8 If enrolment fails ("I couldn't see Sneha's face clearly…") | Expected when the face is too small, too dark or turned. Move closer and retry. Note how many tries it took |

## 4. Walking towards things (proximity alerts)

Hold the laptop (or a webcam on a cable) at **chest height**, facing straight ahead. Walk at a normal,
slow indoor pace. Have a helper watch the path. **These alerts are an aid, never a safety guarantee.**

| Step | Expected |
| --- | --- |
| 4.1 Start about 4 m from a chair, walk straight at it, stop at arm's length | **Exactly one** "Careful, a chair in front of you, about … feet away" when you are about 1.5 to 2 m away. No repeat while you stand there |
| 4.2 Stand still 1 m from the chair for 30 s | Nothing more is said |
| 4.3 Walk back to about 3.5 m, then towards it again | **One** new alert on the second approach |
| 4.4 From 1.5 m, turn your head slowly left and right past the chair | **No** alert |
| 4.5 Walk past the chair with it about 1.2 m to your side | **No** alert |
| 4.6 Walk towards a bed from about 4 m | One alert. It may say "close ahead" instead of a distance when the bed has left the bottom of the view |
| 4.7 Walk towards a closed door from about 5 m | "Door in front of you" once when it is first seen, then **one** "Careful, a door in front of you, …" at about 1.5 to 2 m. Doors are the weakest detection (see README). Note the 🚪 candidate scores in the terminal if nothing is said |
| 4.8 While walking, ask "What do you see?" | Objects ahead are listed even while you move |

For each approach, note at what real distance the alert came (a helper with a tape measure, or floor
tape marks every 0.5 m).

## 4b. "Find my bottle" guidance

You need a bottle (or cup, phone, chair), a table and about 3 m of clear floor. Hold the camera at
chest height, as in section 4. A helper watches the path. Keep the terminal visible: every
guidance sentence is also printed as `🧭 Guide: …`, and the window shows `FINDING: …`.

| Step | Expected |
| --- | --- |
| G1 Bottle on a table about 2.5 m away, **in view**, a little to the left of centre. Hold still 3 s. Press space, say "Find my bottle" | Within about 3 s: "Found the bottle, a little to your left, …" (or "your bottle" if you taught it). Terminal: `🧭 Intent: FIND`, `🔎 Find mode: …` |
| G2 Turn slowly left | "Straight ahead, …" once it is centred. **Not** a cue every second: at least 1.5 s between cues, and the same cue repeats at most every 4 s |
| G3 Walk slowly towards it, keeping it centred | When you get closer the words change ("a few steps away" → "close"). Then **once**: "The bottle should be within reach, straight ahead. Reach out slowly." At that moment it should really be within about an arm's length (measure: note the distance). Guidance then stops: nothing more is said about it |
| G4 Look at the bottle (still, 3 s), then turn about 90° to the right so it is out of view. Ask "Find my bottle" | "Looking for the bottle. I can't see it right now. I last saw it … ago, on your left …. Turn about 90 degrees to your left, slowly." The angle is approximate (±30° is fine, it is estimated from camera motion). The side must be **left** |
| G5 Turn left slowly, keep turning | Every 6 s or so "Keep turning left." Once it is in view: "Found the bottle, …" and then the same as G2 and G3 |
| G6 Start guidance (G1), then cover the camera for 3 s | "I lost sight of the bottle. It was …". Uncover: "Found the bottle, …" again |
| G7 During guidance, walk towards a chair in your path (or have someone step in front) | The safety warning ("Careful, a chair …") is spoken and **is not cut off by a guidance cue**. Guidance continues about 2 s after the warning |
| G8a Ask "Find my keys" | Keys are not something SENSE can detect: "I can't look for that. I can find a bottle, cup, …" (no guidance starts) |
| G8b With no cup anywhere (and none seen this session), ask "Find the cup" | "Looking for the cup. I can't see it right now. Turn slowly in a circle …". "Keep turning slowly." about every 6 s. After about 45 s: "I couldn't find the cup. … Stopping the search." Never "Found". (If a cup was seen earlier, you get the memory sentence of G4 instead) |
| G9 Start guidance, then press space and say "Stop looking" | "Okay, I stopped looking for …". No more guidance cues |
| G10 Hide the bottle (move it to another room) where SENSE last saw it, then ask "Find my bottle" | It sends you to the old place, then says "The bottle should be in front of you, but I can't see it. It may have been moved." Never "Found" |
| G11 Ask "Take me to the door" with a door in view | "Found the door, …" and guidance. Close up (the door fills the view from side to side): "The door is right in front of you. Slow down." |

Notes: "within reach" depends on the distance estimate, which is coarse until you calibrate (section
5). If it comes too early or too late, write down the real distance. Things whose size SENSE does not
know (e.g. a person far away who is cut off by the frame) can be guided to but may never get "within
reach"; guidance then ends after 2 minutes.

## 4c. "What changed since I left?"

You need a table with 3 or 4 things SENSE knows (bottle, cup, phone, laptop, book, a chair nearby)
and the laptop on the table **not moving** (its camera pointing at the things). Keep the terminal
visible: the comparison is printed as `🔁 Scene diff vs snapshot …` with every bucket (moved / same /
gone / unchecked / appeared).

| Step | Expected |
| --- | --- |
| S1 Start SENSE, let it look at the table for 10 s. Press space, say "I'm leaving" | "Okay, I remembered N things: …" naming what is on the table (the number matches what is there). Terminal: `🧭 Intent: MARK` |
| S2 Without touching anything, wait 30 s, ask "What changed since I left?" | "Since you left 30 seconds ago: Nothing has changed that I can see." No item is called moved, missing or new |
| S3 Say "I'm leaving". Move the bottle about 40 cm sideways (to the other side of the table, still in view). Step out of view. After 30 s ask "What changed since I left?" | "… your bottle (or the bottle) was moved from … to …", with the sides matching what you did (left/right as seen from the laptop). "Everything else is where it was." |
| S4 Say "I'm leaving". Take the cup away (out of the room). Wait 30 s, ask "What changed?" (plain) | "… The cup is no longer where it was …". Also: a plain "What changed?" asked right after this uses the snapshot; asking "What changed?" again a minute later goes back to the normal "since you last asked" summary |
| S5 Say "I'm leaving". Put a new item (a book, a phone) on the table. Wait 30 s, ask "I'm back" | "… New: the book in front of you …" (side matches) |
| S6 Have someone SENSE knows by face (section 3) walk in front of the camera while you are "away" in S5 | The answer also says "… Sneha was here." Strangers are never named |
| S7 Say "I'm leaving". Turn the laptop about 90° away from the table and keep it there. Ask "What changed since I left?" | Nothing on the table is called missing: "… I haven't looked where the bottle, the cup and … were yet." Now turn it back to the table, wait 5 s, ask again: a normal answer as in S2 |
| S8 Restart test: quit SENSE (`q`) with the table in view. Move one item. Start SENSE again, let it look at the table for 10 s, ask "What changed since I left?" | Starts with "Since I last ran, …". With 3 or more unmoved items in view: the moved item is reported as moved. With fewer: "I can't tell if anything was moved, because the camera may have been turned since then" (honest, not a failure) |
| S9 Ask "What changed in the last 10 minutes?" | A summary of the events in that window (the old "what changed" wording: "N things changed: …") |
| S10 Fresh database test: rename `sense_memory.db` away, start SENSE, immediately ask "What changed since I left?" | "I don't have a picture of the room from before. Next time, say I'm leaving before you go …". Put the old database back afterwards |

Write down for S3: how far you moved the bottle and whether SENSE noticed. Very small moves (under
about 12° of the view, roughly 20 cm at 1 m) are deliberately ignored, and moves straight towards or
away from the camera are not detected at all.

## 5. Distance calibration (optional, recommended)

```bash
python tools/validate_distance.py --live --distances 1 1.5 2 3 --person-height <your height in m> --save
```

Expected: a table of true distance against estimate. Most rows should say "yes" under "in range?".
After `--save`, SENSE prints "📏 Distance: calibrated focal length." on the next start.

## 6. Long run

Leave SENSE running for 30 minutes in a room where people come and go.

| Check | Expected |
| --- | --- |
| Speed | The window stays smooth; the terminal timing lines do not grow over time |
| Memory | Activity Monitor: the Python process memory levels off and does not keep climbing |
| Repeats | No sentence is repeated over and over |
| Quit | `q` closes the window and the program ends within a few seconds. Run it again: no "database is locked" error |

---

### What to send back

For each section, write pass or fail, with the exact sentence SENSE said and the terminal lines
around any failure (for 4c, the `🔁 Scene diff` line). A short phone video of section 4 helps a lot.
