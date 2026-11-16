# SENSE demo checklist (MacBook, built-in camera)

Run this top to bottom the night before, and steps 1 to 4 again 15 minutes before the demo.
Tick each box. If a step fails, go to **"If something fails"** at the bottom.

You need: a plastic bottle, a chair, a door you can walk towards (a door with a visible
frame and handle works best), a printed page with large text, a medicine box, and a helper
("Sneha") for the naming demo.

---

## 1. Code and environment (5 min, Wi-Fi ON)

```bash
cd ~/path/to/SENSE-
git status                      # must say "nothing to commit" (or only sense_memory.db)
git fetch origin
git checkout main && git pull   # or the branch you are demoing
git log -1 --oneline            # write down this commit id
python3 -c "import ultralytics, cv2, speech_recognition, whisper, lap; print('python packages OK')"
tesseract --version | head -1   # 5.x  (if missing: brew install tesseract)
python3 tools/run_checks.py     # about 15 s; must end with "0 failed."
```

- [ ] `0 failed.` from `run_checks.py`
- [ ] All offline models are on disk (each line must print a file, not "No such file"):

```bash
ls -la yolov8s.pt yolov8n-oiv7.pt models/*.onnx ~/.cache/whisper/tiny.pt
```

If one is missing, start SENSE once **with Wi-Fi on** (step 3) and it downloads itself.

## 2. Camera and microphone (2 min)

- [ ] Close Zoom / FaceTime / Photo Booth (they hold the camera).
- [ ] `python3 hello_camera.py` shows your built-in camera picture. Press **q**.
- [ ] Microphone + Whisper, offline engine the app uses:

```bash
python3 -c "
import speech_recognition as sr; r=sr.Recognizer()
with sr.Microphone() as m: print('SAY: where is my bottle'); a=r.listen(m, timeout=5, phrase_time_limit=5)
print('heard:', r.recognize_whisper(a, model='tiny'))"
```

- [ ] It prints roughly "Where is my bottle?". If macOS asks for microphone permission for Terminal/VS Code, allow it.
- [ ] Volume up; `say "test"` is audible.

## 3. Start SENSE (Wi-Fi OFF from here on)

- [ ] Turn **Wi-Fi off** (this is your offline proof). Optional: show the Wi-Fi icon to the audience.
- [ ] `python3 sense_master.py`
- [ ] The terminal shows all of these lines:
  - `🚪 Door/stairs detection on: ['door', 'door handle', 'stairs']`
  - `🙂 Face recognition on. People enrolled by face: ...`
  - `📖 Text reading ready (Tesseract: ...)`
- [ ] You hear: **"Sense Master Online. Restored N objects and M people."**
  If it adds "... taught with the old clothing method", those names can't be recognised. Before the
  demo say **"Forget Sneha"** (for each such name; this also deletes a face profile of that name)
  and teach Sneha again in step B.
- [ ] Click once on the **video window**. Keys (space, R, q) only work while that window is in front.
- [ ] Reset the settings a previous run may have saved: press space, say **"Normal mode"**
  → "Normal mode. I will tell you about your items and people you know."

### How to give every voice command

1. Video window in front, press **space**. You hear a short "tink".
2. **Wait until the terminal prints `🎙️ Listening now...`** (about 1 second after the tink).
   Words spoken before that line are lost (this is how "This is Sneha" turns into just "Sneha").
3. Speak the command in one breath, then stay silent. The answer comes 1 to 3 s later.
4. The terminal prints `👂 Heard: '...'` and `🧭 Intent: ...`. If the words are wrong, just repeat.

## 4. Demo sequence (exact words and expected answers)

Keep the laptop still on the table unless a step says otherwise.
SENSE only updates its memory while the camera is still ("CAM MOVING" in red = it is waiting).

### A. What is around me (30 s)
- [ ] Bottle on the table about 1 m from the camera, upright, in good light. Wait 5 s.
- [ ] Say **"What do you see?"** → "Currently, I see the white bottle on your left, about 2 to 4 feet away ..."
  (the colour and side depend on your bottle; distances are always a range, never one exact number)

### B. Person naming (1 min)
- [ ] Sneha sits **alone** in front of the camera, face visible, about 1 m away, good light.
- [ ] Say **"This is Sneha"** → "Okay. Sneha, please look at the camera for a moment."
      Sneha looks at the camera for 2 seconds → "Got it. I will recognise Sneha by face."
- [ ] Sneha leaves the view for 10 s, then comes back → "Sneha is here, in front of you."
- [ ] Say **"What do you see?"** → "Currently, I see Sneha in front of you, about ... feet away."
- [ ] Another person (or Sneha facing away) is only ever "a person", never "Sneha".
- [ ] Say **"Who was here today?"** → "Today I saw Sneha, who is here right now."

### C. Find mode (1 min)
- [ ] Put the bottle to one side of the table. Say **"Find my bottle"**.
  - visible → "Found the bottle, a little to your left ..." or "The bottle is right in front of you ..."
  - not visible → "Looking for the bottle. I can't see it right now. I last saw it ... Turn about N degrees to your left, slowly."
- [ ] Turn the laptop slowly towards it → "Found the bottle ...", then "Straight ahead ...".
- [ ] Say **"Stop looking"** → "Okay, I stopped looking for the bottle."

### D. Door and obstacle (2 min, carry the laptop open, camera forward, walk slowly)
- [ ] Stand about 4 m from a chair. Walk slowly towards it.
  → once: "Careful, a chair in front of you, about 5 to 7 feet away." (said once per approach, not repeated)
- [ ] Stand about 4 to 5 m from the door, door fully in view. → "Door in front of you."
  Walk slowly towards it → once: "Careful, a door in front of you, about 5 to 7 feet away."
- [ ] Say **"Where is the door?"** → "There is a door in front of you."
- [ ] Say what the range means: distances come from the object's typical size, so they are
  estimates with a range, not measurements.

### E. Object moved / disappeared, "What changed?" (2 min, laptop still)
- [ ] Bottle on the **left** side of the camera view. Wait 10 s. Say **"What changed?"** (this only sets the starting point).
- [ ] Take the bottle away while the camera watches. Wait **10 s** (SENSE waits 8 s so a missed frame is not "gone").
- [ ] Say **"What changed?"** → "One thing changed: the white bottle disappeared."
- [ ] Say **"Where is the bottle?"** → "I last saw the white bottle on your left ... but it disappeared while I was watching, so it may have been moved."
- [ ] Put the bottle back on the **right** side of the view. Wait 5 s.
- [ ] Say **"What changed?"** → "One thing changed: the white bottle was moved, now on your right."
- [ ] Say **"Where was the bottle before?"** → "Before it was moved ... the white bottle was on your left. Now it is on your right."
- [ ] Say **"When did you last see the bottle?"** → "Right now, I see the white bottle on your right ..."
- [ ] Optional, camera turn ≠ disappearance: turn the laptop about 90° away for 10 s and back.
  "What changed?" must **not** say the bottle disappeared.

### F. Reading text (2 min)
- [ ] Hold the printed page **20 to 30 cm** from the camera, flat, text filling about half the picture, good light.
- [ ] Say **"Read this"** (or press **R**) → "Reading. Hold it steady." then "It says: ...".
- [ ] Medicine box: front of the box (big product name) first. Say **"Read this"**.
  If SENSE adds "Some words were unclear, so please check it, especially any numbers." that is the
  intended honest behaviour: point it out, never present a dose as certain.
- [ ] Use the words "Read this" exactly. "Read the medicine label" is **not** recognised as a reading command.

### G. Voice settings (30 s)
- [ ] **"Be quiet"** → "Quiet mode. I will only warn you about obstacles."
- [ ] **"Talk more"** → "Okay, I will tell you about every change I notice."
- [ ] **"Speak slower"** → "Okay, I will speak slower." then **"Speak faster"** → "Okay, I will speak faster."
- [ ] Finish with **"Normal mode"**. (These settings are saved for the next run.)

### H. Offline proof
- [ ] Wi-Fi has been off since step 3 and every step above worked.
- [ ] Press **q** to quit.

## If something fails during the demo

| Symptom | Do this |
| --- | --- |
| Space does nothing | Click on the video window, press space again |
| "I didn't hear anything" / only "Sneha" was heard | Press space, wait for `🎙️ Listening now...`, then speak |
| "I don't see anyone clearly" on "This is Sneha" | Sneha faces the camera, closer (about 1 m), more light; keep the laptop still for 1 s; say it again |
| "I couldn't see Sneha's face clearly" | Better light on the face, only one person in view, try again |
| A wrong name is spoken | Say "forget Sneha", then teach again with only Sneha in view |
| Bottle not mentioned | Bottle upright, 0.5 to 1.5 m away, nothing in front of it; wait 5 s with the laptop still |
| "What changed?" says nothing changed | The laptop moved (memory pauses while the camera moves) or you asked within 8 s; wait 10 s and ask again |
| Door not announced | Step back until the whole door (top to bottom) is in view; doors with a visible frame and handle work best; plain flush white doors are the known weak case |
| Text not read | Closer (20 cm), steady for 2 s, more light; press R |
| Voice works but no sound | Check the volume; run `say test` in another terminal |
| SENSE crashes or freezes | Press q (or Ctrl+C in the terminal), run `python3 sense_master.py` again. Memory is kept in `sense_memory.db` |
| Camera not found at start | Quit other camera apps, run again |
