# Text reading (OCR)

SENSE can read printed text that you hold up to the camera, like a medicine label, a sign or a page.
The code is in the `ocr/` folder and is **optional**. If it's missing or Tesseract isn't installed, SENSE runs exactly as before.

## Setup (once, on the Mac)

```bash
brew install tesseract        # the OCR program; English is included
tesseract --version           # should print 5.x
```

No pip package is needed. At start-up SENSE prints one of these lines:

```
📖 Text reading ready (Tesseract: /opt/homebrew/bin/tesseract). Say 'read this' or press R.
📖 Text reading off: Tesseract not found. Install it with: brew install tesseract
```

## Using it

| Say (or press) | What happens |
| --- | --- |
| "Read this" / "What does this say?" / "Read the label" / key **R** | SENSE says "Reading. Hold it steady." It waits up to 2.5 s for a steady, sharp frame and reads it once. Then it says "It says: ..." or "I can't find any readable text ..." |
| "Keep reading" / "Start reading" | Reading mode: SENSE reads every 2 s and announces only **new** text ("Text: WET FLOOR"). The same text isn't repeated for 30 s. Hazard warnings still come first |
| "Stop reading" | Ends reading mode. Saying "read this" while reading mode is on also ends it, and SENSE says so |

Tips: hold the text about 30 cm from the camera, flat, in good light, and keep it still for a second.

## How it works

```
sense_master.py main loop (thread 1)
   │ raw_frame (already captured: OCR never opens a camera)
   ▼
OCRController.feed()  ── OFF: returns at once (zero cost)
   │ WAITING: skip while the camera moves or the frame is blurry
   ▼
image_preprocessing.prepare()  gray → enlarge → CLAHE contrast → dark-text polarity
   ▼
TesseractEngine.start()  writes a temp PNG, starts `tesseract` with subprocess.Popen and returns at once
   ⋮  (tesseract runs as a separate PROCESS on another CPU core; the loop keeps going)
OCRController.poll()  each frame: proc.poll() → done? parse TSV words + confidences
   ▼
text_filter  drop low-confidence and junk words, skip repeats
   ▼
OCRMessage  → answer: speak() when SENSE is quiet
            → reading mode: notifier.push(P2), so safety (P1) always wins
```

**Why a process and not a thread?** SENSE may use at most two Python threads, and both are taken: the main loop and the voice listener. Calling Tesseract inside the loop, which is what `pytesseract` does, would freeze the camera and hazard warnings for 0.3 to 1.5 s. A separate process is the same trick SENSE already uses for the `say` voice. Tesseract is limited to one CPU core (`OMP_THREAD_LIMIT=1`) so YOLO stays fast.

Tesseract first tries layout mode 3 (automatic). Only if that finds nothing does it try mode 11 (sparse text), because in testing each mode missed a sign the other one read. All the numbers are in `ocr/config.py`: confidence 60, blur limit 40, steady wait 2.5 s, reading interval 2 s, repeat cooldown 30 s, and at most 40 spoken words.

## Tests

```bash
python -m unittest discover -s ocr/tests -t .     # OCR only (44 tests)
python tools/run_checks.py                        # everything, OCR included
```

These tests run **without** a camera. The real-Tesseract tests are skipped automatically if Tesseract isn't installed.

## Physical tests (only you can run these, on the Mac)

1. Start SENSE and check that the 📖 line says "ready".
2. Hold a medicine box 30 cm from the webcam, press space and say "read this". You should hear "Reading. Hold it steady" and then the label text within about 3 s. Note how long it took. The terminal prints `📖 OCR 0.xxs: N words -> '...'`.
3. Press **R** with the same box. You should get the same result without using voice.
4. Point the camera at a plain wall and press **R**. You should hear "I can't find any readable text ...", not made-up words.
5. Say "keep reading", hold up a sign, then put it down and hold it up again within 30 s. The sign should be read once. Say "stop reading" and check that it goes quiet.
6. While reading mode is on, walk towards a chair. The proximity warning must still be spoken, and must not be delayed by the text.
7. Ask "where is my phone?" and "what do you see?" while reading mode is on. Both must answer as usual.
8. Run once with Tesseract temporarily hidden (`brew unlink tesseract`, then `brew link tesseract` afterwards). SENSE must start normally and say "Text reading is not set up" when you press R.

## Known limits

- Printed text only. Handwriting, curved bottles and very small print are unreliable.
- English only by default. For other languages, install them (`brew install tesseract-lang`) and set `language` in `ocr/config.py`, for example `"eng+hin"`.
- It reads the whole frame, not just the object you're holding, so text in the background can be read too.
- In "quiet" alert mode, reading-mode announcements are not spoken, because they are priority P2. "Read this" answers are always spoken.
