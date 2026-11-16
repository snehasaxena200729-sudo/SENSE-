# Text reading (OCR)

SENSE can read printed text that you hold up to the camera, like a medicine label, a sign or a page.
The code is in the `ocr/` folder and is **optional**. If it's missing or Tesseract isn't installed, SENSE runs exactly as before.

## Setup (once, on the Mac)

```bash
brew install tesseract        # the OCR program; English is included
tesseract --version           # should print 5.x
```

The text detector model (EAST, 96 MB) downloads by itself into `models/` the first time SENSE starts with internet, with a checksum check, the same way the face models do. After that everything is offline. Without it, SENSE prints `📖 Text detector: classic (EAST model missing: weaker fallback)` and still works, with a simpler detector.

No pip package is needed. At start-up SENSE prints one of these lines:

```
📖 Text reading ready (Tesseract: /opt/homebrew/bin/tesseract). Say 'read this' or press R.
📖 Text reading off: Tesseract not found. Install it with: brew install tesseract
```

## Using it

| Say (or press) | What happens |
| --- | --- |
| "Read this" / "What does this say?" / "Read the label" / key **R** | SENSE says "Reading. Hold it steady." It finds the text blocks in the picture (yellow boxes on screen for 6 s), picks the one you're holding up (green box), and reads only that. Then it says "It says: ...", or reads it with a warning ("I'm not sure about 500, so please check it"), or says what to change ("Move it closer to the camera") |
| "Keep reading" / "Start reading" | Reading mode: SENSE reads every 2 s and announces only **new** text ("Text: WET FLOOR"). The same text isn't repeated for 30 s. Hazard warnings still come first |
| "Stop reading" | Ends reading mode. Saying "read this" while reading mode is on also ends it, and SENSE says so |

Tips: hold the text about 30 cm from the camera, flat, in good light, and keep it still for a second.

**Unclear readings are flagged.** When Tesseract was unsure of a word it would speak (confidence below 85),
or dropped a real-looking word, SENSE adds: "Some words were unclear, so please check it, especially any
numbers." In testing, misread doses ("more than 6" for "8") scored 60 to 80 and would otherwise have been
spoken as fact. A clean reading is not a guarantee either: always check medicine names and doses.

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
python -m unittest discover -s ocr/tests -t .     # OCR only (83 tests)
python tools/make_ocr_testset.py && python tools/ocr_eval.py   # accuracy on test images, old vs new
python tools/run_checks.py                        # everything, OCR included
```

These tests run **without** a camera. The real-Tesseract tests are skipped automatically if Tesseract isn't installed, and the real-detector ones if the EAST model isn't downloaded.

One older test, `test_reads_small_text`, fails in the cloud with Tesseract 5.3.4 on main too: it checks the old whole-frame reading, which misses "EXIT" there. The new "read this" path reads the same image correctly ("EXIT. Room 204").

## Physical tests (only you can run these, on the Mac)

1. Start SENSE and check that the 📖 line says "ready".
2. Hold a medicine box 30 cm from the webcam, press space and say "read this". You should hear "Reading. Hold it steady" and then the label text within about 3 s. Note how long it took. The terminal prints `📖 OCR 0.xxs: N words -> '...'`.
3. Press **R** with the same box. You should get the same result without using voice.
4. Point the camera at a plain wall and press **R**. You should hear "I can't find any readable text ...", not made-up words.
5. Say "keep reading", hold up a sign, then put it down and hold it up again within 30 s. The sign should be read once. Say "stop reading" and check that it goes quiet.
6. While reading mode is on, walk towards a chair. The proximity warning must still be spoken, and must not be delayed by the text.
7. Ask "where is my phone?" and "what do you see?" while reading mode is on. Both must answer as usual.
8. Run once with Tesseract temporarily hidden (`brew unlink tesseract`, then `brew link tesseract` afterwards). SENSE must start normally and say "Text reading is not set up" when you press R.

## Finding and choosing the text (upgrade, 16 Nov 2026)

### Why the first version failed

1. **No box around an unfamiliar object.** OCR drew nothing, and YOLO boxes appear only for the few COCO classes SENSE trusts, after a confidence average and a 4-frame vote. A medicine strip, a packet or a sign is not a COCO class, so it never got a box. Those YOLO gates stay as they are (they stop false "phone" alerts); text now gets its own detector.
2. **"Read this" read the wrong text.** The whole 1280 x 720 frame went to Tesseract, so the label was mixed with posters and screens, then cut at 40 words. Letters of a label at arm's length are 10 to 18 px tall and were never enlarged (Tesseract likes about 30 px). The steadiness check looked at the background, not at the label.

### How it works now ("read this" / R)

```
frame (no camera of its own)
  -> text_detector.py   EAST finds word boxes, any object, any class (classic fallback without the model)
  -> text_regions.py    words -> lines -> blocks (one label, one sign); a block on a YOLO object gets its name
  -> target_selector.py score each block: centre 0.35, letter size 0.30, sharpness 0.15, on an object 0.20
                        blocks on one object, or close together, are one target
                        best must beat the runner-up clearly, else: "hold the one you want in the middle"
  -> rectify.py         crop with margin; deskew by the EAST angle; perspective warp if a clear 4-corner outline
  -> image_preprocessing.py  scale letters to ~34 px, then up to 3 recipes chosen from brightness/grain/shadow:
                        clahe | sharpen (unsharp mask) | otsu | adaptive | denoise (non-local means)
  -> Tesseract (own process, page mode 7 for one line, 6 for a few lines, 3 for a page)
  -> quality.py         confidence + coverage (share of detected words actually read) + cut-at-edge
                        -> clear / unclear / partial / unreadable; stops early on a good reading;
                        otherwise votes word by word across the recipes
  -> answer             "It says: ...", "I'm not sure about 500, so please check it",
                        "I can only read part of it: ... <tip>", or "I can see some text but I can't read it. <tip>"
  -> camera_guidance.py one tip: too dark, glare, blurry, too small, cut off, tilted, too close, label turned away
```

SENSE never asks "left or right?". When two texts are equally likely, it says how to frame the one you want and reads nothing, so it never guesses.

**Speed.** Tesseract still runs as a separate process, so it never freezes the loop. The EAST detector runs inside the main loop, normally once per "read this" (again every 0.8 s only while the picture is blurry, moving or ambiguous). In the cloud test machine (4 CPU cores) one detection took about 0.47 s at the 960 x 544 input size, 0.2 s at 640 x 352. Apple Silicon should be several times faster, but please check the camera window does not stutter for long when you press R. If it does, set `east_input_size = (640, 352)` in `ocr/config.py`: faster, a little less accurate (see results).

**Switches** in `ocr/config.py`: `use_text_detection = False` brings back the old whole-frame reading; `use_east = False` uses the classic detector only; `use_perspective`, `max_attempts`, the score weights and every guidance threshold are there too.

"Keep reading" (reading mode) still reads the whole frame every 2 s, as before. Only "read this" uses the new path.

### Test images and results

`python tools/make_ocr_testset.py` builds `ocr_testset/` (git-ignored): **8 public photos** (signs, a poster, book pages, a document, and 2 frames from the ICDAR 2015 head-mounted-camera set) and **19 rendered labels** pasted into real photos and degraded like a webcam (small, tilted, leaning, blurred, dark, grainy, JPEG). Rendered images have exact ground truth but are **not real camera captures**. `python tools/ocr_eval.py` runs every image through the real controller, new and old pipeline, and prints the table below.

Measured 16 Nov 2026 in the cloud (Linux, 4 CPU cores, Tesseract 5.3.4, OpenCV 4.14, EAST at 960 x 544). Not yet measured on the Mac.

| Measure | New | Old (whole frame) |
| --- | --- | --- |
| Text found by the detector (24 images with text) | 22 of 24 | no detector |
| Right target chosen (23 images with a known target or no text) | 19 of 23 | no choice made |
| Expected words spoken, average (24 images) | 0.80 | 0.68 |
| Spoken words that are right (images where something was read) | 0.93 (n=20) | 0.80 (n=19) |
| Tilted labels and pages (3 images): words spoken | 1.00, 1.00, 1.00 | 0.30, 0.20, 0.00 |
| Held packet with a poster behind it | read only the packet | read the poster too |
| Two equal signs side by side | framing tip, nothing read | read both as one text |
| Plain "It says" with fewer than 80% of the words right | 0 of 27 | 0 of 27 |
| Slowest single camera frame during a read | 0.85 s (the detection frame) | 0.2 s |

The last "honest" row looks the same for both because the old code also added "some words were unclear" to bad readings. The difference is in what was read: the old code mixed background text into the answer, and read tilted text as garbage ("1005 D>" for "VITAMIN D3 1000 IU").

Full table (`Words read` = share of expected words spoken; `Extra words` = share of spoken words not in the expected text; `Honest` = no plain "It says" for a reading under 80% right):

| Set | Image | Case | Text found | Chosen box inside target | Right target | Words read (new) | Words read (old) | Extra words (new) | Extra words (old) | Honest (new) | Honest (old) | Time |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| real | who_poster.png | document | yes | - | - | 1.00 | 1.00 | 0.04 | 0.04 | yes | yes | 1.3s |
| real | street_signs.png | multiple regions | yes | - | yes | 0.00 | 1.00 | - | 0.60 | yes | yes | 3.5s |
| real | book_page.png | book | yes | - | - | 1.00 | 1.00 | 0.05 | 0.05 | yes | yes | 3.4s |
| real | book_page_tilted.png | tilted | yes | - | - | 1.00 | 0.30 | 0.05 | 0.62 | yes | yes | 2.7s |
| real | video_sign.png | signboard | yes | 0.34 | NO | 1.00 | 1.00 | 0.00 | 0.50 | yes | yes | 0.9s |
| real | eurotext.png | document | yes | - | - | 1.00 | 1.00 | 0.47 | 0.47 | yes | yes | 1.0s |
| real | icdar_room_sign.jpg | room label | yes | 1.00 | yes | 0.50 | 0.00 | 0.50 | - | yes | yes | 0.9s |
| real | icdar_far_exit.jpg | small text | yes | 0.03 | NO | 0.00 | 0.00 | - | - | yes | yes | 1.0s |
| rendered | med_bottle.jpg | medicine bottle | yes | 1.00 | yes | 0.89 | 0.89 | 0.11 | 0.11 | yes | yes | 0.9s |
| rendered | med_bottle_small.jpg | medicine bottle | yes | 1.00 | yes | 0.80 | 0.60 | 0.20 | 0.33 | yes | yes | 0.8s |
| rendered | med_packet.jpg | medicine packet | yes | 1.00 | yes | 1.00 | 1.00 | 0.00 | 0.00 | yes | yes | 0.7s |
| rendered | med_packet_angle.jpg | medicine packet | yes | 1.00 | yes | 1.00 | 0.78 | 0.00 | 0.00 | yes | yes | 0.7s |
| rendered | document.jpg | document | yes | 1.00 | yes | 1.00 | 1.00 | 0.00 | 0.00 | yes | yes | 0.8s |
| rendered | book.jpg | book | yes | 1.00 | yes | 1.00 | 1.00 | 0.00 | 0.00 | yes | yes | 0.9s |
| rendered | room_label.jpg | room label | yes | 1.00 | yes | 1.00 | 1.00 | 0.00 | 0.00 | yes | yes | 0.7s |
| rendered | signboard.jpg | signboard | yes | 0.66 | yes | 1.00 | 1.00 | 0.00 | 0.00 | yes | yes | 1.0s |
| rendered | small_text.jpg | small text | yes | 1.00 | yes | 1.00 | 0.67 | 0.00 | 0.20 | yes | yes | 0.7s |
| rendered | tilted_22.jpg | tilted | yes | 1.00 | yes | 1.00 | 0.20 | 0.00 | 0.50 | yes | yes | 0.7s |
| rendered | tilted_35.jpg | tilted | yes | 1.00 | yes | 1.00 | 0.00 | 0.00 | - | yes | yes | 0.7s |
| rendered | blurry_soft.jpg | blurry | yes | 1.00 | yes | 1.00 | 1.00 | 0.00 | 0.00 | yes | yes | 0.7s |
| rendered | blurry_motion.jpg | blurry | NO | - | NO | 0.00 | 0.00 | - | - | yes | yes | 4.0s |
| rendered | blurry_heavy.jpg | blurry | NO | - | NO | 0.00 | 0.00 | - | - | yes | yes | 4.0s |
| rendered | dim_light.jpg | dim light | yes | 1.00 | yes | 1.00 | 1.00 | 0.00 | 0.00 | yes | yes | 0.8s |
| rendered | multi_held_and_poster.jpg | multiple regions | yes | 1.00 | yes | 1.00 | 1.00 | 0.00 | 0.40 | yes | yes | 0.8s |
| rendered | multi_two_equal.jpg | multiple regions | - | - | yes | - | - | - | - | yes | yes | 3.5s |
| rendered | no_text_room.jpg | no text | - | - | yes | - | - | - | - | yes | yes | 3.9s |
| rendered | no_text_bottle.jpg | no text | - | - | yes | - | - | - | - | yes | yes | 4.1s |

**Failures, honestly:**

- `blurry_motion` (15 px hand-shake blur) and `blurry_heavy`: EAST finds no text, so SENSE says "I can't find any text..." (generic). It never reads garbage, but the tip should say "hold it still". Measuring blur on a noisy webcam frame is unreliable, so this is not fixed yet.
- `icdar_far_exit` (a real head-camera frame, EXIT sign far away, letters about 15 px): SENSE chose a texture in the middle of the picture, could not read it, and gave a generic tip. Far, small signs are not what "read this" is for, but it shows that EAST boxes some textures.
- `icdar_room_sign` (real, letters 13 to 19 px): half the words read ("Genexts Theatre (06). Carpark"), spoken as "I can only read part of it", with the tip to move the sign into view. It was 0 words before.
- `street_signs` (two separate signs): SENSE asked to frame one sign, which is right by its rules, but it means nothing was read.
- `video_sign`: the Chinese characters on the sign are skipped (English only) and the chosen box covers both, so "inside target" is low.
- At the 640 x 352 detector size (`--east-size 640x352`), speed improves (slowest frame 0.50 s), but on the rendered set words read drop from 0.86 to 0.78 and one rendered image got a plain "It says" for an incomplete reading, so 960 x 544 stays the default.

### Your own captures

Put webcam photos in `ocr_testset/mine/` with a `labels.json` like the ones in `ocr_testset/rendered/` (file -> category, expected text, target box or null), then run `python tools/ocr_eval.py --set mine`. Real captures of your own medicine bottles and door signs are the best test there is; the numbers above are not a promise for them.

## Known limits

- Printed text only. Handwriting, strongly curved bottles and very small print are unreliable. A round bottle label is read in the part that faces the camera; turn the bottle to hear the rest.
- EAST sometimes boxes textures (keyboards, grilles). The target score and the quality check stop most of these, but a far, small sign can lose to a texture in the middle of the picture.
- The detector runs inside the camera loop: one pause of about 0.1 to 0.5 s each time you press R (measured 0.43 s in the cloud inside SENSE). Reading mode ("keep reading") does not use it.
- English only by default. For other languages, install them (`brew install tesseract-lang`) and set `language` in `ocr/config.py`, for example `"eng+hin"`.
- In "quiet" alert mode, reading-mode announcements are not spoken, because they are priority P2. "Read this" answers are always spoken.
