import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import cv2
import numpy as np
import time
import subprocess
import threading
import queue
import argparse
import os
from collections import defaultdict, Counter
import speech_recognition as sr
from ultralytics import YOLO
import memory_db
import spatial
import distance
import answers
from identity import EntityResolver
from events import EventTracker
from notifier import Notifier, P1, P2, P3
from hazards import LoomingDetector, ApproachTracker, SceneHazardDetector
from people import PersonIdentityGate, count_phrase
from continuity import ObjectContinuity
from face_id import FaceID, Enrolment, decide as face_decide
from guidance import FindGuide
import whereabouts

# =====================================================================
# 1. INITIALIZATION & SETUP
# =====================================================================
# Video source: webcam index (default 0) or a path to a recorded video file.
# Example: python sense_master.py --source test_clips/desk.mp4
parser = argparse.ArgumentParser(description="SENSE Master Assistive System")
parser.add_argument("--source", default="0",
                    help="Camera index (e.g. 0) or path to a video file for repeatable testing")
parser.add_argument("--scene-model", default="yolov8n-oiv7.pt",
                    help="door/stairs model; try yolov8s-oiv7.pt if tools/check_doors.py shows it sees your door better")
args = parser.parse_args()
video_source = int(args.source) if args.source.isdigit() else args.source

memory_db.init_database()
print("Starting SENSE Master Spatial Intelligence System (Dual-Tier Object Re-ID)...")
model = YOLO("yolov8s.pt")

# ByteTrack returns its Kalman-FILTERED box, not the detection itself. For a new
# track the filter starts small and "grows" into the real box over ~1 s - which
# looked exactly like walking towards it (found in the smoke run: a false
# "person very close"). Size-based distance and approach need the raw detection,
# so we keep a copy of the detector output before the tracker replaces it.
# Registered before the first model.track() call, so it runs before the tracker.
_raw_dets = {"xyxy": None, "conf": None, "cls": None}


def _stash_raw_detections(predictor):
    b = predictor.results[0].boxes
    _raw_dets.update(xyxy=b.xyxy.cpu().numpy(), conf=b.conf.cpu().numpy(), cls=b.cls.cpu().numpy())


model.add_callback("on_predict_postprocess_end", _stash_raw_detections)


def raw_detection_box(tracked_box, conf, cls_id):
    """The detector's own box behind a tracked box: same class, same score (the
    tracker copies the score of the detection it matched), best overlap."""
    if _raw_dets["xyxy"] is None:
        return tracked_box
    best, best_iou = tracked_box, 0.3
    for xyxy, c, k in zip(_raw_dets["xyxy"], _raw_dets["conf"], _raw_dets["cls"]):
        if int(k) != cls_id or abs(float(c) - conf) > 1e-4:
            continue
        ov = box_iou(tracked_box, tuple(float(v) for v in xyxy))
        if ov > best_iou:
            best, best_iou = tuple(float(v) for v in xyxy), ov
    return best
# Doors / stairs: COCO has no such classes, so a small Open Images model is used
# (downloaded once on the first run, then offline). SENSE still works without it.
try:
    scene_hazards = SceneHazardDetector(YOLO(args.scene_model))
    if not scene_hazards.class_ids:
        raise ValueError("model has no door/stairs classes")
    print(f"🚪 Door/stairs detection on: {sorted(set(scene_hazards.class_ids.values()))}")
except Exception as e:
    scene_hazards = None
    print(f"🚪 Door/stairs detection off ({e}). Connect to the internet once to download {args.scene_model}.")
camera = cv2.VideoCapture(video_source)

recognizer = sr.Recognizer()
recognizer.pause_threshold = 1.0   # silence that ends a question (was 1.3 s)
recognizer.energy_threshold = 300

# High-Speed Optical Flow Parameters (winSize 31x31, Level 3)
feature_params = dict(maxCorners=150, qualityLevel=0.25, minDistance=12, blockSize=7)
lk_params = dict(winSize=(31, 31), maxLevel=3,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 15, 0.03))

ret, old_frame = camera.read()
if not ret:
    raise SystemExit(f"Could not read from video source: {args.source}")
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

# Load Permanent People & Custom Objects from SQLite!
# Audit fix 1: the saved "people profiles" are colour histograms of a SHIRT,
# not of a face. They cannot tell people apart, so they no longer name anyone.
# Only names that a real face check verifies (person_gate) are ever spoken.
legacy_people = memory_db.load_all_people_visual_profiles()
LEGACY_PEOPLE = {n.lower() for n in legacy_people}
known_people = {}            # name -> enrolment time, for people verifiable by FACE
person_gate = PersonIdentityGate()
# Audit fix 4: real identity = face recognition (OpenCV YuNet + SFace, offline
# after a one-time model download). Without it, nobody is ever named.
face_ident = FaceID.create()
enrolment = None             # an active "this is Sneha" face-sample collection
face_eye_px = {}             # track id -> (time, eye distance in px) for distance
FACE_EVERY_N = 3             # identity check on every 3rd frame per person (CPU budget)


def refresh_people():
    """Reload face profiles from SQLite into the recogniser and known_people."""
    global whisper_prompt
    known_people.clear()
    if face_ident is not None:
        face_ident.set_profiles(memory_db.load_face_profiles())
        known_people.update(face_ident.since)
    # Hints only help Whisper SPELL names it has been taught (not "Neha" or "snail").
    # They cannot create an identity: naming needs a verified face.
    hint_names = list(known_people) + [n.title() for n in LEGACY_PEOPLE if n.title() not in known_people]
    # No command words here: on silence Whisper can echo this prompt back, and an
    # echoed "forget" / "rename" was obeyed as a real command (it deleted memory).
    whisper_prompt = ", ".join(hint_names + ["water bottle", "medicine bottle", "phone",
                                            "laptop"]) + "."


whisper_prompt = ""
refresh_people()
if face_ident is not None:
    print(f"🙂 Face recognition on. People enrolled by face: {list(known_people) or 'none yet'}")
live_people = {}             # track id -> what we know about an (unnamed) person in view
known_objects = memory_db.load_all_custom_object_profiles()
if legacy_people:
    print(f"👥 Ignoring {len(legacy_people)} old clothing-colour people profiles "
          f"({', '.join(legacy_people)}): they cannot verify identity.")
print(f"📦 Restored personal items: {list(known_objects.keys())}")

# --- Roadmap state (all used only by the main thread) ---
SESSION_ID = str(int(time.time()))       # world angles are only valid within one run
SESSION_START = time.time()
resolver = EntityResolver(memory_db.load_entities())     # E1 stable identities
event_tracker = EventTracker()                           # E2 debounced events
last_db_write = {}                                       # E2 throttle: name -> time
DB_WRITE_INTERVAL_S = 1.0
last_changes_query = None                                # E6 "what changed" bookmark
looming = LoomingDetector()                              # sudden obstacles, any object
approach = ApproachTracker()                             # fast-growing YOLO boxes, other classes
last_frame_time = None
track_conf = {}                                          # tid -> smoothed detector confidence
guide = FindGuide()                                      # "find my bottle" guidance (guidance.py)
item_whereabouts = whereabouts.create(SESSION_ID)        # "on the table, left of the laptop" (whereabouts/)
guide_target = None                                      # {"kind": item|person|door|stairs, "name": ...}
last_guide_seen = (0.0, [])                              # (time, observations) of the last YOLO frame

# Field-test fix: random things were called "phone". A detection now has to be
# confident ON AVERAGE (exponential moving average per track) and consistent
# for several frames before SENSE believes it. Small items are confused most.
MIN_CONFIDENCE = {"phone": 0.55, "remote": 0.55, "mouse": 0.55, "cup": 0.50,
                  "bottle": 0.45, "book": 0.50, "laptop": 0.50, "person": 0.50}
DEFAULT_MIN_CONFIDENCE = 0.50
CONF_SMOOTHING = 0.3            # EMA weight of the newest frame
MIN_FRAMES_TO_CONFIRM = 4       # was 2
MIN_CLASS_SHARE = 0.6           # winning class must have 60% of the votes
# Audit fix 2: focal length in pixels. Measured once with tools/validate_distance.py
# (stored as f / frame width); until then it is derived from the assumed field of view.
try:
    FOCAL_RATIO = float(memory_db.get_setting("focal_ratio", "0")) or None
except ValueError:
    FOCAL_RATIO = None
FOCAL_UNC = distance.CALIBRATED_FOCAL_UNCERTAINTY if FOCAL_RATIO else distance.FOCAL_UNCERTAINTY
print("📏 Distance: " + ("calibrated focal length." if FOCAL_RATIO else
      "NOT calibrated (assumed 65 degree view). Run tools/validate_distance.py to calibrate."))
try:
    speech_rate = int(memory_db.get_setting("speech_rate", "190"))   # O2 preference
except ValueError:
    speech_rate = 190
print(f"🧩 Restored {sum(1 for e in resolver.entities.values() if e.get('name'))} stable object identities.")

# System States
live_memory = {}
track_votes = defaultdict(list)
track_last_seen = {}       # tid -> last time ByteTrack reported it (to prune the per-track tables)
track_locked_identities = {}
active_people_crops = {}
active_object_crops = {}
is_listening = False
last_camera_move_time = 0

# Voice hand-off queue (listener thread -> main thread).
# The listener thread only records and transcribes, then puts the text here.
# The main loop answers it, so all shared dicts are touched by ONE thread only.
# Items are tuples: ("text", transcript) or ("error", message_to_speak).
voice_queue = queue.Queue()

# Event detection now lives in events.py (Roadmap E2).

# Heading & Display Persistence Buffers
cumulative_camera_heading = 0.0
frame_count = 0

ALLOWED_ITEMS = {
    "cell phone": "phone",
    "remote": "remote",
    "cup": "cup",
    "bottle": "bottle",
    "book": "book",
    "laptop": "laptop",
    "mouse": "mouse"
}

# Roadmap O1: obstacles the COCO-trained YOLOv8 model already recognises.
# They are remembered like other objects and can trigger approach warnings.
HAZARD_ITEMS = {
    "chair": "chair", "couch": "couch", "bed": "bed", "dining table": "table",
    "potted plant": "plant", "tv": "tv", "dog": "dog", "cat": "cat",
    "suitcase": "suitcase", "bench": "bench", "bicycle": "bicycle",
}
ALLOWED_ITEMS.update(HAZARD_ITEMS)
HAZARD_NAMES = set(HAZARD_ITEMS.values())

# Foundation fix 2 + 3: persistent identities and distance-trend proximity
# alerts for the things you can walk into (continuity.py). One instance per
# detector, because they run at different rates.
CONTINUITY_CLASSES = HAZARD_NAMES | {"person"}
coco_objects = ObjectContinuity(CONTINUITY_CLASSES, source="coco")
scene_objects = ObjectContinuity({"door", "stairs", "door handle"}, source="scene")
shift_since_coco = [0.0, 0.0]        # camera image motion since each tracker last ran
shift_since_scene = [0.0, 0.0]
last_scene_announce = {}             # class -> (time, angle) of the last "door ahead"
CONTINUITY_MIN_CREATE_CONF = 0.35    # weaker boxes may only continue an existing object

# Object sizes for distance estimates now live in distance.py (audit fix 2).

# =====================================================================
# 2. AUDIO & 360-DEGREE SPATIAL COMPASS HELPERS
# =====================================================================
speech_process = None

def speak(message):
    global speech_process
    subprocess.run(["killall", "say"], stderr=subprocess.DEVNULL)
    print(f"\n🔊 SENSE: '{message}'")
    speech_process = subprocess.Popen(["say", "-r", str(speech_rate), message])

def is_speaking():
    """True while the last `say` process is still talking (non-blocking check)."""
    return speech_process is not None and speech_process.poll() is None

# Roadmap E3: decides which events are worth saying out loud.
notifier = Notifier(speak, is_speaking, level=memory_db.get_setting("notify_level", "normal"))

try:  # optional text reading (ocr/ package); SENSE runs exactly as before without it
    from ocr import OCRController
    ocr = OCRController.create()
except Exception as e:
    ocr = None
    print(f"📖 Text reading off ({e})")

def get_spatial_zone_from_x(x_coordinate, frame_width=640):
    third = frame_width / 3.0
    if x_coordinate < third:
        return "on your left"
    elif x_coordinate < (third * 2):
        return "in front of you"
    else:
        return "on your right"

def calculate_world_relative_direction(original_zone, camera_heading_degrees):
    zone_angles = {
        "on your left": -30.0,
        "in front of you": 0.0,
        "on your right": 30.0
    }
    original_angle = zone_angles.get(original_zone, 0.0)
    raw_relative_angle = original_angle - camera_heading_degrees
    normalized_angle = ((raw_relative_angle + 180.0) % 360.0) - 180.0

    if abs(normalized_angle) > 135.0:
        return "behind you"
    elif normalized_angle > 45.0:
        return "on your right"
    elif normalized_angle < -45.0:
        return "on your left"
    else:
        return "in front of you"

def format_time_ago(seconds):
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds} seconds ago"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes} minute{'s' if minutes > 1 else ''} ago"
    hours = minutes // 60
    return f"{hours} hour{'s' if hours > 1 else ''} ago"

# =====================================================================
# 3. VISUAL SIGNATURES, COLOR EXTRACTION & RE-ID
# =====================================================================
def get_unified_chroma_signature(crop):
    """Generates light-invariant 2D chrominance signature."""
    if crop is None or crop.size == 0:
        return None
    h, w, _ = crop.shape
    if h < 25 or w < 25:
        return None

    resized = cv2.resize(crop, (60, 80))
    ycrcb = cv2.cvtColor(resized, cv2.COLOR_BGR2YCrCb)
    hist = cv2.calcHist([ycrcb], [1, 2], None, [16, 20], [0, 256, 0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist.astype(np.float32)

def get_dominant_color_name(crop):
    """Accurately identifies white, black, gray, and vibrant colors under warm indoor lighting."""
    if crop is None or crop.size == 0:
        return ""
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    
    # S < 70 means the pixel has low saturation (it's white, gray, or black!)
    color_mask = (s >= 70) & (v >= 60)
    color_count = np.count_nonzero(color_mask)
    total_pixels = crop.shape[0] * crop.shape[1]
    
    # If less than 15% of pixels are strongly colored, it's a neutral object!
    if color_count < (total_pixels * 0.15):
        median_v = np.median(v)
        if median_v < 60:
            return "black"
        elif median_v > 125:
            return "white"
        else:
            return "gray"
            
    # Vibrant saturated color
    median_h = np.median(h[color_mask])
    if median_h < 10 or median_h >= 170: return "red"
    elif median_h < 25: return "orange"
    elif median_h < 35: return "yellow"
    elif median_h < 85: return "green"
    elif median_h < 100: return "cyan"
    elif median_h < 130: return "blue"
    elif median_h < 170: return "purple"
    return ""

def match_custom_object(crop, base_category, threshold=0.55):
    """Matches crop against your personal saved items."""
    if not known_objects:
        return None
    sig = get_unified_chroma_signature(crop)
    if sig is None:
        return None

    best_match = None
    best_score = -1.0
    for name, data in known_objects.items():
        if data["category"] == base_category:
            score = cv2.compareHist(sig, data["sig"], cv2.HISTCMP_CORREL)
            if score > best_score and score > threshold:
                best_score = score
                best_match = name
    return best_match

STOPWORDS = {"now", "here", "there", "it", "this", "that", "me", "you", "them", "someone", "something", "anything", "please"}

def learn_person(name):
    """'This is Sneha'. Audit fix 4: learns the FACE (several samples over a few
    seconds, collected by the main loop so nothing blocks). The old version saved
    the shirt colour, which matched other people."""
    global enrolment
    if face_ident is None:
        speak("Face recognition is off, so I won't name people. "
              + (f"The reason: {FaceID.last_error}" if FaceID.last_error else ""))
        return
    if not active_people_crops:
        speak("I don't see anyone clearly in front of the camera.")
        return
    clean_name = name.strip(" .,!?").title()
    if not clean_name or clean_name.lower() in STOPWORDS:
        speak("That does not seem like a person's name.")
        return
    def face_size(tid):     # the person showing the biggest face is the one being introduced
        crop = active_people_crops[tid]["crop"]
        _, face = face_ident.best_face(crop, (0, 0, crop.shape[1], crop.shape[0]))
        return 0.0 if face is None else float(face[2] * face[3])
    sizes = {tid: face_size(tid) for tid in active_people_crops}
    target = max(sizes, key=sizes.get)
    if sizes[target] == 0.0:
        speak("I can't see a face. Please look at the camera and say it again.")
        return
    enrolment = Enrolment(clean_name, target, time.time())
    speak(f"Okay. {clean_name}, please look at the camera for a moment.")
    print(f"🙂 Enrolling '{clean_name}' on track {target}...")


def finish_enrolment(now):
    """Called by the main loop when the face samples are collected (or time ran out)."""
    global enrolment
    e, enrolment = enrolment, None
    if not e.ok():
        speak(f"I couldn't see {e.name}'s face clearly. Face the camera in good light and say it again.")
        return
    if not e.consistent():
        speak("The face changed while I was learning. Please try again with only one person in view.")
        return
    others = {n: embs for n, embs in face_ident.profiles.items() if n.lower() != e.name.lower()}
    votes = [face_decide(emb, others)[0] for emb in e.samples] if others else []
    clash = [v for v in votes if isinstance(v, str) and v in others]
    if len(clash) > len(votes) // 2:
        speak(f"This face already looks like {clash[0]}. If that is wrong, say forget {clash[0]}, then try again.")
        return
    memory_db.save_face_embeddings(e.name, e.samples, created_at=now)
    refresh_people()
    person_gate.reset_track(e.track_id)   # old "unknown" checks must not count against the new name
    print(f"✨ Enrolled {e.name} by face ({len(e.samples)} samples).")
    speak(f"Got it. I will recognise {e.name} by face.")

def learn_custom_object(custom_name):
    """Voice trainer: strictly locks object category based on what you said."""
    global active_object_crops
    if not active_object_crops:
        speak("I don't see any familiar object in front of the camera to name.")
        return

    clean_name = custom_name.strip().lower()
    
    # Detect expected base category directly from your speech!
    expected_cat = None
    for item in ALLOWED_ITEMS.values():
        if item in clean_name:
            expected_cat = item
            break

    # Pick the target crop that matches what you said, or the largest one
    if expected_cat:
        matching_tids = [tid for tid, d in active_object_crops.items() if d["category"] == expected_cat]
        target_tid = max(matching_tids, key=lambda k: active_object_crops[k]["area"]) if matching_tids else max(active_object_crops.keys(), key=lambda k: active_object_crops[k]["area"])
    else:
        target_tid = max(active_object_crops.keys(), key=lambda k: active_object_crops[k]["area"])

    target_data = active_object_crops[target_tid]
    crop = target_data["crop"]
    base_cat = expected_cat if expected_cat else target_data["category"]
    
    sig = get_unified_chroma_signature(crop)
    if sig is not None:
        known_objects[clean_name] = {"category": base_cat, "sig": sig}
        # Roadmap E1/E2: the track had a generic name ("White Bottle"); retire it so
        # the renamed object does not also produce a false "gone" event.
        old_eid = resolver.track_to_entity.get(target_tid)
        old_name = resolver.name_of(old_eid) if old_eid else None
        if old_name:
            event_tracker.items.pop(old_name.title(), None)
            live_memory.pop(old_name.title(), None)
        track_locked_identities[target_tid] = clean_name.title()
        memory_db.save_custom_object_profile(clean_name, base_cat, sig)
        item_words = clean_name[3:] if clean_name.startswith("my ") else clean_name
        speak(f"Got it. I have permanently memorized your {item_words}.")
        print(f"📦 Successfully memorized item: '{clean_name}' as a {base_cat}!")
    else:
        speak("Could not capture the object clearly. Please try again.")

# =====================================================================
# 4. VOICE INTENT PARSER & WORLD-CENTRIC QUERY ENGINE
# =====================================================================
def extract_name_from_speech(text):
    clean_text = text.lower().strip(".,!?")
    for trigger in ["this is", "my name is", "remember as", "call her", "call him", "call this", "call it"]:
        if trigger in clean_text:
            return clean_text.split(trigger)[-1].strip()
    words = clean_text.split()
    return words[-1] if words else None

def answer_voice_query(user_text):
    user_text = user_text.lower().strip()
    print(f"👂 Transcribed: '{user_text}'")
    current_time = time.time()
    ocr_reply = ocr.handle_command(user_text, current_time) if ocr is not None else None
    if ocr_reply:  # "read this" / "keep reading" / "stop reading"; anything else goes on below
        speak(ocr_reply)
        return
    intent = answers.classify_intent(user_text)   # Roadmap E6 (rule-based, offline)
    print(f"🧭 Intent: {intent}")

    # Intent A: Name Learning ("This is Sneha", "This is my water bottle")
    if intent == "LEARN":
        extracted = extract_name_from_speech(user_text)
        if not extracted:
            speak("I didn't catch the name.")
            return

        # whole words only: "Catherine" contains "cat" but is a person's name
        words = extracted.split()
        is_object_name = any(item in words for item in ALLOWED_ITEMS.values()) or (words[:1] == ["my"] and words[1:2] != ["name"])
        
        if is_object_name:
            learn_custom_object(extracted)
        else:
            learn_person(extracted.capitalize())
        return

    # Field test: "rename my bottle to medicine" keeps the learned look + history
    if intent == "RENAME":
        rename_command(user_text)
        return

    # Roadmap E6 / O2: privacy and preference commands
    if intent == "FORGET":
        forget_command(user_text)
        return
    if intent in ("QUIET", "VERBOSE", "NORMAL"):
        level = {"QUIET": "quiet", "VERBOSE": "verbose", "NORMAL": "normal"}[intent]
        notifier.set_level(level)
        memory_db.set_setting("notify_level", level)
        speak({"quiet": "Quiet mode. I will only warn you about obstacles.",
               "verbose": "Okay, I will tell you about every change I notice.",
               "normal": "Normal mode. I will tell you about your items and people you know."}[level])
        return
    if intent in ("SLOWER", "FASTER"):
        global speech_rate
        speech_rate = max(130, min(260, speech_rate + (-30 if intent == "SLOWER" else 30)))
        memory_db.set_setting("speech_rate", speech_rate)
        speak("Okay, I will speak slower." if intent == "SLOWER" else "Okay, I will speak faster.")
        return

    # "Find my bottle": spoken left / right / straight guidance (guidance.py)
    if intent == "STOP_FIND":
        if guide.active:
            speak(guide.stop("user"))
        else:
            speak("I wasn't looking for anything.")
        return
    if intent == "FIND":
        start_find(user_text, current_time)
        return

    # Intent B: Room Overview ("What do you see?") - what is visible right now
    if intent == "OVERVIEW":
        active_items = []
        for name, data in live_memory.items():
            if current_time - data["last_seen"] <= 2.5:
                active_items.append(f"{name} {data['last_zone']}, {data['dist_phrase']}")
        people_now = [d for d in live_people.values() if current_time - d["last_seen"] <= 2.5]
        if len(people_now) == 1:
            active_items.append(f"a person {people_now[0]['zone']}, {people_now[0]['dist_phrase']}")
        elif people_now:
            active_items.append(count_phrase(len(people_now)))
        # Things you can walk into, tracked by continuity.py: they are reported
        # even while you walk (live_memory above only fills when the camera is
        # steady, so "what do you see" used to answer nothing while moving).
        said = " ".join(active_items).lower()
        for obj in sorted(coco_objects.objects.values(), key=lambda o: -o.hits):
            if obj.cls == "person" or current_time - obj.last_seen > 2.5 or obj.cls in said:
                continue
            angle = spatial.pixel_to_angle((obj.box[0] + obj.box[2]) / 2, globals().get("frame_width", 1280))
            d = obj.current_distance(current_time)
            dist_words = (", " + distance.phrase(distance.Estimate(d, d * 0.8, d * 1.2))) if d else ""
            active_items.append(f"{object_label(obj)} {spatial.direction_phrase(angle)}{dist_words}")
            said += " " + obj.cls
        if scene_hazards is not None:   # doors / stairs from the second model
            for hz_name, hz_angle in scene_hazards.visible(current_time).items():
                active_items.append(f"a {hz_name} {spatial.direction_phrase(hz_angle)}")

        if not active_items:
            speak("I do not see any familiar objects or people right now.")
        else:
            speak("Currently, I see " + " and ".join(active_items) + ".")
        return

    # Doors / stairs: "where is the door?"
    if scene_hazards is not None and intent == "WHERE":
        for hz_name in scene_hazards.HAZARD_NAMES:
            if hz_name in user_text or (hz_name == "stairs" and "stair" in user_text):
                d = scene_hazards.seen.get(hz_name)
                handle = scene_hazards.seen.get(scene_hazards.HANDLE) if hz_name == "door" else None
                if (not d or current_time - d["time"] > scene_hazards.VISIBLE_S) and handle \
                        and current_time - handle["time"] <= scene_hazards.VISIBLE_S:
                    # only the handle is visible: say exactly that (audit fix 3)
                    speak(f"I can't see the whole door, but there is a door handle "
                          f"{spatial.direction_phrase(handle['angle'])}.")
                elif d and current_time - d["time"] <= scene_hazards.VISIBLE_S:
                    speak(f"There is a {hz_name} {spatial.direction_phrase(d['angle'])}.")
                elif d:
                    speak(f"I saw a {hz_name} {answers.format_time_ago(current_time - d['time'])}, "
                          f"but I can't see it right now.")
                else:
                    speak(f"I haven't seen a {hz_name} yet.")
                return

    # Roadmap E6: memory conversations
    if intent == "CHANGES":
        answer_changes(current_time)
        return
    if intent == "WHO":
        answer_who(current_time)
        return
    if intent == "RECENT":
        answer_recent(current_time)
        return

    # Audit fix 1: never answer where a named person is from unverified sightings
    who = asked_about_unverifiable_person(user_text)
    if who:
        speak(f"I only knew {who.capitalize()} by clothing colour, which is not reliable, so I won't guess. "
              f"Please introduce {who.capitalize()} to me again by face.")
        return

    # Intents WHERE / WHEN / PREVIOUS need a target item or person
    found_target = find_target(user_text)
    if not found_target:
        speak("I heard you, but I did not catch which item or person you asked about.")
        return

    if intent == "PREVIOUS":
        answer_previous(found_target, current_time)
        return

    # 1. Live Check ("I can see it right now")
    live_matches = [name for name in live_memory if found_target.lower() in name.lower() and (current_time - live_memory[name]["last_seen"] <= 2.5)]
    if live_matches:
        announcements = []
        for m in live_matches:
            d = live_memory[m]
            rel = live_relation(m)
            announcements.append(f"{spoken_name(m)} {d['last_zone']}, {d['dist_phrase']}" + (f", {rel}" if rel else ""))
        prefix = "Right now, I see " if intent == "WHEN" else "I see "
        speak(prefix + " and ".join(announcements) + ".")
        return

    # 2. Memory: newest matching record, world-anchored and uncertainty-aware (E4 + E5)
    db_record = memory_db.find_latest_memory(found_target)
    person_since = {p.lower(): t for p, t in known_people.items()}.get(found_target.lower())
    if db_record is not None and person_since and db_record["last_seen"] < person_since:
        db_record = None      # that sighting came from clothing colour, not a face
    if db_record is None:
        speak(f"I have no record of {found_target} in my memory database.")
        return
    name = spoken_name(db_record["name"])
    age = current_time - db_record["last_seen"]
    if intent == "WHEN":
        speak(f"I last saw {name} {answers.format_time_ago(age)}, {db_record['last_zone']}.")
        return
    where, turn_hint, same_session = location_words(db_record)
    if person_since:
        # people walk around: report the last verified sighting, never "probably still there"
        speak(f"I last recognised {name} {answers.format_time_ago(age)}, {where}. I can't see them now.")
        return
    relation = memory_relation(db_record) if same_session else None
    anchored = item_whereabouts.describe(db_record, name, age, where, same_session, turn_hint)
    sentence, level, score = anchored or answers.describe_memory(
        name, age, db_record.get("status"), where, same_session, turn_hint, relation)
    sentence += "" if anchored else item_whereabouts.stale_note(db_record, current_time)
    print(f"🎯 Confidence {score:.2f} ({level})")
    speak(sentence[0].upper() + sentence[1:])


# ---------------------------------------------------------------------
# Roadmap helpers: names, locations, relations, memory answers
# ---------------------------------------------------------------------
def spoken_name(name):
    """'my bottle' -> 'your bottle', 'white cup' -> 'the white cup', 'sneha' -> 'Sneha'."""
    lower = name.lower()
    if lower in {p.lower() for p in known_people}:
        return name.capitalize()
    if lower.startswith("my "):
        return "your " + lower[3:]
    return "the " + lower


def find_target(user_text):
    candidates = set(known_objects.keys()) | {p.lower() for p in known_people}
    candidates |= set(ALLOWED_ITEMS.values())
    candidates |= {n.lower() for n in live_memory if n not in ("Unfamiliar Person", "Person")}
    candidates |= set(memory_db.get_all_memory_names())
    candidates -= unverifiable_people()   # old shirt-colour sightings are not evidence
    match = answers.fuzzy_find(user_text, sorted(candidates))   # tolerant of mishearing
    return match.title() if match else None


def unverifiable_people():
    """Names taught with the old clothing-colour method and not verifiable now."""
    return LEGACY_PEOPLE - {p.lower() for p in known_people}


def asked_about_unverifiable_person(user_text):
    names = sorted(unverifiable_people())
    return answers.fuzzy_find(user_text, names) if names else None


def location_words(rec):
    """Where is a remembered thing NOW, relative to the user? (Roadmap E4)"""
    if rec.get("session_id") == SESSION_ID and rec.get("world_angle") is not None:
        rel = spatial.relative_angle(rec["world_angle"], cumulative_camera_heading)
        return spatial.direction_phrase(rel), spatial.turn_instruction(rel), True
    # Seen in an earlier run: the heading was reset, so only the old zone is known
    return rec["last_zone"], None, False


def _relation_candidates(exclude):
    people = {p.lower() for p in known_people}
    out = []
    now = time.time()
    for n, d in live_memory.items():
        if now - d["last_seen"] > 2.5:      # only things visible right now
            continue
        if n.lower() == exclude.lower() or n in ("Unfamiliar Person", "Person") or n.lower() in people:
            continue
        out.append({"name": n.lower(), "world_angle": d.get("world_angle"),
                    "last_y": d.get("last_y"), "last_seen": d["last_seen"]})
    return out


def live_relation(name):
    d = live_memory.get(name)
    if not d or d.get("category") == "person" or name.lower() in {p.lower() for p in known_people}:
        return None
    target = {"name": name.lower(), "world_angle": d.get("world_angle"),
              "last_y": d.get("last_y"), "last_seen": d["last_seen"]}
    return spatial.relation_phrase(target, spatial.nearest_neighbour(target, _relation_candidates(name)))


def memory_relation(rec):
    if rec["name"] in {p.lower() for p in known_people}:
        return None
    target = {"name": rec["name"], "world_angle": rec.get("world_angle"),
              "last_y": rec.get("last_y"), "last_seen": rec["last_seen"]}
    others = [{"name": r["name"], "world_angle": r.get("world_angle"), "last_y": r.get("last_y"),
               "last_seen": r["last_seen"]}
              for r in memory_db.get_memories_seen_since(rec["last_seen"] - 30)
              if r.get("session_id") == SESSION_ID and r["name"] not in {p.lower() for p in known_people}]
    return spatial.relation_phrase(target, spatial.nearest_neighbour(target, others))


def answer_changes(current_time):
    """'What changed?' - events since the last time you asked (max 30 minutes back)."""
    global last_changes_query
    since = current_time - 1800
    if last_changes_query is not None:
        since = max(since, last_changes_query)
    events_list = [ev for ev in memory_db.get_events_since(since)
                   if ev["type"] in ("MOVED", "GONE")
                   or (ev["type"] == "RETURNED" and ev.get("detail"))
                   or (ev["type"] == "APPEARED" and ev["ts"] > SESSION_START + 15)]
    for ev in events_list:
        ev["name"] = spoken_name(ev["name"])
        if ev["type"] == "RETURNED":
            ev["type"] = "MOVED"   # came back somewhere else = it was moved
    last_changes_query = current_time
    summary = answers.summarise_changes(events_list)
    speak(summary[0].upper() + summary[1:])


def answer_who(current_time):
    """'Who was here?' - known people seen today, newest first."""
    lt = time.localtime(current_time)
    midnight = time.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday, 0, 0, 0, 0, 0, -1))
    if not known_people:
        speak("Nobody has been introduced to me by face yet, so I can't tell you who was here."
              if face_ident is not None else
              "Face recognition is off, so I can't tell you who was here.")
        return
    people = {p.lower(): p for p in known_people}
    since = {p.lower(): t for p, t in known_people.items()}
    # only sightings verified by face (older ones came from clothing colour)
    seen = [r for r in memory_db.get_memories_seen_since(midnight)
            if r["name"] in people and r["last_seen"] >= since[r["name"]]]
    if not seen:
        speak("I have not recognised anyone by face today.")
        return
    if not seen:
        speak("I have not recognised anyone I know today.")
        return
    parts = []
    for r in seen:
        age = current_time - r["last_seen"]
        if age <= 2.5:
            parts.append(f"{people[r['name']]}, who is here right now")
        else:
            parts.append(f"{people[r['name']]}, last seen {answers.format_time_ago(age)}")
    speak("Today I saw " + " and ".join(parts) + ".")


def answer_recent(current_time):
    """'What did you see?' - things seen in the last 10 minutes that are not visible now."""
    recent = [r for r in memory_db.get_memories_seen_since(current_time - 600)
              if current_time - r["last_seen"] > 2.5]
    if not recent:
        speak("I have not seen anything else in the last ten minutes.")
        return
    parts = [f"{spoken_name(r['name'])} {answers.format_time_ago(current_time - r['last_seen'])}"
             for r in recent[:5]]
    speak("In the last ten minutes I also saw " + ", ".join(parts) + ".")


def answer_previous(found_target, current_time):
    """'Where was my bottle before?' - uses the event history (Roadmap E2/E6)."""
    history = memory_db.get_item_history(found_target)
    for ev in history:
        if ev["type"] in ("MOVED", "RETURNED") and ev.get("detail"):
            name = spoken_name(ev["name"])
            speak(f"Before it was moved {answers.format_time_ago(current_time - ev['ts'])}, "
                  f"{name} was {ev['detail'].replace('from ', '')}. Now it is {ev['zone']}.")
            return
    if history:
        speak(f"I have not seen {spoken_name(history[0]['name'])} move. "
              f"It was {history[0]['zone']} when I last noticed it.")
    else:
        speak(f"I have no history for {found_target}.")


def rename_command(user_text):
    """'rename my bottle to my medicine' / 'change my bottle to medicine'.

    The object keeps its colour fingerprint, last position and event history;
    only the name changes, in SQLite and in every in-memory table."""
    parsed = answers.parse_rename(user_text)
    if not parsed:
        speak("Say it like this: rename my bottle to my medicine.")
        return
    old_words, new_name = parsed
    candidates = set(known_objects.keys()) | {p.lower() for p in known_people} | set(memory_db.get_all_memory_names())
    old = answers.fuzzy_find(old_words, sorted(candidates))
    if not old:
        speak(f"I don't know anything called {old_words}. You can point the camera at it and say: this is my {new_name.replace('my ', '', 1)}.")
        return
    old = old.lower()
    if old.startswith("my ") and not new_name.startswith("my "):
        new_name = "my " + new_name
    if new_name == old:
        speak(f"It is already called {spoken_name(old)}.")
        return
    memory_db.rename(old, new_name)
    item_whereabouts.rename(old, new_name)
    if old in known_objects:
        known_objects[new_name] = known_objects.pop(old)
    refresh_people()      # face profiles were renamed in SQLite
    for tid, n in list(track_locked_identities.items()):
        if n.lower() == old:
            track_locked_identities[tid] = new_name.title()
    person_gate.rename(old, new_name.title())
    for k in list(live_memory):
        if k.lower() == old:
            live_memory[new_name.title()] = live_memory.pop(k)
    for k in list(event_tracker.items):
        if k.lower() == old:
            event_tracker.items[new_name.title()] = event_tracker.items.pop(k)
    for ent in resolver.entities.values():
        if (ent.get("name") or "").lower() == old:
            ent["name"] = new_name
    print(f"✏️ Renamed '{old}' -> '{new_name}'")
    speak(f"Okay. {spoken_name(old)[0].upper() + spoken_name(old)[1:]} is now called {spoken_name(new_name)}.")


def forget_command(user_text):
    """Privacy: 'forget Sneha' / 'forget my bottle' deletes profile, memory and history."""
    candidates = set(known_objects.keys()) | {p.lower() for p in known_people} | set(memory_db.get_all_memory_names())
    target = answers.fuzzy_find(user_text.replace("forget", " "), sorted(candidates))
    if not target:
        speak("Tell me who or what to forget, for example: forget my bottle.")
        return
    memory_db.forget(target)
    item_whereabouts.forget(target)
    refresh_people()      # face profiles were deleted in SQLite
    for store in (known_people, legacy_people):
        for k in list(store):
            if k.lower() == target.lower():
                del store[k]
    LEGACY_PEOPLE.discard(target.lower())
    person_gate.forget_name(target)
    known_objects.pop(target.lower(), None)
    for k in list(live_memory):
        if k.lower() == target.lower():
            del live_memory[k]
    for tid in [t for t, n in track_locked_identities.items() if n.lower() == target.lower()]:
        del track_locked_identities[tid]
    for k in list(event_tracker.items):
        if k.lower() == target.lower():
            del event_tracker.items[k]
    speak(f"Done. I have forgotten {target}.")


def start_find(user_text, now):
    """Start "find my ..." guidance. The main loop then feeds guide.observe()
    with every sighting of the target and speaks what guide.poll() returns."""
    global guide_target
    target = None
    for hz in ("door", "stairs"):
        if hz in user_text or (hz == "stairs" and "stair" in user_text):
            if scene_hazards is None:
                speak(f"Door and stairs detection is off, so I can't look for the {hz}.")
                return
            target = {"kind": hz, "name": hz}
            label = f"the {hz}"
    if target is None:
        who = asked_about_unverifiable_person(user_text)
        if who:
            speak(f"I can't recognise {who.capitalize()} reliably yet. Please introduce them to me again by face.")
            return
        found = find_target(user_text)
        if not found:
            speak("I can't look for that. I can find a bottle, cup, phone, remote, book, laptop, mouse, "
                  "chair, bed, table, door, stairs, people I know by face, and items you taught me.")
            return
        name = found.lower()
        if name in {p.lower() for p in known_people}:
            target = {"kind": "person", "name": name}
        else:
            target = {"kind": "item", "name": name}
        label = spoken_name(found)
    guide_target = target
    print(f"🔎 Find mode: {target}")

    seen_t, seen = last_guide_seen
    visible_now = [o for o in seen if guide_matches(o)] if now - seen_t <= 0.7 else []
    if visible_now:
        speak(guide.start_visible(label, now, visible_now[0], cumulative_camera_heading))
        return
    speak(guide.start(label, now, find_memory(target, now)))


def find_memory(target, now):
    """Last remembered sighting, for "I last saw it on your left, near the laptop"."""
    if target["kind"] not in ("item", "person"):
        return None
    rec = memory_db.find_latest_memory(target["name"])
    if rec is None:
        return None
    if target["kind"] == "person":
        since = {p.lower(): t for p, t in known_people.items()}.get(target["name"])
        if since and rec["last_seen"] < since:
            return None      # that sighting came from clothing colour, not a face
    where, turn, same_session = location_words(rec)
    relation = memory_relation(rec) if same_session and target["kind"] == "item" else None
    if relation and target["name"].split()[-1] in relation:
        relation = None      # "the bicycle, next to the yellow bicycle": the same thing twice
    relation = (item_whereabouts.relation_for(rec) if target["kind"] == "item" else None) or relation
    if same_session and target["kind"] == "item" and (rec.get("status") or "").upper() == "GONE":
        where += ", but it disappeared from there, so it may have been moved"   # same honesty as "where is"
    return {"world_angle": rec.get("world_angle") if same_session else None,
            "age_text": answers.format_time_ago(now - rec["last_seen"]),
            "where": where if same_session else f"{where} at that time",
            "relation": relation,
            "turn": turn}


def guide_matches(o):
    """Does this sighting belong to the thing we are guiding to?"""
    if guide_target is None:
        return False
    kind, name = guide_target["kind"], guide_target["name"]
    if kind in ("door", "stairs"):
        return o.get("cls") == kind
    if kind == "person":
        return (o.get("person") or "").lower() == name
    if name in known_objects:                  # a personal item: only that item
        return name in o["names"]
    # a generic class ("bottle") or a remembered label ("white bottle")
    return name in o["names"] or name.split()[-1] == o.get("cls")


def box_iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def object_label(obj):
    """Spoken name for a persistent object: a person only by verified face."""
    if obj.cls == "person":
        return person_gate.spoken(obj.bt_id) if obj.bt_id is not None else "a person"
    return obj.cls if obj.cls == "stairs" else f"a {obj.cls}"


def update_coco_objects(dets, now, frame_w, frame_h):
    """Foundation fix 2 + 3: identity continuity and ONE alert per approach."""
    coco_objects.update(dets, now, tuple(shift_since_coco), frame_size=(frame_w, frame_h))
    shift_since_coco[:] = [0.0, 0.0]
    for obj, words in coco_objects.proximity_alerts(now, frame_w):
        where = spatial.direction_phrase(spatial.pixel_to_angle((obj.box[0] + obj.box[2]) / 2, frame_w))
        notifier.push(P1, f"near:coco:{obj.oid}", f"Careful, {object_label(obj)} {where}, {words}")


def update_scene_objects(dets, now, frame_w, frame_h, f_px):
    """Doors / stairs from the second model, with the same continuity layer."""
    cdets = [{"cls": n, "box": b, "conf": c, "bt_id": None, "can_create": True,
              "est": None if n == "door handle" else
              distance.object_distance(n, b, frame_w, frame_h, f_px, focal_unc=FOCAL_UNC)}
             for n, c, b in dets]
    _, created = scene_objects.update(cdets, now, tuple(shift_since_scene), frame_size=(frame_w, frame_h))
    shift_since_scene[:] = [0.0, 0.0]
    for obj, _ in created:
        angle = spatial.pixel_to_angle((obj.box[0] + obj.box[2]) / 2, frame_w)
        prev = last_scene_announce.get(obj.cls)
        last_scene_announce[obj.cls] = (now, angle)
        if prev and now - prev[0] < 10.0 and abs(angle - prev[1]) < 25.0:
            continue      # the same door flickering in and out: say it once
        # was: box taller than 70% of the frame - but a 2 m door 3 m away already
        # is (smoke test). Use the distance estimate, or "wider than the view".
        d = obj.current_distance(now)
        very_close = obj.filled or (d is not None and d <= distance.ARM_REACH_M)
        notifier.push(P1, f"seen:{obj.cls}", f"{obj.cls.capitalize()} {spatial.direction_phrase(angle)}"
                      + (", very close" if very_close else ""))
    for obj, words in scene_objects.proximity_alerts(now, frame_w):
        where = spatial.direction_phrase(spatial.pixel_to_angle((obj.box[0] + obj.box[2]) / 2, frame_w))
        notifier.push(P1, f"near:scene:{obj.oid}", f"Careful, {object_label(obj)} {where}, {words}")


def handle_event(ev):
    """Log an event to SQLite and decide whether it is worth saying (Roadmap E2 + E3)."""
    name, kind = ev["name"], ev["type"]
    detail = f"from {ev['from_zone']}" if ev.get("from_zone") else None
    memory_db.log_event(ev["ts"], name, kind, ev.get("zone"), ev.get("world_angle"),
                        SESSION_ID, ev.get("dist_feet"), detail)
    icon = {"APPEARED": "🆕", "MOVED": "↔️", "GONE": "📤", "OUT_OF_VIEW": "🙈", "RETURNED": "👋"}.get(kind, "•")
    print(f"{icon} {kind}: '{name}' {ev.get('zone') or ''} {detail or ''}".rstrip())
    if kind in ("GONE", "OUT_OF_VIEW"):
        memory_db.set_object_status(name, kind)

    lower = name.lower()
    is_person = lower in {p.lower() for p in known_people}
    is_personal = is_person or lower in known_objects
    if ev.get("world_angle") is not None:
        where = spatial.direction_phrase(spatial.relative_angle(ev["world_angle"], cumulative_camera_heading))
    else:
        where = ev.get("zone") or ""
    label = spoken_name(name)
    label = label[0].upper() + label[1:]

    text = None
    if is_person:
        if kind in ("APPEARED", "RETURNED"):
            text = f"{label} is here, {where}"
        elif kind == "GONE":
            text = f"{label} has left"
    else:
        if kind == "MOVED" or (kind == "RETURNED" and ev.get("from_zone")):
            text = f"{label} was moved. It is now {where}"
        elif kind == "GONE":
            text = f"{label} is no longer {where}"
        elif kind == "APPEARED":
            text = f"I see {spoken_name(name)} {where}"
    if text and not ev.get("brief"):
        notifier.push(P2 if is_personal else P3, lower, text)


# Whisper hint words (whisper_prompt) come from refresh_people(): names are only
# included for people SENSE can verify, so Whisper is not nudged into hearing a
# name that was never said.
EARCON = "/System/Library/Sounds/Tink.aiff"   # short macOS sound: "speak now"
WHISPER_JUNK = {"", "listening", "you", "thank you", "thanks for watching", "bye"}
LISTEN_START_TIMEOUT_S = 5.0     # time to START speaking (as before PR #3: 4.5)
LISTEN_MAX_PHRASE_S = 5.0        # longest question (as before PR #3). Noise can keep the
                                 # mic "hearing speech", so this cap is the worst case.

def listen_in_background():
    """Listener thread: cue, record, transcribe. Never answers (main thread does).

    Speed fix (krish, field test 2): PR #3 waited for a spoken "Listening.",
    then a beep, then 0.4 s of noise calibration, and allowed 8 s phrases:
    about 1.5 s extra before the mic opened and up to 3 s extra recording.
    Now: stop any speech, play only the short beep (~0.15 s, it has ended
    before the mic opens, so the mic cannot hear SENSE), then record with the
    old limits. Timings are printed so slow steps are visible.
    """
    global is_listening
    is_listening = True
    t0 = time.time()
    try:
        subprocess.run(["killall", "say"], stderr=subprocess.DEVNULL)
        if os.path.exists(EARCON):
            subprocess.run(["afplay", EARCON], timeout=2, stderr=subprocess.DEVNULL)
        else:  # non-Mac fallback: a very short spoken cue, finished before the mic opens
            subprocess.run(["say", "-r", str(speech_rate), "Yes?"], timeout=3)
        with sr.Microphone() as source:
            t_mic = time.time()
            print(f"\n🎙️ Listening now... (mic open after {t_mic - t0:.2f}s)")
            audio = recognizer.listen(source, timeout=LISTEN_START_TIMEOUT_S,
                                      phrase_time_limit=LISTEN_MAX_PHRASE_S)
        t_rec = time.time()
        # Microphone is released before the (slower) transcription
        text = recognizer.recognize_whisper(
            audio,
            model="tiny",
            initial_prompt=whisper_prompt
        ).strip()
        t_done = time.time()
        print(f"👂 Heard: '{text}'  ⏱ recorded {t_rec - t_mic:.1f}s, Whisper {t_done - t_rec:.1f}s, "
              f"total {t_done - t0:.1f}s since space")
        if text.lower().strip(" .!?,") in WHISPER_JUNK:
            voice_queue.put(("error", "I didn't hear a question. Press space and speak after the beep."))
        else:
            # Do NOT answer here: hand the text to the main thread.
            voice_queue.put(("text", text))
    except sr.WaitTimeoutError:
        voice_queue.put(("error", "I didn't hear anything. Press space and speak after the beep."))
    except Exception as e:
        print(f"Ear note: {e}")
        voice_queue.put(("error", "Sorry, I had trouble hearing that."))
    finally:
        is_listening = False

def trigger_voice():
    global is_listening
    if not is_listening:
        # Set the flag BEFORE starting the thread, so a second key press
        # can never start a second listener (keeps us at 2 threads max).
        is_listening = True
        threading.Thread(target=listen_in_background, daemon=True).start()

def process_voice_queue():
    """Runs on the main thread: answers any transcripts the listener produced."""
    while True:
        try:
            kind, payload = voice_queue.get_nowait()
        except queue.Empty:
            return
        if kind == "text":
            answer_voice_query(payload)
        else:
            speak(payload)

# Startup Greeting
saved_items = memory_db.get_all_remembered_items()
print(f"💾 Restored {len(saved_items)} objects and {len(known_people)} people.")
startup = f"Sense Master Online. Restored {len(saved_items)} objects and {len(known_people)} people."
if face_ident is None:
    startup += " Face recognition is off, so I will not name people."
elif unverifiable_people():
    n = len(unverifiable_people())
    startup += (f" {n} {'person was' if n == 1 else 'people were'} taught with the old clothing method. "
                f"Please introduce them again by face.")
speak(startup)

# =====================================================================
# 5. MAIN REAL-TIME MULTIMODAL LOOP
# =====================================================================
while True:
    # Answer pending voice questions first, on the main thread, using the
    # crops and memory from the previous frame (before they are cleared below).
    process_voice_queue()

    success, frame = camera.read()
    if not success:
        break

    current_time = time.time()
    frame_height, frame_width, _ = frame.shape
    f_px = distance.focal_px(frame_width, focal_ratio=FOCAL_RATIO)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    raw_frame = frame.copy()   # crops come from here: `frame` gets HUD boxes drawn on it
    frame_count += 1

    # -------------------------------------------------------------
    # SUB-SYSTEM A: ROBUST BACKGROUND OPTICAL FLOW & HEADING
    # -------------------------------------------------------------
    bg_mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
    bg_mask[0:int(frame_height * 0.45), :] = 255

    camera_dx = 0.0
    camera_dy = 0.0
    camera_speed = 0.0

    if p0 is not None and len(p0) > 0:
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        if p1 is not None and st is not None:
            good_new = p1[st == 1]
            good_old = p0[st == 1]
            
            if len(good_new) >= 4:
                displacements = good_new - good_old
                camera_dx = np.median(displacements[:, 0])
                camera_dy = np.median(displacements[:, 1])
                camera_speed = (camera_dx**2 + camera_dy**2)**0.5
                
                deg_shift = -camera_dx * (65.0 / float(frame_width))
                cumulative_camera_heading += deg_shift
                cumulative_camera_heading = ((cumulative_camera_heading + 180.0) % 360.0) - 180.0

                if len(good_new) < 60:
                    fresh_pts = cv2.goodFeaturesToTrack(frame_gray, mask=bg_mask, **feature_params)
                    if fresh_pts is not None:
                        good_new = np.vstack((good_new.reshape(-1, 1, 2), fresh_pts))

                p0 = good_new.reshape(-1, 1, 2)
            else:
                p0 = cv2.goodFeaturesToTrack(frame_gray, mask=bg_mask, **feature_params)
        else:
            p0 = cv2.goodFeaturesToTrack(frame_gray, mask=bg_mask, **feature_params)
    else:
        p0 = cv2.goodFeaturesToTrack(frame_gray, mask=bg_mask, **feature_params)

    old_gray = frame_gray.copy()

    # Sudden obstacle, any kind (wall, door, box, hand): optical-flow looming
    frame_dt = (current_time - last_frame_time) if last_frame_time else 0.0
    last_frame_time = current_time
    ttc = looming.update(frame_gray, frame_dt)
    if ttc is not None:
        notifier.push(P1, "looming", "Stop. Something is right in front of you")
    shift_since_coco[0] += camera_dx
    shift_since_coco[1] += camera_dy
    shift_since_scene[0] += camera_dx
    shift_since_scene[1] += camera_dy
    if scene_hazards is not None:
        scene_dets = scene_hazards.detect(raw_frame, frame_count, current_time)
        if scene_dets is not None:      # the door model ran on this frame
            update_scene_objects(scene_dets, current_time, frame_width, frame_height,
                                 distance.focal_px(frame_width, focal_ratio=FOCAL_RATIO))
    is_camera_moving = camera_speed > 3.0
    if is_camera_moving:
        last_camera_move_time = current_time
    camera_is_stable = (current_time - last_camera_move_time) > 0.8
    if ocr is not None:  # does nothing until asked; tesseract runs as its own process, never blocks
        ocr.feed(raw_frame, current_time, camera_moving=is_camera_moving)
        ocr_msg = ocr.poll(current_time, can_speak=not is_listening and not is_speaking())
        if ocr_msg and ocr_msg.is_answer:     # the user asked: answer now (SENSE is quiet)
            speak(ocr_msg.text)
        elif ocr_msg:                         # "keep reading" news: the notifier lets safety go first
            notifier.push(P2, ocr_msg.key, ocr_msg.text)

    # -------------------------------------------------------------
    # SUB-SYSTEM B: PERCEPTION & TRACKING (YOLOv8s + ByteTrack)
    # -------------------------------------------------------------
    should_run_yolo = camera_is_stable or (frame_count % 3 == 0)

    if should_run_yolo:
        # Foundation fix 2: conf=0.10 (was 0.30). ByteTrack's second stage keeps
        # tracks alive with LOW-score boxes (0.1-0.25) during blur or partial
        # occlusion; filtering them out before tracking broke tracks. SENSE's
        # own per-class confidence gates below still decide what is announced.
        results = model.track(frame, persist=True, conf=0.10, tracker="bytetrack.yaml", verbose=False)
        boxes = results[0].boxes
    else:
        # Foundation fix 2: no stale boxes. Objects keep their identity in
        # continuity.py while YOLO is skipped, but nothing old is shown or used.
        boxes = None
    continuity_dets = []

    candidates = []
    if should_run_yolo:     # skipped frame: keep the last crops ("this is Sneha" while the image moves)
        active_people_crops.clear()
        active_object_crops.clear()
    frame_observations = {}   # name -> this frame's stable observation (Roadmap E2)
    enrol_candidates = []     # (eye px, track id, face embedding) while enrolling
    guide_seen = []           # confirmed sightings this frame, for "find my ..." guidance

    if boxes is not None and boxes.id is not None:
        for box in boxes:
            raw_name = model.names[int(box.cls[0])]
            track_id = int(box.id[0])
            conf = float(box.conf[0])
            track_last_seen[track_id] = current_time
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            # Persistent identity + proximity for things you can walk into
            sense_name = "person" if raw_name == "person" else HAZARD_ITEMS.get(raw_name)
            if sense_name in CONTINUITY_CLASSES:
                rb = raw_detection_box((x1, y1, x2, y2), conf, int(box.cls[0]))
                bx = (max(0, rb[0]), max(0, rb[1]), min(frame_width, rb[2]), min(frame_height, rb[3]))
                if sense_name == "person":
                    eye_t, eye_px_c = face_eye_px.get(track_id, (0.0, None))
                    est_c = distance.person_distance(bx, frame_width, frame_height, f_px,
                                                     eye_px=eye_px_c if current_time - eye_t < 0.5 else None,
                                                     focal_unc=FOCAL_UNC)
                else:
                    est_c = distance.object_distance(sense_name, bx, frame_width, frame_height,
                                                     f_px, focal_unc=FOCAL_UNC)
                continuity_dets.append({"cls": sense_name, "box": bx, "conf": conf, "bt_id": track_id,
                                        "can_create": conf >= CONTINUITY_MIN_CREATE_CONF, "est": est_c})

            # Sudden-obstacle check for the OTHER YOLO classes (box growth),
            # on fresh detections only, even while walking.
            if sense_name not in CONTINUITY_CLASSES and conf >= 0.35 and approach.update(
                    track_id, current_time, y2 - y1, frame_height,
                    spatial.pixel_to_angle(center_x, frame_width),
                    is_person=(raw_name == "person")):
                notifier.push(P1, f"approach:{raw_name}", f"Careful, a {raw_name} right in front of you")

            if raw_name == "person":
                entity_name = "person"
            elif raw_name in ALLOWED_ITEMS:
                entity_name = ALLOWED_ITEMS[raw_name]
            else:
                continue

            # Smoothed confidence per track; drop tracks that are not convincing
            if should_run_yolo:
                prev_conf = track_conf.get(track_id, conf)
                track_conf[track_id] = (1 - CONF_SMOOTHING) * prev_conf + CONF_SMOOTHING * conf
            if track_conf.get(track_id, conf) < MIN_CONFIDENCE.get(entity_name, DEFAULT_MIN_CONFIDENCE):
                continue

            candidates.append({
                "track_id": track_id,
                "name": entity_name,
                "conf": conf,
                "box": (x1, y1, x2, y2),
                "center": (center_x, center_y),
                "cls_id": int(box.cls[0]),
            })

    if should_run_yolo:
        update_coco_objects(continuity_dets, current_time, frame_width, frame_height)

    # Duplicate suppression: two boxes for the SAME thing (YOLO sometimes gives
    # one object two classes). Was: any box within 75 px of a stronger one, which
    # also deleted a phone lying on a laptop. Now the boxes must overlap heavily.
    filtered_candidates = []
    candidates.sort(key=lambda c: c["conf"], reverse=True)

    for cand in candidates:
        cx, cy = cand["center"]
        conflict = False
        for chosen in filtered_candidates:
            if box_iou(cand["box"], chosen["box"]) > 0.7:
                conflict = True
                break
        if not conflict:
            filtered_candidates.append(cand)

    # -------------------------------------------------------------
    # SUB-SYSTEM D: DUAL-TIER OBJECT & PERSON RE-ID
    # -------------------------------------------------------------
    for cand in filtered_candidates:
        tid = cand["track_id"]
        track_votes[tid].append(cand["name"])
        if len(track_votes[tid]) > 15:
            track_votes[tid].pop(0)

        # Lock-on needs a few consistent frames (field-test fix: fewer random "phones")
        winner_name, winner_votes = Counter(track_votes[tid]).most_common(1)[0]
        if len(track_votes[tid]) >= MIN_FRAMES_TO_CONFIRM and winner_votes / len(track_votes[tid]) >= MIN_CLASS_SHARE:
            entity_settled = True   # becomes False while a new object's colour is still being voted
            x1, y1, x2, y2 = cand["box"]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame_width, x2), min(frame_height, y2)
            center_x, center_y = cand["center"]
            
            zone = get_spatial_zone_from_x(center_x, frame_width)
            w_px = x2 - x1
            h_px = y2 - y1
            crop = raw_frame[y1:y2, x1:x2]

                       # --- BRANCH 1: PERSON RE-ID ---
            if winner_name == "person":
                # Audit fix 1 + 4: detection + tracking say "a person is here", never
                # who. A name appears only after face checks pass the identity gate.
                person_gate.touch(tid, current_time)
                enrolling = enrolment is not None
                if face_ident is not None and should_run_yolo and (enrolling or frame_count % FACE_EVERY_N == 0):
                    result, eye_px, emb = face_ident.check(raw_frame, (x1, y1, x2, y2))
                    if not enrolling:
                        person_gate.observe(tid, result)
                    elif emb is not None:
                        enrol_candidates.append((eye_px or 0.0, tid, emb))
                    if eye_px:
                        face_eye_px[tid] = (current_time, eye_px)
                verified = person_gate.label(tid)

                # Audit fix 2: geometry with cut-off detection and an error range;
                # the eye distance of a visible face is the most reliable cue
                eye_t, eye_px = face_eye_px.get(tid, (0.0, None))
                est = distance.person_distance((x1, y1, x2, y2), frame_width, frame_height, f_px,
                                               eye_px=eye_px if current_time - eye_t < 0.5 else None,
                                               focal_unc=FOCAL_UNC)
                display_name = verified if verified else "Person"

                if crop.size > 0:
                    active_people_crops[tid] = {
                        "crop": crop,
                        "area": w_px * h_px,
                        "box": (x1, y1, x2, y2),
                        "current_label": display_name,
                        "frame_width": frame_width
                    }
                is_custom_item = verified is not None

            # --- BRANCH 2: DUAL-TIER OBJECT RE-ID ---
            else:
                est = distance.object_distance(winner_name, (x1, y1, x2, y2), frame_width,
                                               frame_height, f_px, focal_unc=FOCAL_UNC)

                if crop.size > 0:
                    active_object_crops[tid] = {
                        "crop": crop,
                        "area": w_px * h_px,
                        "category": winner_name
                    }

                # Check if this tracklet is locked to a custom item OF THIS SAME CATEGORY!
                is_custom_item = False
                if tid in track_locked_identities:
                    locked_name = track_locked_identities[tid].lower()
                    if locked_name in known_objects and known_objects[locked_name]["category"] == winner_name:
                        display_name = track_locked_identities[tid]
                        is_custom_item = True
                    else:
                        del track_locked_identities[tid]

                if not is_custom_item:
                    custom_match = match_custom_object(crop, winner_name) if crop.size > 0 else None
                    if custom_match:
                        display_name = custom_match.title()
                        track_locked_identities[tid] = display_name
                        is_custom_item = True
                    else:
                        # Roadmap E1: stable identity. Colour is voted over many
                        # frames and the name is frozen once settled, so one
                        # physical object keeps ONE memory record (identity.py).
                        sig = get_unified_chroma_signature(crop) if crop.size > 0 else None
                        color_now = get_dominant_color_name(crop) if crop.size > 0 else ""
                        eid = resolver.resolve(tid, winner_name, sig, color_now,
                                               (center_x, center_y), current_time,
                                               learn=camera_is_stable)
                        display_name = resolver.display_label(eid, winner_name) if eid else winner_name.capitalize()
                        entity_settled = eid is not None and resolver.name_of(eid) is not None

            # "Find my ..." guidance: the detector's own box (not ByteTrack's
            # Kalman-smoothed one, which lags behind while you walk or turn)
            gb = raw_detection_box(cand["box"], cand["conf"], cand["cls_id"])
            gb = (max(0, gb[0]), max(0, gb[1]), min(frame_width, gb[2]), min(frame_height, gb[3]))
            g_est = est if winner_name == "person" else distance.object_distance(
                winner_name, gb, frame_width, frame_height, f_px, focal_unc=FOCAL_UNC)
            guide_seen.append({"names": {display_name.lower(), winner_name}, "cls": winner_name,
                               "person": verified if winner_name == "person" else None,
                               "angle": spatial.pixel_to_angle((gb[0] + gb[2]) / 2, frame_width),
                               "dist_m": g_est.metres,
                               "max_m": g_est.high if g_est.upper_bound_only else None,
                               "filled": distance.clipped_edges(gb, frame_width, frame_height) >= {"left", "right"}})

            dist_feet = distance.feet_or_none(est)     # None when unknown
            dist_phrase = distance.phrase(est)         # a range or a band, never false precision

            # Update Live Memory (Roadmap E4: every observation gets a WORLD angle)
            if camera_is_stable and display_name == "Person":
                # unnamed people are counted per track, not merged under one name
                live_people[tid] = {"last_seen": current_time, "zone": zone, "dist_phrase": dist_phrase}
            elif camera_is_stable:
                is_trackable = entity_settled
                obs_angle = spatial.world_angle(center_x, frame_width, cumulative_camera_heading)

                live_memory[display_name] = {
                    "last_zone": zone,
                    "last_seen": current_time,
                    "dist_feet": dist_feet,
                    "dist_phrase": dist_phrase,
                    "last_x": center_x,
                    "last_y": center_y,
                    "world_angle": obs_angle,
                    "category": winner_name,
                }

                if is_trackable:
                    # Events and SQLite writes happen once per frame below (E2)
                    frame_observations[display_name] = {
                        "zone": zone, "world_angle": obs_angle,
                        "x": center_x, "y": center_y, "dist_feet": dist_feet,
                        "box": (x1, y1, x2, y2), "category": winner_name,   # for whereabouts/
                    }

            # Visual Rendering
            if is_custom_item:
                box_color = (0, 255, 120)  # Green: Personal item or known person!
            elif winner_name == "person":
                box_color = (0, 165, 255)  # Orange: Unfamiliar person
            else:
                box_color = (0, 220, 255)  # Cyan: Color-tagged generic object

            tag_status = f"({dist_phrase})" if camera_is_stable else "(CAM MOVING)"
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            cv2.putText(
                frame, 
                f"{display_name} {tag_status}", 
                (x1, max(25, y1 - 8)), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.55, 
                box_color,
                2
            )

    # -------------------------------------------------------------
    # SUB-SYSTEM D.1: EVENTS, MEMORY WRITES & NOTIFICATIONS (Roadmap E2/E3)
    # The state machine in events.py waits GONE_AFTER_S before declaring
    # an item gone, and tells "gone" apart from "you turned away"
    # (OUT_OF_VIEW) using the heading. SQLite is written on events, and
    # otherwise at most once per second per item (was: every frame).
    # -------------------------------------------------------------
    new_events = event_tracker.update(frame_observations, current_time,
                                      camera_is_stable, cumulative_camera_heading)
    moved_now = set()
    for ev in new_events:
        handle_event(ev)
        if ev["type"] == "MOVED":
            moved_now.add(ev["name"])

    if camera_is_stable:
        event_names = {ev["name"] for ev in new_events}
        for name, obs in frame_observations.items():
            if name in event_names or current_time - last_db_write.get(name, 0) >= DB_WRITE_INTERVAL_S:
                memory_db.save_object_observation(
                    name.lower(), obs["zone"], obs["x"], obs["y"],
                    status="MOVED" if name in moved_now else "VISIBLE",
                    world_angle=obs["world_angle"], session_id=SESSION_ID,
                    dist_feet=obs["dist_feet"], seen_at=current_time)
                last_db_write[name] = current_time
    item_whereabouts.observe(frame_observations, current_time, frame_width, frame_height, camera_is_stable)

    # Persist newly named object identities (rare) and free old track buffers
    for eid, ent in resolver.pop_dirty():
        if ent.get("name"):
            memory_db.save_entity(eid, ent["name"], ent["category"], ent["color"], ent["sig"], ent["last_seen"])
    resolver.prune_tracks(current_time)
    person_gate.prune(current_time)
    for old_tid in [t for t, (ts, _) in face_eye_px.items() if current_time - ts > 3]:
        del face_eye_px[old_tid]
    if enrolment is not None and enrol_candidates:
        # the biggest clear face this frame, whatever its track id (see Enrolment)
        _, enrol_tid, enrol_emb = max(enrol_candidates, key=lambda c: c[0])
        enrolment.add(enrol_emb)
        enrolment.track_id = enrol_tid
    if enrolment is not None and enrolment.finished(current_time):
        finish_enrolment(current_time)
    for old_tid in [t for t, d in live_people.items() if current_time - d["last_seen"] > 5]:
        del live_people[old_tid]

    # "Find my ..." guidance: feed it this frame's sightings of the target.
    # Only on frames where a detector ran (a skipped frame is not "not seen").
    if should_run_yolo:
        last_guide_seen = (current_time, guide_seen)
        if guide.active and guide_target["kind"] in ("item", "person"):
            guide.observe(current_time, [o for o in guide_seen if guide_matches(o)], cumulative_camera_heading)
    if guide.active and guide_target["kind"] in ("door", "stairs") and scene_hazards is not None \
            and scene_objects.last_update == current_time:
        guide.observe(current_time, [
            {"angle": spatial.pixel_to_angle((o.box[0] + o.box[2]) / 2, frame_width),
             "dist_m": o.current_distance(current_time), "max_m": None, "filled": o.filled}
            for o in scene_objects.visible(current_time) if o.cls == guide_target["kind"]],
            cumulative_camera_heading)

    # Roadmap E3: speak at most one (merged) announcement, only if it matters.
    # While guiding, only safety alerts (P1) are spoken; other news waits.
    spoken_now = notifier.tick(camera_moving=is_camera_moving, hold=is_listening or guide.active)
    if guide.active:
        # Guidance never talks over anything and always yields to hazards:
        # it speaks only when SENSE is silent and no safety alert is pending
        # or was said in the last 2 s.
        can_guide = (spoken_now is None and not is_speaking() and not is_listening
                     and notifier.safety_clear())
        cue = guide.poll(current_time, cumulative_camera_heading, can_speak=can_guide)
        if cue:
            print(f"🧭 Guide: {cue}")
            speak(cue)
    approach.prune(current_time)
    # Stress test (10,000 frames): track_votes / track_conf kept one entry per
    # ByteTrack id ever seen and never shrank. Forget ids unseen for 30 s.
    for old_tid in [t for t, ts in track_last_seen.items() if current_time - ts > 30]:
        del track_last_seen[old_tid]
        track_votes.pop(old_tid, None)
    for old_tid in [t for t in track_conf if t not in track_votes and t not in track_last_seen]:
        del track_conf[old_tid]

    # -------------------------------------------------------------
    # SUB-SYSTEM E: HEADS-UP DISPLAY (HUD)
    # -------------------------------------------------------------
    heading_display = f"HEAD: {int(cumulative_camera_heading)}°"
    if guide.active:
        cv2.putText(frame, f"FINDING: {guide.label} ({guide.state})", (20, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)
    if is_camera_moving:
        cv2.putText(frame, f"CAM MOVING ({heading_display})", (20, 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
    elif is_listening:
        cv2.putText(frame, "STATUS: LISTENING... SPEAK NOW!", (20, 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    else:
        cv2.putText(frame, f"SENSE BRAIN ACTIVE ({heading_display})", (20, 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

    if scene_hazards is not None:   # doors / stairs in purple
        for hz_name, hx1, hy1, hx2, hy2 in scene_hazards.boxes:
            cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), (200, 0, 200), 2)
            cv2.putText(frame, hz_name.title(), (hx1, max(15, hy1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 0, 200), 2)
    cv2.imshow("SENSE Master Assistive System", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        subprocess.run(["killall", "say"], stderr=subprocess.DEVNULL)
        break
    elif key == ord(' '):
        trigger_voice()
    elif key == ord('r') and ocr is not None:   # read the text in front of the camera
        speak(ocr.handle_command("read this", current_time))

# Save the learned appearance of every named object identity, then close the DB
for eid, ent in resolver.entities.items():
    if ent.get("name"):
        memory_db.save_entity(eid, ent["name"], ent["category"], ent["color"], ent["sig"], ent["last_seen"])
memory_db.close()
if ocr is not None:
    ocr.close()

camera.release()
cv2.destroyAllWindows()