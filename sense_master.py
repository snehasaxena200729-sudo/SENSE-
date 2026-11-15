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
import answers
from identity import EntityResolver
from events import EventTracker
from notifier import Notifier, P1, P2, P3
from hazards import LoomingDetector, ApproachTracker

# =====================================================================
# 1. INITIALIZATION & SETUP
# =====================================================================
# Video source: webcam index (default 0) or a path to a recorded video file.
# Example: python sense_master.py --source test_clips/desk.mp4
parser = argparse.ArgumentParser(description="SENSE Master Assistive System")
parser.add_argument("--source", default="0",
                    help="Camera index (e.g. 0) or path to a video file for repeatable testing")
args = parser.parse_args()
video_source = int(args.source) if args.source.isdigit() else args.source

memory_db.init_database()
print("Starting SENSE Master Spatial Intelligence System (Dual-Tier Object Re-ID)...")
model = YOLO("yolov8s.pt")
camera = cv2.VideoCapture(video_source)

recognizer = sr.Recognizer()
recognizer.pause_threshold = 1.3
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
known_people = memory_db.load_all_people_visual_profiles()
known_objects = memory_db.load_all_custom_object_profiles()
print(f"👥 Restored people profiles: {list(known_people.keys())}")
print(f"📦 Restored personal items: {list(known_objects.keys())}")

# --- Roadmap state (all used only by the main thread) ---
SESSION_ID = str(int(time.time()))       # world angles are only valid within one run
SESSION_START = time.time()
resolver = EntityResolver(memory_db.load_entities())     # E1 stable identities
event_tracker = EventTracker()                           # E2 debounced events
last_db_write = {}                                       # E2 throttle: name -> time
DB_WRITE_INTERVAL_S = 1.0
hazard_history = defaultdict(list)                       # O1: tid -> [(time, box height)]
last_changes_query = None                                # E6 "what changed" bookmark
looming = LoomingDetector()                              # sudden obstacles, any object
approach = ApproachTracker()                             # fast-growing YOLO boxes, any class
last_frame_time = None
track_conf = {}                                          # tid -> smoothed detector confidence

# Field-test fix: random things were called "phone". A detection now has to be
# confident ON AVERAGE (exponential moving average per track) and consistent
# for several frames before SENSE believes it. Small items are confused most.
MIN_CONFIDENCE = {"phone": 0.55, "remote": 0.55, "mouse": 0.55, "cup": 0.50,
                  "bottle": 0.45, "book": 0.50, "laptop": 0.50, "person": 0.50}
DEFAULT_MIN_CONFIDENCE = 0.50
CONF_SMOOTHING = 0.3            # EMA weight of the newest frame
MIN_FRAMES_TO_CONFIRM = 4       # was 2
MIN_CLASS_SHARE = 0.6           # winning class must have 60% of the votes
try:
    speech_rate = int(memory_db.get_setting("speech_rate", "190"))   # O2 preference
except ValueError:
    speech_rate = 190
print(f"🧩 Restored {sum(1 for e in resolver.entities.values() if e.get('name'))} stable object identities.")

# System States
live_memory = {}
track_votes = defaultdict(list)
track_identity_history = defaultdict(list)
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
last_detected_boxes = None

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

OBJECT_REAL_SIZES_CM = {
    "chair": 90.0, "couch": 85.0, "bed": 60.0, "table": 75.0, "plant": 50.0,
    "tv": 60.0, "dog": 50.0, "cat": 30.0, "suitcase": 60.0, "bench": 80.0, "bicycle": 100.0,
    "phone": 15.0,
    "remote": 18.0,
    "bottle": 23.0,
    "cup": 10.0,
    "laptop": 26.0,
    "book": 23.0,
    "mouse": 11.0
}

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

def format_distance(distance_feet):
    if distance_feet < 2.2:
        return "within arm's reach"
    elif distance_feet < 5.5:
        return f"about {round(distance_feet)} feet away"
    else:
        return f"further away, about {round(distance_feet)} feet"

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

def match_person(crop, threshold=0.50):
    if not known_people:
        return "Unfamiliar Person"
    h, w, _ = crop.shape
    if h < 30 or w < 30:
        return "Unfamiliar Person"

    # Pure Torso Crop: Focus strictly on chest/clothing, completely ignoring the room wall!
    torso = crop[int(h * 0.25):int(h * 0.75), int(w * 0.20):int(w * 0.80)]
    sig = get_unified_chroma_signature(torso)
    if sig is None:
        return "Unfamiliar Person"

    best_match = "Unfamiliar Person"
    best_score = -1.0
    for name, saved_sig in known_people.items():
        score = cv2.compareHist(sig, saved_sig, cv2.HISTCMP_CORREL)
        if score > best_score and score > threshold:
            best_score = score
            best_match = name
    return best_match

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
    global active_people_crops
    if not active_people_crops:
        speak("I don't see anyone clearly in front of the camera to name.")
        return

    clean_name = name.strip().capitalize()
    if clean_name.lower() in ["neha", "snea", "sneh", "snail"]:
        clean_name = "Sneha"

    if clean_name.lower() in STOPWORDS:
        speak("That does not seem like a person's name.")
        return

    unfamiliar_candidates = {
        tid: data for tid, data in active_people_crops.items()
        if data["current_label"] == "Unfamiliar Person"
    }

    if unfamiliar_candidates:
        target_track_id = max(unfamiliar_candidates.keys(), key=lambda k: unfamiliar_candidates[k]["area"])
    else:
        target_track_id = max(active_people_crops.keys(), key=lambda k: active_people_crops[k]["area"])

    target_data = active_people_crops[target_track_id]
    h, w, _ = target_data["crop"].shape
    # Pure Torso Crop for permanent memorization
    torso = target_data["crop"][int(h * 0.25):int(h * 0.75), int(w * 0.20):int(w * 0.80)]
    sig = get_unified_chroma_signature(torso)
    
    if sig is not None:
        known_people[clean_name] = sig
        track_identity_history[target_track_id] = [clean_name] * 12
        track_locked_identities[target_track_id] = clean_name
        memory_db.save_person_visual_profile(clean_name, sig, sig)
        
        box = target_data["box"]
        cx = int((box[0] + box[2]) / 2)
        cy = int((box[1] + box[3]) / 2)
        zone = get_spatial_zone_from_x(cx, target_data["frame_width"])
        memory_db.save_object_observation(
            clean_name.lower(), zone, cx, cy, status="VISIBLE",
            world_angle=spatial.world_angle(cx, target_data["frame_width"], cumulative_camera_heading),
            session_id=SESSION_ID)

        speak(f"Got it. I have permanently memorized {clean_name}.")
        print(f"✨ Successfully memorized: {clean_name}")
    else:
        speak("Could not capture person clearly. Please try again.")

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

# Adapt loaded profiles from SQLite
for p_name in list(known_people.keys()):
    tup = known_people[p_name]
    if isinstance(tup, tuple):
        known_people[p_name] = tup[0]

# =====================================================================
# 4. VOICE INTENT PARSER & WORLD-CENTRIC QUERY ENGINE
# =====================================================================
def extract_name_from_speech(text):
    clean_text = text.lower().strip(".,!?")
    for trigger in ["this is", "my name is", "remember as", "call her", "call him", "call this", "call it"]:
        if trigger in clean_text:
            raw_name = clean_text.split(trigger)[-1].strip()
            if raw_name in ["neha", "snea", "sneh", "snail"]:
                return "Sneha"
            return raw_name
    words = clean_text.split()
    return words[-1] if words else None

def answer_voice_query(user_text):
    user_text = user_text.lower().strip()
    print(f"👂 Transcribed: '{user_text}'")
    current_time = time.time()
    intent = answers.classify_intent(user_text)   # Roadmap E6 (rule-based, offline)
    print(f"🧭 Intent: {intent}")

    # Intent A: Name Learning ("This is Sneha", "This is my water bottle")
    if intent == "LEARN":
        extracted = extract_name_from_speech(user_text)
        if not extracted:
            speak("I didn't catch the name.")
            return

        is_object_name = any(item in extracted for item in list(ALLOWED_ITEMS.values())) or ("my " in extracted and not extracted.startswith("my name"))
        
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

    # Intent B: Room Overview ("What do you see?") - what is visible right now
    if intent == "OVERVIEW":
        active_items = []
        for name, data in live_memory.items():
            if current_time - data["last_seen"] <= 2.5:
                active_items.append(f"{name} {data['last_zone']}, {data['dist_phrase']}")

        if not active_items:
            speak("I do not see any familiar objects or people right now.")
        else:
            speak("Currently, I see " + " and ".join(active_items) + ".")
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
    if db_record is None:
        speak(f"I have no record of {found_target} in my memory database.")
        return
    name = spoken_name(db_record["name"])
    age = current_time - db_record["last_seen"]
    if intent == "WHEN":
        speak(f"I last saw {name} {answers.format_time_ago(age)}, {db_record['last_zone']}.")
        return
    where, turn_hint, same_session = location_words(db_record)
    relation = memory_relation(db_record) if same_session else None
    sentence, level, score = answers.describe_memory(
        name, age, db_record.get("status"), where, same_session, turn_hint, relation)
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
    if "neha" in user_text or "sneha" in user_text:
        return "Sneha"
    if "mummy" in user_text or "mom" in user_text:
        return "Mummy"
    candidates = set(known_objects.keys()) | {p.lower() for p in known_people}
    candidates |= set(ALLOWED_ITEMS.values())
    candidates |= {n.lower() for n in live_memory if n not in ("Unfamiliar Person", "Person")}
    candidates |= set(memory_db.get_all_memory_names())
    match = answers.fuzzy_find(user_text, sorted(candidates))   # tolerant of mishearing
    return match.title() if match else None


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
    people = {p.lower(): p for p in known_people}
    seen = [r for r in memory_db.get_memories_seen_since(midnight) if r["name"] in people]
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
    if old in known_objects:
        known_objects[new_name] = known_objects.pop(old)
    for k in list(known_people):
        if k.lower() == old:
            known_people[new_name.title()] = known_people.pop(k)
    for tid, n in list(track_locked_identities.items()):
        if n.lower() == old:
            track_locked_identities[tid] = new_name.title()
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
    for store in (known_people,):
        for k in list(store):
            if k.lower() == target.lower():
                del store[k]
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


def check_hazard(tid, display_name, category, box_h, dist_feet, center_x, frame_width, now):
    """Roadmap O1: warn when an obstacle or person is ahead AND getting closer.

    'Getting closer' = the box grew by more than 15% within about half a second
    (looming). Requiring growth avoids nagging about furniture that simply
    sits near a stationary camera.
    """
    hist = hazard_history[tid]
    hist.append((now, box_h))
    while hist and now - hist[0][0] > 1.0:
        hist.pop(0)
    old = [h for t, h in hist if now - t >= 0.4]
    if not old:
        return
    growth = box_h / max(1.0, old[-1])
    ahead = abs(spatial.pixel_to_angle(center_x, frame_width)) < 15
    if ahead and dist_feet < 5.0 and growth > 1.15:
        if category == "person":
            who = display_name if display_name not in ("Unfamiliar Person", "Person") else "someone"
            text = f"Careful, {who} is coming towards you, {format_distance(dist_feet)}"
        else:
            text = f"Careful, {category} ahead, {format_distance(dist_feet)}"
        notifier.push(P1, f"hazard:{tid}", text)


EARCON = "/System/Library/Sounds/Tink.aiff"   # short macOS sound: "speak now"
WHISPER_JUNK = {"", "listening", "you", "thank you", "thanks for watching", "bye"}

def listen_in_background():
    """Listener thread: prompt, record, transcribe. Never answers (main thread does).

    Field-test fix (voice window closed after ~1 s): the old code started the
    microphone 0.7 s after starting to say "Listening.", so the mic heard SENSE's
    own voice, treated it as the question and stopped. Now we WAIT for the
    prompt to finish, play a short beep, calibrate to the room's noise, and give
    the user up to 7 s to start and 8 s to speak.
    """
    global is_listening
    is_listening = True
    try:
        subprocess.run(["killall", "say"], stderr=subprocess.DEVNULL)
        subprocess.run(["say", "-r", str(speech_rate), "Listening."], timeout=5)  # blocks until spoken
        if os.path.exists(EARCON):
            subprocess.run(["afplay", EARCON], timeout=2, stderr=subprocess.DEVNULL)
        with sr.Microphone() as source:
            print("\n🎙️ Listening now... (Say: 'This is my water bottle' or 'Where is the bottle?')")
            recognizer.adjust_for_ambient_noise(source, duration=0.4)
            recognizer.energy_threshold = max(recognizer.energy_threshold, 300)
            audio = recognizer.listen(source, timeout=7.0, phrase_time_limit=8.0)
        # Microphone is released before the (slower) transcription
        text = recognizer.recognize_whisper(
            audio,
            model="tiny",
            initial_prompt="Sneha, Mummy, water bottle, medicine bottle, phone, laptop, rename, forget."
        ).strip()
        print(f"👂 Heard: '{text}'")
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
speak(f"Sense Master Online. Restored {len(saved_items)} objects and {len(known_people)} people.")

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
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_count += 1

    # -------------------------------------------------------------
    # SUB-SYSTEM A: ROBUST BACKGROUND OPTICAL FLOW & HEADING
    # -------------------------------------------------------------
    bg_mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
    bg_mask[0:int(frame_height * 0.45), :] = 255

    camera_speed = 0.0
    camera_dx = 0.0

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
    is_camera_moving = camera_speed > 3.0
    if is_camera_moving:
        last_camera_move_time = current_time
    camera_is_stable = (current_time - last_camera_move_time) > 0.8

    # -------------------------------------------------------------
    # SUB-SYSTEM B: PERCEPTION & TRACKING (YOLOv8s + ByteTrack)
    # -------------------------------------------------------------
    should_run_yolo = camera_is_stable or (frame_count % 3 == 0)

    if should_run_yolo:
        results = model.track(frame, persist=True, conf=0.30, tracker="bytetrack.yaml", verbose=False)
        boxes = results[0].boxes
        if boxes is not None and len(boxes) > 0:
            last_detected_boxes = boxes
    else:
        # RETAIN previous boxes so they NEVER vanish or blink!
        boxes = last_detected_boxes

    candidates = []
    active_people_crops.clear()
    active_object_crops.clear()
    frame_observations = {}   # name -> this frame's stable observation (Roadmap E2)

    if boxes is not None and boxes.id is not None:
        for box in boxes:
            raw_name = model.names[int(box.cls[0])]
            track_id = int(box.id[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            # Sudden-obstacle check for ANY of YOLO's 80 classes (not just the
            # whitelist), on fresh detections only, even while walking.
            if should_run_yolo and conf >= 0.35 and approach.update(
                    track_id, current_time, y2 - y1, frame_height,
                    spatial.pixel_to_angle(center_x, frame_width)):
                what = "someone" if raw_name == "person" else f"a {raw_name}"
                notifier.push(P1, f"approach:{track_id}", f"Careful, {what} right in front of you")

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
                "center": (center_x, center_y)
            })

    # Spatial Conflict Resolution (<75 px)
    filtered_candidates = []
    candidates.sort(key=lambda c: c["conf"], reverse=True)

    for cand in candidates:
        cx, cy = cand["center"]
        conflict = False
        for chosen in filtered_candidates:
            chosen_cx, chosen_cy = chosen["center"]
            if ((cx - chosen_cx)**2 + (cy - chosen_cy)**2)**0.5 < 75:
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
            crop = frame[y1:y2, x1:x2]

                       # --- BRANCH 1: PERSON RE-ID ---
            if winner_name == "person":
                aspect_ratio = h_px / max(1, w_px)
                real_size = 165.0 if aspect_ratio > 2.0 else 75.0
                dist_feet = (real_size * 600.0) / (h_px * 30.48)
                
                if crop.size > 0:
                    matched_label = match_person(crop)
                else:
                    matched_label = "Unfamiliar Person"

                track_identity_history[tid].append(matched_label)
                if len(track_identity_history[tid]) > 10:
                    track_identity_history[tid].pop(0)

                counts = Counter(track_identity_history[tid])

                # Check active track lock
                if tid in track_locked_identities:
                    # If this person has consistently scored unfamiliar, release the lock!
                    if counts["Unfamiliar Person"] >= 6:
                        del track_locked_identities[tid]
                        display_name = "Unfamiliar Person"
                    else:
                        display_name = track_locked_identities[tid]
                else:
                    # Require at least 4 consistent frames to lock identity
                    known_matches = [m for m in counts if m != "Unfamiliar Person" and counts[m] >= 4]
                    if known_matches:
                        display_name = known_matches[0]
                        track_locked_identities[tid] = display_name
                    else:
                        display_name = "Unfamiliar Person"

                if crop.size > 0:
                    active_people_crops[tid] = {
                        "crop": crop,
                        "area": w_px * h_px,
                        "box": (x1, y1, x2, y2),
                        "current_label": display_name,
                        "frame_width": frame_width
                    }
                is_custom_item = (display_name != "Unfamiliar Person")

            # --- BRANCH 2: DUAL-TIER OBJECT RE-ID ---
            else:
                real_size = OBJECT_REAL_SIZES_CM.get(winner_name, 18.0)
                measuring_dim = max(w_px, h_px)
                dist_feet = (real_size * 600.0) / (measuring_dim * 30.48)

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

            dist_phrase = format_distance(dist_feet)

            # Update Live Memory (Roadmap E4: every observation gets a WORLD angle)
            if camera_is_stable:
                is_trackable = display_name not in ["Unfamiliar Person", "Person"] and entity_settled
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

    # Persist newly named object identities (rare) and free old track buffers
    for eid, ent in resolver.pop_dirty():
        if ent.get("name"):
            memory_db.save_entity(eid, ent["name"], ent["category"], ent["color"], ent["sig"], ent["last_seen"])
    resolver.prune_tracks(current_time)
    for old_tid in [t for t, hist in hazard_history.items() if not hist or current_time - hist[-1][0] > 5]:
        del hazard_history[old_tid]

    # Roadmap E3: speak at most one (merged) announcement, only if it matters
    notifier.tick(camera_moving=is_camera_moving, hold=is_listening)
    approach.prune(current_time)
    for old_tid in [t for t in track_conf if t not in track_votes]:
        del track_conf[old_tid]

    # -------------------------------------------------------------
    # SUB-SYSTEM E: HEADS-UP DISPLAY (HUD)
    # -------------------------------------------------------------
    heading_display = f"HEAD: {int(cumulative_camera_heading)}°"
    if is_camera_moving:
        cv2.putText(frame, f"CAM MOVING ({heading_display})", (20, 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
    elif is_listening:
        cv2.putText(frame, "STATUS: LISTENING... SPEAK NOW!", (20, 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    else:
        cv2.putText(frame, f"SENSE BRAIN ACTIVE ({heading_display})", (20, 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

    cv2.imshow("SENSE Master Assistive System", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        subprocess.run(["killall", "say"], stderr=subprocess.DEVNULL)
        break
    elif key == ord(' '):
        trigger_voice()

# Save the learned appearance of every named object identity, then close the DB
for eid, ent in resolver.entities.items():
    if ent.get("name"):
        memory_db.save_entity(eid, ent["name"], ent["category"], ent["color"], ent["sig"], ent["last_seen"])
memory_db.close()

camera.release()
cv2.destroyAllWindows()