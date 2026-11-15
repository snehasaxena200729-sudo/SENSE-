import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import cv2
import numpy as np
import time
import subprocess
import threading
from collections import defaultdict, Counter
import speech_recognition as sr
from ultralytics import YOLO
import memory_db

# =====================================================================
# 1. INITIALIZATION & SETUP
# =====================================================================
memory_db.init_database()
print("Starting SENSE Master Spatial Intelligence System (Dual-Tier Object Re-ID)...")
model = YOLO("yolov8s.pt")
camera = cv2.VideoCapture(0)

recognizer = sr.Recognizer()
recognizer.pause_threshold = 1.3
recognizer.energy_threshold = 300

# High-Speed Optical Flow Parameters (winSize 31x31, Level 3)
feature_params = dict(maxCorners=150, qualityLevel=0.25, minDistance=12, blockSize=7)
lk_params = dict(winSize=(31, 31), maxLevel=3,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 15, 0.03))

ret, old_frame = camera.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

# Load Permanent People & Custom Objects from SQLite!
known_people = memory_db.load_all_people_visual_profiles()
known_objects = memory_db.load_all_custom_object_profiles()
print(f"👥 Restored people profiles: {list(known_people.keys())}")
print(f"📦 Restored personal items: {list(known_objects.keys())}")

# System States
live_memory = {}
track_votes = defaultdict(list)
track_identity_history = defaultdict(list)
track_locked_identities = {}
active_people_crops = {}
active_object_crops = {}
is_listening = False
last_camera_move_time = 0

# Event Detection State (appear / move / vanish)
committed_zone = {}
zone_votes = defaultdict(list)
previously_visible_names = set()

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

OBJECT_REAL_SIZES_CM = {
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
def speak(message):
    subprocess.run(["killall", "say"], stderr=subprocess.DEVNULL)
    print(f"\n🔊 SENSE: '{message}'")
    subprocess.Popen(["say", "-r", "190", message])

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
        memory_db.save_object_observation(clean_name.lower(), zone, cx, cy, status="VISIBLE")

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
        track_locked_identities[target_tid] = clean_name.title()
        memory_db.save_custom_object_profile(clean_name, base_cat, sig)
        speak(f"Got it. I have permanently memorized your {clean_name}.")
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

    # Intent A: Name Learning ("This is Sneha", "This is my water bottle")
    if any(phrase in user_text for phrase in ["this is", "my name is", "remember as", "call this", "call it"]):
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

    # Intent B: Room Overview ("What do you see?")
    if any(phrase in user_text for phrase in ["what do you see", "room", "everything", "around me"]):
        active_items = []
        for name, data in live_memory.items():
            if current_time - data["last_seen"] <= 2.5:
                active_items.append(f"{name} {data['last_zone']}, {data['dist_phrase']}")

        if not active_items:
            speak("I do not see any familiar objects or people right now.")
        else:
            speak("Currently, I see " + " and ".join(active_items) + ".")
        return

    # Intent C: Specific Target Search ("Where is my water bottle?", "Where is red bottle?", "Where is Sneha?")
    found_target = None
    all_searchable = list(known_objects.keys()) + list(known_people.keys()) + list(ALLOWED_ITEMS.values())

    if "neha" in user_text or "sneha" in user_text:
        found_target = "Sneha"
    elif "mummy" in user_text or "mom" in user_text:
        found_target = "Mummy"
    else:
        for item in all_searchable:
            if item.lower() in user_text:
                found_target = item.title()
                break

    # Color-based fallback search (e.g., "Where is red bottle?")
    if not found_target:
        for live_name in live_memory.keys():
            if any(word in user_text for word in live_name.lower().split()):
                found_target = live_name
                break

    if not found_target:
        for word in user_text.split():
            clean_w = word.strip(".,!?")
            if clean_w in STOPWORDS:
                continue
            if memory_db.get_object_memory(clean_w) is not None:
                found_target = clean_w.title()
                break

    if not found_target:
        speak("I heard you, but I did not catch which item or person you asked about.")
        return

    # 1. Live Check
    live_matches = [name for name in live_memory if found_target.lower() in name.lower() and (current_time - live_memory[name]["last_seen"] <= 2.5)]
    if live_matches:
        announcements = []
        for m in live_matches:
            d = live_memory[m]
            announcements.append(f"{m} {d['last_zone']}, {d['dist_phrase']}")
        speak("I see " + " and ".join(announcements) + ".")
        return

    # 2. Database + 360-Degree World Compass Transformation
    db_record = memory_db.get_object_memory(found_target.lower())
    if db_record is not None:
        time_elapsed = current_time - db_record["last_seen"]
        time_str = format_time_ago(time_elapsed)
        original_zone = db_record["last_zone"]

        corrected_zone = calculate_world_relative_direction(original_zone, cumulative_camera_heading)

        if abs(cumulative_camera_heading) > 25:
            if abs(cumulative_camera_heading) > 135:
                turn_desc = "turned around"
            elif cumulative_camera_heading < 0:
                turn_desc = "turned left"
            else:
                turn_desc = "turned right"

            pronoun = "they" if (found_target.lower() in [p.lower() for p in known_people.keys()] or found_target.lower() in ["mummy", "sneha"]) else "it"
            speak(f"{found_target} was last seen {original_zone} {time_str}. But since you {turn_desc}, {pronoun} should now be {corrected_zone}.")
        else:
            speak(f"{found_target} was last seen {original_zone}, {time_str}.")
    else:
        speak(f"I have no record of {found_target} in my memory database.")

def listen_in_background():
    global is_listening
    is_listening = True
    speak("Listening.")
    time.sleep(0.7)
    
    try:
        with sr.Microphone() as source:
            print("\n🎙️ Listening now... (Say: 'This is my water bottle' or 'Where is the bottle?')")
            audio = recognizer.listen(source, timeout=4.5, phrase_time_limit=5.0)
            text = recognizer.recognize_whisper(
                audio, 
                model="tiny", 
                initial_prompt="Sneha, Mummy, water bottle, medicine bottle, phone, laptop."
            )
            print(f"👂 Heard: '{text.strip()}'")
            answer_voice_query(text.strip())
    except Exception as e:
        print(f"Ear note: {e}")
        speak("Sorry, I had trouble hearing that.")
    finally:
        is_listening = False

def trigger_voice():
    global is_listening
    if not is_listening:
        threading.Thread(target=listen_in_background, daemon=True).start()

# Startup Greeting
saved_items = memory_db.get_all_remembered_items()
print(f"💾 Restored {len(saved_items)} objects and {len(known_people)} people.")
speak(f"Sense Master Online. Restored {len(saved_items)} objects and {len(known_people)} people.")

# =====================================================================
# 5. MAIN REAL-TIME MULTIMODAL LOOP
# =====================================================================
while True:
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
    currently_visible_names = set()

    if boxes is not None and boxes.id is not None:
        for box in boxes:
            raw_name = model.names[int(box.cls[0])]
            track_id = int(box.id[0])
            conf = float(box.conf[0])

            if raw_name == "person":
                entity_name = "person"
            elif raw_name in ALLOWED_ITEMS:
                entity_name = ALLOWED_ITEMS[raw_name]
            else:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

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

        # Instant lock-on within 2 frames!
        if len(track_votes[tid]) >= 2:
            winner_name, _ = Counter(track_votes[tid]).most_common(1)[0]
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
                        color_prefix = get_dominant_color_name(crop)
                        if color_prefix:
                            display_name = f"{color_prefix.capitalize()} {winner_name.capitalize()}"
                        else:
                            display_name = winner_name.capitalize()

            dist_phrase = format_distance(dist_feet)

            # Update Live Memory & SQLite (+ Event Detection: appear / move)
            if camera_is_stable:
                is_trackable = display_name not in ["Unfamiliar Person", "Person"]
                event_status = "VISIBLE"

                if is_trackable:
                    currently_visible_names.add(display_name)
                    was_away = (display_name not in live_memory) or \
                               (current_time - live_memory[display_name]["last_seen"] > 2.5)

                    if display_name not in committed_zone:
                        committed_zone[display_name] = zone
                        zone_votes[display_name] = []
                        print(f"🆕 NEW: '{display_name}' first seen {zone}.")
                    elif was_away:
                        committed_zone[display_name] = zone
                        zone_votes[display_name] = []
                        print(f"👋 BACK: '{display_name}' reappeared {zone}.")
                    else:
                        # Require 4/5 consistent frames before trusting a zone change,
                        # so a box jittering near a left/center/right boundary doesn't
                        # register as a false "moved" event.
                        zone_votes[display_name].append(zone)
                        if len(zone_votes[display_name]) > 5:
                            zone_votes[display_name].pop(0)
                        majority_zone, vote_count = Counter(zone_votes[display_name]).most_common(1)[0]
                        if majority_zone != committed_zone[display_name] and vote_count >= 4:
                            print(f"↔️ MOVED: '{display_name}' shifted from {committed_zone[display_name]} to {majority_zone}.")
                            committed_zone[display_name] = majority_zone
                            event_status = "MOVED"

                live_memory[display_name] = {
                    "last_zone": zone,
                    "last_seen": current_time,
                    "dist_feet": dist_feet,
                    "dist_phrase": dist_phrase,
                    "last_x": center_x,
                    "last_y": center_y
                }

                if is_trackable:
                    memory_db.save_object_observation(
                        display_name.lower(),
                        zone,
                        center_x,
                        center_y,
                        status=event_status
                    )

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
    # SUB-SYSTEM D.1: VANISH DETECTION
    # Only runs on stable frames: if the camera itself is panning,
    # "not detected this frame" just means "out of view right now",
    # not "actually gone" -- comparing two stable snapshots avoids
    # that false positive. Note: panning away and back will still
    # register a GONE event if the item isn't in the new field of
    # view, since a single camera can't tell "removed" from
    # "outside current view" -- a known limitation for now.
    # -------------------------------------------------------------
    if camera_is_stable:
        vanished_names = previously_visible_names - currently_visible_names
        for name in vanished_names:
            if name in live_memory:
                last = live_memory[name]
                print(f"📤 GONE: '{name}' is no longer visible (last seen {last['last_zone']}).")
                memory_db.save_object_observation(
                    name.lower(),
                    last["last_zone"],
                    last.get("last_x", 0),
                    last.get("last_y", 0),
                    status="MISSING"
                )
        previously_visible_names = currently_visible_names

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

camera.release()
cv2.destroyAllWindows()