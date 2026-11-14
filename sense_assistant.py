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

# 1. Initialize Database & Vision Brain
memory_db.init_database()
print("Starting SENSE Master Vision Brain...")
model = YOLO("yolov8s.pt")
camera = cv2.VideoCapture(0)

# 2. Configure Patient Offline Ear (Whisper)
recognizer = sr.Recognizer()
recognizer.pause_threshold = 1.3
recognizer.energy_threshold = 300

# 3. Optical Flow Parameters for Camera Ego-Motion
feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=15, blockSize=7)
lk_params = dict(winSize=(15, 15), maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

ret, old_frame = camera.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

track_votes = defaultdict(list)
live_memory = {}
is_listening = False
last_camera_move_time = 0

ALLOWED_ITEMS = {
    "cell phone": "phone",
    "remote": "remote",
    "cup": "cup",
    "bottle": "bottle",
    "book": "book",
    "laptop": "laptop",
    "mouse": "mouse"
}

def speak(message):
    subprocess.run(["killall", "say"], stderr=subprocess.DEVNULL)
    print(f"\n🔊 SENSE: '{message}'")
    subprocess.Popen(["say", "-r", "190", message])

def get_spatial_zone(x_coordinate, frame_width=640):
    third = frame_width / 3.0
    if x_coordinate < third:
        return "on your left"
    elif x_coordinate < (third * 2):
        return "in front of you"
    else:
        return "on your right"

def format_time_ago(seconds):
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds} seconds ago"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes} minute{'s' if minutes > 1 else ''} ago"
    hours = minutes // 60
    return f"{hours} hour{'s' if hours > 1 else ''} ago"

def answer_voice_query(user_text):
    user_text = user_text.lower().strip()
    print(f"👂 Transcribed: '{user_text}'")
    current_time = time.time()

    if any(phrase in user_text for phrase in ["what do you see", "room", "everything", "around me", "where are my things"]):
        active_now = [f"a {name} {data['last_zone']}" for name, data in live_memory.items() 
                      if current_time - data["last_seen"] <= 2.5]
        if not active_now:
            speak("I do not see any familiar objects right now.")
        else:
            speak("Currently, I see " + " and ".join(active_now) + ".")
        return

    found_target = None
    for item in ALLOWED_ITEMS.values():
        if item in user_text:
            found_target = item
            break

    if not found_target:
        speak("I heard you, but I did not catch which item you asked about.")
        return

    live_data = live_memory.get(found_target)
    if live_data and (current_time - live_data["last_seen"] <= 2.5):
        speak(f"Your {found_target} is {live_data['last_zone']}.")
        return

    db_record = memory_db.get_object_memory(found_target)
    if db_record is not None:
        time_elapsed = current_time - db_record["last_seen"]
        time_str = format_time_ago(time_elapsed)
        zone = db_record["last_zone"]
        speak(f"Your {found_target} is not visible right now. It was last seen {zone}, {time_str}.")
    else:
        speak(f"I have no record of your {found_target} in my memory database.")

def listen_in_background():
    global is_listening
    is_listening = True
    speak("Listening.")
    time.sleep(0.7)
    
    try:
        with sr.Microphone() as source:
            print("\n🎙️ Listening now...")
            audio = recognizer.listen(source, timeout=5.0, phrase_time_limit=6.0)
            print("🧠 Transcribing with Whisper...")
            text = recognizer.recognize_whisper(audio, model="tiny.en")
            if text and text.strip():
                answer_voice_query(text)
            else:
                speak("I didn't catch that.")
    except Exception as e:
        print(f"Ear note: {e}")
        speak("Sorry, I had trouble hearing that.")
    finally:
        is_listening = False

def trigger_voice():
    global is_listening
    if not is_listening:
        threading.Thread(target=listen_in_background, daemon=True).start()

# Startup announcement
saved_items = memory_db.get_all_remembered_items()
if saved_items:
    item_names = [row[0] for row in saved_items]
    print(f"💾 Restored knowledge from DB: {item_names}")
    speak(f"Sense ready. Restored {len(saved_items)} objects from database. Ego motion shield active.")
else:
    speak("Sense ready. Ego motion shield active.")

while True:
    success, frame = camera.read()
    if not success:
        break

    current_time = time.time()
    frame_height, frame_width, _ = frame.shape
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # -------------------------------------------------------------
    # 1. CAMERA EGO-MOTION TRACKER (Lucas-Kanade Optical Flow)
    # -------------------------------------------------------------
    camera_speed = 0.0
    if p0 is not None and len(p0) > 0:
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        if p1 is not None:
            good_new = p1[st == 1]
            good_old = p0[st == 1]
            if len(good_new) > 10:
                displacements = good_new - good_old
                camera_dx = np.median(displacements[:, 0])
                camera_dy = np.median(displacements[:, 1])
                camera_speed = (camera_dx**2 + camera_dy**2)**0.5
                p0 = good_new.reshape(-1, 1, 2)
            else:
                p0 = cv2.goodFeaturesToTrack(frame_gray, mask=None, **feature_params)
        else:
            p0 = cv2.goodFeaturesToTrack(frame_gray, mask=None, **feature_params)
    else:
        p0 = cv2.goodFeaturesToTrack(frame_gray, mask=None, **feature_params)

    old_gray = frame_gray.copy()

    # If camera speed > 3.0 px/frame, the user is panning/moving the laptop!
    is_camera_moving = camera_speed > 3.0
    if is_camera_moving:
        last_camera_move_time = current_time

    # Grace period: Wait 0.8s after camera stops moving before trusting stationary positions
    camera_is_stable = (current_time - last_camera_move_time) > 0.8

    # -------------------------------------------------------------
    # 2. OBJECT DETECTION & TRACKING
    # -------------------------------------------------------------
    results = model.track(frame, persist=True, conf=0.30, tracker="bytetrack.yaml", verbose=False)
    boxes = results[0].boxes

    candidates = []

    if boxes is not None and boxes.id is not None:
        for box in boxes:
            raw_name = model.names[int(box.cls[0])]
            if raw_name not in ALLOWED_ITEMS:
                continue

            track_id = int(box.id[0])
            conf = float(box.conf[0])
            friendly_name = ALLOWED_ITEMS[raw_name]

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            candidates.append({
                "track_id": track_id,
                "name": friendly_name,
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

    # Majority Voting & Memory Updates (Only update position when camera is STABLE!)
    for cand in filtered_candidates:
        tid = cand["track_id"]
        track_votes[tid].append(cand["name"])
        if len(track_votes[tid]) > 15:
            track_votes[tid].pop(0)

        if len(track_votes[tid]) >= 6:
            winner_name, _ = Counter(track_votes[tid]).most_common(1)[0]
            x1, y1, x2, y2 = cand["box"]
            center_x, center_y = cand["center"]
            zone = get_spatial_zone(center_x, frame_width)

            # EGO-MOTION SHIELD IN ACTION:
            # Only record/update the object's official zone if the camera is stable!
            if camera_is_stable:
                live_memory[winner_name] = {
                    "last_zone": zone,
                    "last_seen": current_time
                }
                memory_db.save_object_observation(
                    winner_name, 
                    zone, 
                    center_x, 
                    center_y, 
                    status="VISIBLE"
                )

            box_color = (0, 255, 120) if camera_is_stable else (0, 165, 255)
            status_tag = f"({zone})" if camera_is_stable else "(CAM MOVING)"
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            cv2.putText(
                frame, 
                f"{winner_name} {status_tag}", 
                (x1, max(25, y1 - 8)), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.55, 
                box_color, 
                2
            )

    # Visual HUD status
    if is_camera_moving:
        cv2.putText(frame, "CAM MOVING: MOVEMENT ALERTS PAUSED", (20, 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
    elif is_listening:
        cv2.putText(frame, "STATUS: LISTENING... SPEAK NOW!", (20, 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    else:
        cv2.putText(frame, "SENSE ACTIVE (TAP SPACEBAR TO TALK)", (20, 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("SENSE Master System", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        subprocess.run(["killall", "say"], stderr=subprocess.DEVNULL)
        break
    elif key == ord(' '):
        trigger_voice()

camera.release()
cv2.destroyAllWindows()