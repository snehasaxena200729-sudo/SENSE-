import cv2
import time
import subprocess
import threading
import speech_recognition as sr
from collections import defaultdict
from ultralytics import YOLO

# 1. Initialize Vision Model and Camera
model = YOLO("yolov8s.pt")
camera = cv2.VideoCapture(0)

detection_history = defaultdict(int)
confirmed_memory = {}
is_listening = False

ALLOWED_ITEMS = {
    "cell phone": "phone",
    "remote": "remote",
    "cup": "cup",
    "bottle": "bottle",
    "book": "book",
    "laptop": "laptop",
    "mouse": "mouse"
}

recognizer = sr.Recognizer()

def speak(message):
    """Speaks out loud without freezing the video."""
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
    return f"{minutes} minute{'s' if minutes > 1 else ''} ago"

def answer_voice_query(user_text):
    """Parses what you said and queries the memory!"""
    user_text = user_text.lower()
    print(f"👂 Heard question: '{user_text}'")

    current_time = time.time()

    # Did they ask a general question?
    if any(phrase in user_text for phrase in ["what do you see", "room", "everything", "around me", "where are my things"]):
        active_now = [f"a {name} {data['last_zone']}" for name, data in confirmed_memory.items() 
                      if current_time - data["last_seen"] <= 2.5]
        if not active_now:
            speak("I do not see any familiar objects right now.")
        else:
            speak("Currently, I see " + " and ".join(active_now) + ".")
        return

    # Check for specific item keywords
    found_target = None
    for item in ALLOWED_ITEMS.values():
        if item in user_text:
            found_target = item
            break

    if not found_target:
        speak("I heard you, but I didn't recognize which item you are asking about.")
        return

    # Look up in memory
    target_data = confirmed_memory.get(found_target)
    if target_data is None:
        speak(f"I have not seen your {found_target}.")
        return

    time_elapsed = current_time - target_data["last_seen"]
    zone = target_data["last_zone"]

    if time_elapsed <= 2.5:
        speak(f"Your {found_target} is {zone}.")
    else:
        time_str = format_time_ago(time_elapsed)
        speak(f"Your {found_target} is not visible right now. It was last seen {zone}, {time_str}.")

def listen_in_background():
    """Listens to the microphone in a separate thread so video does not freeze!"""
    global is_listening
    is_listening = True
    speak("Listening.")
    
    try:
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio = recognizer.listen(source, timeout=4.0, phrase_time_limit=4.0)
            text = recognizer.recognize_google(audio)
            answer_voice_query(text)
    except sr.WaitTimeoutError:
        speak("I didn't hear anything.")
    except sr.UnknownValueError:
        speak("Sorry, I could not understand that.")
    except Exception as e:
        print(f"Mic note: {e}")
        speak("Could not access microphone.")
    finally:
        is_listening = False

def trigger_voice_listen():
    global is_listening
    if not is_listening:
        thread = threading.Thread(target=listen_in_background, daemon=True)
        thread.start()

print("------------------------------------------------------------------")
print("SENSE Voice Interaction Engine Online!")
print("👉 Press SPACEBAR, wait for 'Listening', then ask:")
print("   - 'Where is my phone?'")
print("   - 'Where is my remote?'")
print("   - 'What do you see?'")
print("👉 Press 'q' to Quit")
print("------------------------------------------------------------------")

speak("Voice engine ready.")

while True:
    success, frame = camera.read()
    if not success:
        break

    current_time = time.time()
    frame_height, frame_width, _ = frame.shape
    
    results = model.track(frame, persist=True, conf=0.30, tracker="bytetrack.yaml", verbose=False)
    boxes = results[0].boxes

    current_seen_names = set()

    if boxes is not None and boxes.id is not None:
        for box in boxes:
            raw_name = model.names[int(box.cls[0])]

            if raw_name not in ALLOWED_ITEMS:
                continue

            friendly_name = ALLOWED_ITEMS[raw_name]
            current_seen_names.add(friendly_name)
            detection_history[friendly_name] += 1

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            center_x = int((x1 + x2) / 2)
            zone = get_spatial_zone(center_x, frame_width)

            # 10-Frame confirmation rule
            if detection_history[friendly_name] >= 10:
                confirmed_memory[friendly_name] = {
                    "last_zone": zone,
                    "last_seen": current_time
                }

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 120), 2)
                cv2.putText(
                    frame, 
                    f"{friendly_name} ({zone})", 
                    (x1, max(25, y1 - 8)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.55, 
                    (0, 255, 120), 
                    2
                )

    # Decay candidate scores
    for item in list(detection_history.keys()):
        if item not in current_seen_names:
            detection_history[item] = max(0, detection_history[item] - 1)

    # Visual status bar on frame
    if is_listening:
        cv2.putText(frame, "STATUS: LISTENING TO YOU...", (20, 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        cv2.putText(frame, "PRESS SPACEBAR TO TALK", (20, 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    cv2.imshow("SENSE Voice Interaction", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        subprocess.run(["killall", "say"], stderr=subprocess.DEVNULL)
        break
    elif key == ord(' '):
        trigger_voice_listen()

camera.release()
cv2.destroyAllWindows()