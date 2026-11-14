import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import cv2
import time
import subprocess
import threading
import speech_recognition as sr
from ultralytics import YOLO

# 1. Load YOLO (runs at 30+ FPS)
print("Loading SENSE High-Speed Engine...")
model = YOLO("yolov8s.pt")
camera = cv2.VideoCapture(0)

recognizer = sr.Recognizer()
recognizer.pause_threshold = 1.2

# Person Memory: Name -> Visual Color/Appearance Profile
known_people = {}
current_person_label = "Unfamiliar Person"
latest_person_crop = None
is_listening = False

def speak(message):
    subprocess.run(["killall", "say"], stderr=subprocess.DEVNULL)
    print(f"\n🔊 SENSE: '{message}'")
    subprocess.Popen(["say", "-r", "190", message])

def get_person_signature(crop):
    """Computes a high-speed 2D color signature of the person's upper body."""
    h, w, _ = crop.shape
    upper = crop[0:int(h * 0.6), :]
    resized = cv2.resize(upper, (60, 80))
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [12, 16], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist

def match_person(sig, threshold=0.55):
    """Matches signature against known people in memory."""
    if not known_people:
        return "Unfamiliar Person"
    
    best_match = "Unfamiliar Person"
    best_score = -1.0
    for name, saved_sig in known_people.items():
        score = cv2.compareHist(sig, saved_sig, cv2.HISTCMP_CORREL)
        if score > best_score:
            best_score = score
            if score > threshold:
                best_match = name
    return best_match

def learn_person_by_voice(spoken_text):
    """Parses voice to learn the person's name (e.g. 'This is Sneha' or 'Sneha')."""
    global latest_person_crop
    words = spoken_text.split()
    
    # Extract the name (the last word spoken)
    if words:
        name = words[-1].strip(".,!?").capitalize()
        if latest_person_crop is not None and latest_person_crop.size > 0:
            sig = get_person_signature(latest_person_crop)
            known_people[name] = sig
            speak(f"Got it. I have memorized {name}.")
            print(f"✨ Successfully learned profile for: {name}!")
        else:
            speak("I didn't see anyone clearly in front of the camera.")

def listen_in_background():
    """Uses the ALREADY DOWNLOADED 'tiny' model with name hints - 0ms download wait!"""
    global is_listening
    is_listening = True
    speak("Who is this?")
    time.sleep(0.7)
    
    try:
        with sr.Microphone() as source:
            print("\n🎙️ Listening now... Say your name clearly (e.g. 'This is Sneha'):")
            audio = recognizer.listen(source, timeout=4.0, phrase_time_limit=4.0)
            
            # Uses 'tiny' (already cached on your laptop!) + prompt hint
            text = recognizer.recognize_whisper(
                audio, 
                model="tiny", 
                initial_prompt="The user's name is Sneha, Priya, Sarah, Alex, John."
            )
            print(f"👂 Heard: '{text.strip()}'")
            learn_person_by_voice(text.strip())
    except Exception as e:
        print(f"Ear note: {e}")
        speak("I didn't catch a name.")
    finally:
        is_listening = False

def trigger_naming():
    global is_listening
    if not is_listening:
        threading.Thread(target=listen_in_background, daemon=True).start()

print("------------------------------------------------------------------")
print("SENSE High-Speed Person Recognition Online (Zero Download Wait)!")
print("👉 Stand in front of camera: SENSE will track you with 0% lag.")
print("👉 Tap SPACEBAR and say: 'This is Sneha'.")
print("👉 Step away and return: SENSE will recognize you by name!")
print("👉 Press 'q' to quit.")
print("------------------------------------------------------------------")

speak("Person identity engine ready.")

while True:
    success, frame = camera.read()
    if not success:
        break

    results = model.track(frame, persist=True, conf=0.30, tracker="bytetrack.yaml", verbose=False)
    boxes = results[0].boxes

    person_found = False

    if boxes is not None and boxes.id is not None:
        for box in boxes:
            if model.names[int(box.cls[0])] == "person":
                person_found = True
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

                crop = frame[y1:y2, x1:x2]
                if crop.size > 0:
                    latest_person_crop = crop
                    sig = get_person_signature(crop)
                    current_person_label = match_person(sig)

                    # Distance formula
                    h_px = y2 - y1
                    w_px = x2 - x1
                    real_size = 165.0 if (h_px / max(1, w_px)) > 2.0 else 75.0
                    dist_feet = (real_size * 600.0) / (h_px * 30.48)

                    # Colors: Green if recognized, Orange if unfamiliar
                    color = (0, 255, 120) if current_person_label != "Unfamiliar Person" else (0, 165, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    label_text = f"{current_person_label} ({dist_feet:.1f} ft)"
                    cv2.putText(frame, label_text, (x1, max(25, y1 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    if not person_found:
        current_person_label = "No Person in View"

    # Status HUD
    if is_listening:
        cv2.putText(frame, "STATUS: LISTENING FOR NAME...", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
    else:
        cv2.putText(frame, "TAP SPACEBAR TO NAME PERSON | 'q' TO QUIT", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    cv2.imshow("SENSE Smooth Person Recognition", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        subprocess.run(["killall", "say"], stderr=subprocess.DEVNULL)
        break
    elif key == ord(' '):
        trigger_naming()

camera.release()
cv2.destroyAllWindows()