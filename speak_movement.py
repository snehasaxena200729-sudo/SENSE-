import cv2
import time
import math
import subprocess
from ultralytics import YOLO

model = YOLO("yolov8s.pt")
camera = cv2.VideoCapture(0)

# Memory store
object_history = {}

# Anti-Chatter: Keeps track of when we last spoke so we don't spam
last_spoken_time = 0
SPEECH_COOLDOWN = 4.0  # Wait at least 4 seconds before speaking again

def speak(message):
    """Speaks text out loud in the background without freezing the video!"""
    global last_spoken_time
    current_time = time.time()
    
    # If we spoke very recently, stay quiet!
    if current_time - last_spoken_time > SPEECH_COOLDOWN:
        print(f"🔊 SENSE SAYS: '{message}'")
        # Uses your Mac's native speech synthesizer asynchronously
        subprocess.Popen(["say", "-r", "190", message])
        last_spoken_time = current_time

def get_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

print("SENSE Voice Engine Online! Put an item down, then move it. Press 'q' to stop.")

# Welcome greeting
speak("Sense system online. Watching your surroundings.")

while True:
    success, frame = camera.read()
    if not success:
        break

    results = model.track(frame, persist=True, conf=0.30, tracker="bytetrack.yaml", verbose=False)
    boxes = results[0].boxes

    if boxes is not None and boxes.id is not None:
        for box in boxes:
            track_id = int(box.id[0])
            class_id = int(box.cls[0])
            name = model.names[class_id]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            center = (int((x1 + x2) / 2), int((y1 + y2) / 2))

            # 1. Lock starting position
            if track_id not in object_history:
                object_history[track_id] = {
                    "name": name,
                    "start_pos": center,
                    "current_pos": center,
                    "has_moved": False,
                    "announced": False
                }
            else:
                start_pos = object_history[track_id]["start_pos"]
                object_history[track_id]["current_pos"] = center
                distance_moved = get_distance(start_pos, center)

                # 2. Did it move?
                if distance_moved > 90:
                    object_history[track_id]["has_moved"] = True

                    # 3. Speak the alert ONCE!
                    if not object_history[track_id]["announced"]:
                        speak(f"Notice: your {name} was moved.")
                        object_history[track_id]["announced"] = True

                # Draw breadcrumb trail
                if object_history[track_id]["has_moved"]:
                    cv2.line(frame, start_pos, center, (0, 0, 255), 2)
                    cv2.circle(frame, start_pos, 8, (0, 255, 255), -1)

            # Draw box
            color = (0, 0, 255) if object_history[track_id]["has_moved"] else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame, 
                f"{name} #{track_id}", 
                (x1, max(25, y1 - 8)), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, 
                color, 
                2
            )

    cv2.imshow("SENSE Voice Engine", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()