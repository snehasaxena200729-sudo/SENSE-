import cv2
import time
import subprocess
from ultralytics import YOLO

model = YOLO("yolov8s.pt")
camera = cv2.VideoCapture(0)

# -------------------------------------------------------------
# SENSE LONG-TERM MEMORY STORE
# Remembers both active items and items that vanished!
# -------------------------------------------------------------
memory_store = {}
last_query_time = 0

def speak(message):
    """Speaks out loud cleanly."""
    subprocess.run(["killall", "say"], stderr=subprocess.DEVNULL)
    print(f"\n🔊 SENSE: '{message}'")
    subprocess.Popen(["say", "-r", "185", message])

def get_spatial_zone(x_coordinate, frame_width=640):
    third = frame_width / 3.0
    if x_coordinate < third:
        return "on your left"
    elif x_coordinate < (third * 2):
        return "in front of you"
    else:
        return "on your right"

def format_time_ago(seconds):
    """Converts seconds into natural human speech."""
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds} seconds ago"
    else:
        minutes = seconds // 60
        return f"{minutes} minute{'s' if minutes > 1 else ''} ago"

def query_memory():
    """Generates a complete status report of active and recently lost items!"""
    current_time = time.time()
    
    if not memory_store:
        speak("I have not seen any objects yet today.")
        return

    visible_reports = []
    lost_reports = []

    # Group by object name to keep things clean and deduplicated
    for name, data in memory_store.items():
        time_elapsed = current_time - data["last_seen"]

        if time_elapsed <= 2.5:
            # It is currently visible!
            visible_reports.append(f"your {name} is {data['last_zone']}")
        else:
            # It was seen in the past, but is gone now!
            # Only remember things seen in the last 10 minutes
            if time_elapsed < 600:
                time_str = format_time_ago(time_elapsed)
                lost_reports.append(f"your {name} was last seen {data['last_zone']} {time_str}")

    # Build the natural speech sentence
    final_speech = []
    
    if visible_reports:
        final_speech.append("Currently, " + ", and ".join(visible_reports) + ".")
    
    if lost_reports:
        final_speech.append("Also, " + ", and ".join(lost_reports) + ".")

    if not final_speech:
        speak("I do not see any familiar objects right now.")
    else:
        speak(" ".join(final_speech))

print("------------------------------------------------------------------")
print("SENSE Temporal Memory Engine Online!")
print("👉 1. Show an object (like your phone or cup) so SENSE learns it.")
print("👉 2. Hide the object behind your back or put it in your pocket.")
print("👉 3. Press SPACEBAR and listen to SENSE remember where it WAS!")
print("👉 Press 'q' to quit.")
print("------------------------------------------------------------------")

speak("Temporal memory online.")

while True:
    success, frame = camera.read()
    if not success:
        break

    current_time = time.time()
    frame_height, frame_width, _ = frame.shape
    
    results = model.track(frame, persist=True, conf=0.30, tracker="bytetrack.yaml", verbose=False)
    boxes = results[0].boxes

    # Visual guide lines
    cv2.line(frame, (int(frame_width/3), 0), (int(frame_width/3), frame_height), (60, 60, 60), 1)
    cv2.line(frame, (int(frame_width*2/3), 0), (int(frame_width*2/3), frame_height), (60, 60, 60), 1)

    if boxes is not None and boxes.id is not None:
        for box in boxes:
            name = model.names[int(box.cls[0])]
            
            # Skip person
            if name == "person":
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            center_x = int((x1 + x2) / 2)
            zone = get_spatial_zone(center_x, frame_width)

            # Store in our permanent memory notebook under the object's name!
            memory_store[name] = {
                "last_zone": zone,
                "last_seen": current_time
            }

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 120), 2)
            cv2.putText(frame, f"{name} ({zone})", (x1, max(25, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 120), 2)

    cv2.imshow("SENSE Temporal Memory", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        subprocess.run(["killall", "say"], stderr=subprocess.DEVNULL)
        break
    elif key == ord(' '):
        if current_time - last_query_time > 1.5:
            query_memory()
            last_query_time = current_time

camera.release()
cv2.destroyAllWindows()