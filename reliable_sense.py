import cv2
import time
import subprocess
from collections import defaultdict
from ultralytics import YOLO

model = YOLO("yolov8s.pt")
camera = cv2.VideoCapture(0)

# -------------------------------------------------------------
# 1. CANDIDATE BUFFER: Items must prove they are real!
# 2. CONFIRMED MEMORY: Only trusted items are remembered.
# -------------------------------------------------------------
detection_history = defaultdict(int)
confirmed_memory = {}
last_query_time = 0

# Whitelist: Only look for everyday items (blocks ghost knives/toothbrushes!)
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
    """Speaks out loud without freezing the video stream."""
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

def query_specific_item(item_keyword):
    """Answers: 'Where is my [item]?' cleanly and accurately!"""
    current_time = time.time()
    
    target_data = None
    matched_name = item_keyword
    for name, data in confirmed_memory.items():
        if item_keyword in name:
            target_data = data
            matched_name = name
            break

    if target_data is None:
        speak(f"I have not seen your {item_keyword}.")
        return

    time_elapsed = current_time - target_data["last_seen"]
    zone = target_data["last_zone"]

    # If seen within the last 2.5 seconds, it's currently visible
    if time_elapsed <= 2.5:
        speak(f"Your {matched_name} is {zone}.")
    else:
        # It disappeared in the past
        time_str = format_time_ago(time_elapsed)
        speak(f"Your {matched_name} is not visible right now. It was last seen {zone}, {time_str}.")

def query_room_overview():
    """Short summary of ONLY items that are visible RIGHT NOW."""
    current_time = time.time()
    active_now = []

    for name, data in confirmed_memory.items():
        if current_time - data["last_seen"] <= 2.5:
            active_now.append(f"a {name} {data['last_zone']}")

    if not active_now:
        speak("I do not see any familiar objects right now.")
    else:
        speak("Currently, I see " + " and ".join(active_now) + ".")

print("------------------------------------------------------------------")
print("SENSE Reliable Memory Online (Ghost-Shield Active)!")
print("👉 Press 'P' -> Where is my Phone?")
print("👉 Press 'R' -> Where is my Remote?")
print("👉 Press 'C' -> Where is my Cup / Bottle?")
print("👉 Press SPACEBAR -> Quick room overview")
print("👉 Press 'q' -> Quit")
print("------------------------------------------------------------------")

speak("Sense ready.")

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

            # Whitelist filter: Immediately discards knives, toothbrushes, etc.
            if raw_name not in ALLOWED_ITEMS:
                continue

            friendly_name = ALLOWED_ITEMS[raw_name]
            current_seen_names.add(friendly_name)

            # Increase stability score
            detection_history[friendly_name] += 1

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            center_x = int((x1 + x2) / 2)
            zone = get_spatial_zone(center_x, frame_width)

            # ---------------------------------------------------------
            # THE 10-FRAME CONFIRMATION SHIELD:
            # Must be seen for 10 frames in a row before SENSE trusts it!
            # ---------------------------------------------------------
            if detection_history[friendly_name] >= 10:
                confirmed_memory[friendly_name] = {
                    "last_zone": zone,
                    "last_seen": current_time
                }

                # Draw green box for confirmed items
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

    # Decay candidate scores for items not seen in this frame
    for item in list(detection_history.keys()):
        if item not in current_seen_names:
            detection_history[item] = max(0, detection_history[item] - 1)

    cv2.imshow("SENSE Reliable Memory", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        subprocess.run(["killall", "say"], stderr=subprocess.DEVNULL)
        break
    elif key in (ord('p'), ord('P')):
        query_specific_item("phone")
    elif key in (ord('r'), ord('R')):
        query_specific_item("remote")
    elif key in (ord('c'), ord('C')):
        query_specific_item("cup")
    elif key == ord(' '):
        if current_time - last_query_time > 1.5:
            query_room_overview()
            last_query_time = current_time

camera.release()
cv2.destroyAllWindows()