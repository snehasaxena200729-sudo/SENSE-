import cv2
import time
import subprocess
from ultralytics import YOLO

model = YOLO("yolov8s.pt")
camera = cv2.VideoCapture(0)

# Memory store
object_memory = {}
last_query_time = 0

def speak(message):
    """Cuts off any previous talking and speaks the new message cleanly!"""
    subprocess.run(["killall", "say"], stderr=subprocess.DEVNULL)
    print(f"🔊 SENSE: '{message}'")
    subprocess.Popen(["say", "-r", "190", message])

def get_spatial_zone(x_coordinate, frame_width=640):
    third = frame_width / 3.0
    if x_coordinate < third:
        return "on your left"
    elif x_coordinate < (third * 2):
        return "in front of you"
    else:
        return "on your right"

def query_environmental_memory():
    """Reports ONLY active, unique items currently in the room!"""
    current_time = time.time()
    
    # 1. Filter out old ghosts: Only keep items seen in the last 3 seconds
    active_items = {}
    for track_id, item in object_memory.items():
        if current_time - item["last_seen"] < 3.0:
            name = item["name"]
            # Deduplicate: Only keep ONE entry per item name
            active_items[name] = item["current_zone"]

    if not active_items:
        speak("I do not see any objects right now.")
        return

    # 2. Build ONE short, polite sentence
    # Example: "I see a cell phone in front of you, and a cup on your right."
    descriptions = [f"a {name} {zone}" for name, zone in active_items.items()]
    
    if len(descriptions) == 1:
        sentence = f"I see {descriptions[0]}."
    else:
        sentence = "I see " + ", and ".join(descriptions) + "."

    speak(sentence)

print("------------------------------------------------------------------")
print("Fixed SENSE Spatial Memory Online!")
print("👉 Press SPACEBAR once for a short, clean summary.")
print("👉 Press 'q' to quit.")
print("------------------------------------------------------------------")

speak("System online.")

while True:
    success, frame = camera.read()
    if not success:
        break

    current_time = time.time()
    frame_height, frame_width, _ = frame.shape
    
    results = model.track(frame, persist=True, conf=0.30, tracker="bytetrack.yaml", verbose=False)
    boxes = results[0].boxes

    # Draw visual zones
    cv2.line(frame, (int(frame_width/3), 0), (int(frame_width/3), frame_height), (80, 80, 80), 1)
    cv2.line(frame, (int(frame_width*2/3), 0), (int(frame_width*2/3), frame_height), (80, 80, 80), 1)

    if boxes is not None and boxes.id is not None:
        for box in boxes:
            track_id = int(box.id[0])
            name = model.names[int(box.cls[0])]
            
            # Skip person
            if name == "person":
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            current_zone = get_spatial_zone(center[0], frame_width)

            # Update memory
            object_memory[track_id] = {
                "name": name,
                "current_zone": current_zone,
                "last_seen": current_time
            }

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 120), 2)
            cv2.putText(frame, f"{name} ({current_zone})", (x1, max(25, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 120), 2)

    cv2.imshow("Fixed SENSE", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        subprocess.run(["killall", "say"], stderr=subprocess.DEVNULL)
        break
    elif key == ord(' '):  # SPACEBAR
        # 1-second debounce so holding the key doesn't trigger 10 times
        if current_time - last_query_time > 1.5:
            query_environmental_memory()
            last_query_time = current_time

camera.release()
cv2.destroyAllWindows()