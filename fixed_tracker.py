import cv2
import time
from ultralytics import YOLO

# Use yolov8s (or yolov8n if you want maximum speed)
model = YOLO("yolov8s.pt")
camera = cv2.VideoCapture(0)

# Memory store: remembers objects and their stable IDs
# Key: stable_id -> Value: dict of object properties
memory = {}

def calculate_iou(boxA, boxB):
    """
    Measures how much two boxes overlap (from 0.0 to 1.0).
    box = (x1, y1, x2, y2)
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

print("Fixed SENSE Tracker running. Press 'q' to stop.")

while True:
    success, frame = camera.read()
    if not success:
        break

    current_time = time.time()

    # Track with YOLO's native ByteTrack
    results = model.track(
        frame, 
        persist=True, 
        conf=0.25, 
        tracker="bytetrack.yaml", 
        verbose=False
    )
    
    boxes = results[0].boxes

    if boxes is not None and boxes.id is not None:
        for box in boxes:
            # 1. Get raw info
            yolo_id = int(box.id[0])
            conf = float(box.conf[0])
            class_id = int(box.cls[0])
            name = model.names[class_id]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            current_box = (x1, y1, x2, y2)
            center = (int((x1 + x2) / 2), int((y1 + y2) / 2))

            # 2. Update memory for this object
            memory[yolo_id] = {
                "name": name,
                "box": current_box,
                "center": center,
                "last_seen": current_time,
                "conf": conf
            }

            # 3. Draw clean boxes
            # Green for Person, Blue for Objects
            color = (0, 255, 0) if name == "person" else (255, 180, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.circle(frame, center, 4, (0, 0, 255), -1)

            label = f"ID #{yolo_id}: {name} ({int(conf * 100)}%)"
            cv2.putText(
                frame, 
                label, 
                (x1, max(25, y1 - 8)), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, 
                color, 
                2
            )

    # Clean out items from memory that haven't been seen in 10 seconds
    dead_ids = [k for k, v in memory.items() if current_time - v["last_seen"] > 10.0]
    for k in dead_ids:
        del memory[k]

    # Show live memory count
    cv2.putText(
        frame, 
        f"Active Memory Items: {len(memory)}", 
        (20, 35), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        0.7, 
        (0, 255, 255), 
        2
    )

    cv2.imshow("Fixed SENSE Tracker", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()