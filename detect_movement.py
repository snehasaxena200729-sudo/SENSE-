import cv2
import time
import math
from ultralytics import YOLO

model = YOLO("yolov8s.pt")
camera = cv2.VideoCapture(0)

# -------------------------------------------------------------
# ENVIRONMENTAL MEMORY STORE
# Stores the "Home/Initial" position and current state of each object
# -------------------------------------------------------------
object_history = {}

def get_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

print("SENSE Movement Engine Running! Place an object down, then move it.")

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

            # ---------------------------------------------------------
            # 1. If this is a new object, remember its STARTING POSITION
            # ---------------------------------------------------------
            if track_id not in object_history:
                object_history[track_id] = {
                    "name": name,
                    "start_pos": center,      # Where it was first seen
                    "current_pos": center,    # Where it is right now
                    "has_moved": False
                }
                print(f"📦 Locked initial location for {name} (ID: {track_id}) at {center}")
            else:
                # Update current position
                start_pos = object_history[track_id]["start_pos"]
                object_history[track_id]["current_pos"] = center

                # ---------------------------------------------------------
                # 2. Check: Has it moved far from where it started?
                # ---------------------------------------------------------
                distance_moved = get_distance(start_pos, center)

                # If it moved more than 80 pixels, it's a real movement!
                if distance_moved > 80:
                    if not object_history[track_id]["has_moved"]:
                        object_history[track_id]["has_moved"] = True
                        print(f"🚨 MOVEMENT ALERT: {name} (ID: {track_id}) was MOVED! Shifted by {int(distance_moved)} pixels!")

                # ---------------------------------------------------------
                # 3. Draw a "Breadcrumb Line" from START to CURRENT!
                # ---------------------------------------------------------
                if object_history[track_id]["has_moved"]:
                    # Draw a red dashed-style line showing the path it traveled
                    cv2.line(frame, start_pos, center, (0, 0, 255), 2)
                    # Draw a yellow circle at its ORIGINAL starting spot (the "ghost" position)
                    cv2.circle(frame, start_pos, 8, (0, 255, 255), -1)
                    cv2.putText(frame, "Original Spot", (start_pos[0] + 10, start_pos[1]),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

            # Draw current bounding box
            color = (0, 0, 255) if object_history[track_id]["has_moved"] else (0, 255, 0)
            status_text = "MOVED" if object_history[track_id]["has_moved"] else "STATIONARY"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame, 
                f"ID #{track_id} {name} [{status_text}]", 
                (x1, max(25, y1 - 8)), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.55, 
                color, 
                2
            )

    cv2.imshow("SENSE Movement Engine", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()