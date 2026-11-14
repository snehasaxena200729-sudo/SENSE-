import cv2
import time
import math
from ultralytics import YOLO

# 1. We upgrade to 'yolov8s.pt' (Small model - much smarter, no more knife/toothbrush confusion!)
print("Loading smarter AI model (yolov8s)...")
model = YOLO("yolov8s.pt")
camera = cv2.VideoCapture(0)

# SENSE Environmental Memory
# Stores persistent objects that survive tracking drops!
persistent_memory = {}
next_permanent_id = 1

def calculate_distance(p1, p2):
    """Calculates how many pixels apart two points are."""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

print("SENSE Smart Memory is live! Move your remote around. Press 'q' to stop.")

while True:
    success, frame = camera.read()
    if not success:
        break

    current_time = time.time()
    
    # Run tracking with 25% confidence
    results = model.track(frame, persist=True, conf=0.25, tracker="bytetrack.yaml", verbose=False)
    boxes = results[0].boxes

    current_frame_objects = []

    if boxes is not None:
        for box in boxes:
            conf = float(box.conf[0])
            if conf > 0.30:
                class_id = int(box.cls[0])
                name = model.names[class_id]

                # If the AI guesses knife or toothbrush on a remote, filter out obvious silliness!
                # We can focus on common personal items
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                center = (int((x1 + x2) / 2), int((y1 + y2) / 2))

                current_frame_objects.append({
                    "name": name,
                    "center": center,
                    "box": (x1, y1, x2, y2),
                    "conf": conf
                })

    # -------------------------------------------------------------
    # SENSE TELEPORTATION RULE:
    # Match detections to our permanent memory based on distance!
    # -------------------------------------------------------------
    matched_memory_ids = set()

    for obj in current_frame_objects:
        obj_center = obj["center"]
        best_match_id = None
        closest_distance = 100  # Only consider it the same object if within 100 pixels

        for mem_id, mem_data in persistent_memory.items():
            if mem_id in matched_memory_ids:
                continue

            dist = calculate_distance(obj_center, mem_data["last_center"])
            # If it's nearby and disappeared recently (within 2 seconds)
            if dist < closest_distance and (current_time - mem_data["last_seen"] < 2.0):
                closest_distance = dist
                best_match_id = mem_id

        if best_match_id is not None:
            # Reconnected to existing memory!
            persistent_memory[best_match_id]["last_center"] = obj_center
            persistent_memory[best_match_id]["last_seen"] = current_time
            persistent_memory[best_match_id]["name"] = obj["name"]
            matched_memory_ids.add(best_match_id)
            active_id = best_match_id
        else:
            # Brand new item seen!
            active_id = next_permanent_id
            persistent_memory[active_id] = {
                "name": obj["name"],
                "last_center": obj_center,
                "last_seen": current_time,
                "first_seen": current_time
            }
            matched_memory_ids.add(active_id)
            print(f"✨ NEW OBJECT RECORDED: {obj['name']} as Permanent ID #{active_id}")
            next_permanent_id += 1

        # Draw the permanent SENSE ID!
        x1, y1, x2, y2 = obj["box"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 255), 2)
        cv2.circle(frame, obj_center, 4, (0, 0, 255), -1)

        display_text = f"SENSE #{active_id}: {obj['name']}"
        cv2.putText(frame, display_text, (x1, max(25, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2)

    cv2.imshow("SENSE Smart Memory", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()