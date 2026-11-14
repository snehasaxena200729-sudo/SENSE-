import cv2
import time
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
camera = cv2.VideoCapture(0)

# -------------------------------------------------------------
# THIS IS SENSE'S MEMORY NOTEBOOK
# Key = track_id (e.g., 1)
# Value = information about that object
# -------------------------------------------------------------
memory = {}

print("SENSE Memory active! Try showing an object, then hiding it.")

while True:
    success, frame = camera.read()
    if not success:
        break

    current_time = time.time()
    results = model.track(frame, persist=True, verbose=False)
    boxes = results[0].boxes

    # A list of all IDs currently seen in THIS exact frame
    currently_seen_ids = []

    if boxes is not None and boxes.id is not None:
        for box in boxes:
            if float(box.conf[0]) > 0.50:
                track_id = int(box.id[0])
                currently_seen_ids.append(track_id)

                class_id = int(box.cls[0])
                object_name = model.names[class_id]

                x1, y1, x2, y2 = box.xyxy[0]
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)

                # If this is the first time we see this item, welcome it!
                if track_id not in memory:
                    print(f"✨ NEW OBJECT DETECTED: {object_name} (ID: {track_id}) at ({center_x}, {center_y})")
                    memory[track_id] = {
                        "name": object_name,
                        "last_x": center_x,
                        "last_y": center_y,
                        "last_seen": current_time,
                        "status": "VISIBLE"
                    }
                else:
                    # Update its location and timestamp in our memory
                    memory[track_id]["last_x"] = center_x
                    memory[track_id]["last_y"] = center_y
                    memory[track_id]["last_seen"] = current_time
                    memory[track_id]["status"] = "VISIBLE"

                # Draw green box
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, f"ID: {track_id} {object_name}", (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # -------------------------------------------------------------
    # THE MEMORY CHECK: Did any object suddenly vanish?
    # -------------------------------------------------------------
    for track_id, item in memory.items():
        # If it was visible before, but is NOT in this frame...
        if item["status"] == "VISIBLE" and track_id not in currently_seen_ids:
            # Check how long it's been gone (give it 1 second grace period to prevent flickering)
            time_since_seen = current_time - item["last_seen"]
            if time_since_seen > 1.0:
                item["status"] = "DISAPPEARED"
                print(f"⚠️ EVENT: {item['name']} (ID: {track_id}) has DISAPPEARED!")
                print(f"   -> Last known location was: ({item['last_x']}, {item['last_y']})")

    # Display on screen how many items are in SENSE's memory
    total_remembered = len(memory)
    cv2.putText(frame, f"Remembered Items: {total_remembered}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    cv2.imshow("SENSE Environmental Memory", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()