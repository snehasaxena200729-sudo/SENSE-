import cv2
import time
from ultralytics import YOLO

# Load model
model = YOLO("yolov8n.pt")
camera = cv2.VideoCapture(0)

print("Starting Robust SENSE Tracker! Move your remote now. Press 'q' to stop.")

while True:
    success, frame = camera.read()
    if not success:
        break

    # -------------------------------------------------------------
    # 1. conf=0.25: Don't be too strict! Catch objects even at odd angles.
    # 2. tracker="bytetrack.yaml": Uses the famous ByteTrack algorithm,
    #    which is specially designed to NOT lose objects when they blur.
    # -------------------------------------------------------------
    results = model.track(
        frame, 
        persist=True, 
        conf=0.25,                 # Lower threshold so tilts don't break it
        tracker="bytetrack.yaml",   # Robust industry-standard tracker
        verbose=False
    )

    boxes = results[0].boxes

    if boxes is not None and boxes.id is not None:
        for box in boxes:
            track_id = int(box.id[0])
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            object_name = model.names[class_id]

            # Get box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            # Draw a bright cyan box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.circle(frame, (center_x, center_y), 4, (0, 0, 255), -1)

            # Show ID, Name, and Confidence % on screen so we see how sure it is!
            label = f"ID: {track_id} | {object_name} ({int(confidence * 100)}%)"
            cv2.putText(
                frame, 
                label, 
                (x1, max(20, y1 - 10)), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.55, 
                (255, 255, 0), 
                2
            )

    cv2.imshow("Robust SENSE Tracker", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()