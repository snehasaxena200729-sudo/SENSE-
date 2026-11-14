import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
camera = cv2.VideoCapture(0)

print("Starting Object Tracker! Move your phone/bottle around. Press 'q' to stop.")

while True:
    success, frame = camera.read()
    if not success:
        break

    # Notice: We use model.track() instead of model()
    # persist=True tells it to remember IDs across frames!
    results = model.track(frame, persist=True, verbose=False)

    boxes = results[0].boxes

    # Sometimes the AI sees no objects or hasn't assigned IDs yet
    if boxes is not None and boxes.id is not None:
        for box in boxes:
            confidence = float(box.conf[0])

            if confidence > 0.50:
                # 1. Get the sticky note ID! (e.g. 1, 2, 3)
                track_id = int(box.id[0])

                # 2. Get the name
                class_id = int(box.cls[0])
                object_name = model.names[class_id]

                # 3. Get box coordinates and center point
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)

                # Print to terminal
                print(f"Tracking [ID: {track_id}] {object_name} at ({center_x}, {center_y})")

                # 4. Draw a box with the ID number!
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 120), 2)
                cv2.circle(frame, (center_x, center_y), 4, (0, 0, 255), -1)
                
                # Display: e.g. "ID: 1 | cell phone"
                label = f"ID: {track_id} | {object_name}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 120), 2)

    cv2.imshow("SENSE Tracking View", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()