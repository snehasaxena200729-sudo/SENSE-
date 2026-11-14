import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
camera = cv2.VideoCapture(0)

print("Reading raw AI data... Hold an object up! Press 'q' to stop.")

while True:
    success, frame = camera.read()
    if not success:
        break

    # Run the AI
    results = model(frame, verbose=False)

    # results[0].boxes contains all the objects YOLO found in this photo
    boxes = results[0].boxes

    for box in boxes:
        # 1. Get the confidence score (from 0.0 to 1.0)
        confidence = float(box.conf[0])

        # We only care if the AI is at least 50% sure! (Ignore weak guesses)
        if confidence > 0.50:
            # 2. Get the object's class ID and look up its English name
            class_id = int(box.cls[0])
            object_name = model.names[class_id]

            # 3. Get the box coordinates [x1, y1, x2, y2]
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # 4. Calculate the center point of the object
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            # Print it to the terminal!
            print(f"Spotted: {object_name} at Center ({center_x}, {center_y}) | Sureness: {int(confidence * 100)}%")

            # 5. Let's draw our OWN custom box and a dot right at the center!
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 100, 0), 2)
            cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)  # Red dot at center
            cv2.putText(
                frame, 
                f"{object_name} ({int(confidence * 100)}%)", 
                (x1, y1 - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, 
                (255, 100, 0), 
                2
            )

    cv2.imshow("SENSE Raw Data View", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()