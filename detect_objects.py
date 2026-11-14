import cv2
from ultralytics import YOLO

# 1. Load the pre-trained AI brain (it will download a tiny 6MB file the first time)
print("Loading AI model...")
model = YOLO("yolov8n.pt")

# 2. Connect to the camera
camera = cv2.VideoCapture(0)

print("Starting AI Vision! Hold up a cup, phone, or bottle. Press 'q' to stop.")

while True:
    success, frame = camera.read()
    if not success:
        break

    # 3. Give the photo to the AI and ask it to find objects!
    results = model(frame, verbose=False)

    # 4. YOLO has a built-in helper that automatically draws all the boxes and names for us!
    annotated_frame = results[0].plot()

    # 5. Show the AI's view
    cv2.imshow("SENSE AI Brain", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
print("SENSE stopped cleanly!")