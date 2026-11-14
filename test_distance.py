import cv2
from ultralytics import YOLO

model = YOLO("yolov8s.pt")
camera = cv2.VideoCapture(0)

# True real-world length of the longest side (in cm)
OBJECT_SPECS = {
    "cell phone": {"size_cm": 15.0, "type": "handheld"},
    "remote":     {"size_cm": 18.0, "type": "handheld"},
    "bottle":     {"size_cm": 23.0, "type": "handheld"},
    "cup":        {"size_cm": 10.0, "type": "handheld"},
    "laptop":     {"size_cm": 26.0, "type": "handheld"},
    "book":       {"size_cm": 23.0, "type": "handheld"},
    "person":     {"type": "human"}
}

# Calibrated focal length for MacBook Air webcam
FOCAL_LENGTH = 600.0

def calculate_smart_distance(name, width_px, height_px):
    """Calculates distance using aspect ratio and orientation-invariant metrics."""
    if name == "person":
        aspect_ratio = height_px / max(1, width_px)
        # If tall and slender (ratio > 2.0), it's a full standing person!
        # If wider/shorter (ratio <= 2.0), it's just head and torso!
        if aspect_ratio > 2.0:
            real_size_cm = 165.0  # Standing human
        else:
            real_size_cm = 75.0   # Sitting / upper body
        
        measuring_pixel_dim = height_px
    else:
        # For handheld items: ALWAYS use the longest edge so tilts don't break the math!
        real_size_cm = OBJECT_SPECS[name]["size_cm"]
        measuring_pixel_dim = max(width_px, height_px)

    # Physics formula: (Real Size * Focal Length) / Pixel Size
    distance_cm = (real_size_cm * FOCAL_LENGTH) / max(1, measuring_pixel_dim)
    distance_feet = distance_cm / 30.48
    return distance_feet

def get_assistive_phrase(distance_feet):
    if distance_feet < 2.2:
        return "within arm's reach"
    elif distance_feet < 5.5:
        return f"about {round(distance_feet)} feet away"
    else:
        return f"further away, about {round(distance_feet)} feet"

print("------------------------------------------------------------------")
print("SENSE Smart Depth Estimator Online!")
print("👉 Stand across the room: Watch the person distance read ~7 feet.")
print("👉 Hold your phone while across the room: Watch the distances match!")
print("👉 Press 'q' to quit.")
print("------------------------------------------------------------------")

while True:
    success, frame = camera.read()
    if not success:
        break

    results = model.track(frame, persist=True, conf=0.25, tracker="bytetrack.yaml", verbose=False)
    boxes = results[0].boxes

    if boxes is not None and boxes.id is not None:
        for box in boxes:
            raw_name = model.names[int(box.cls[0])]

            if raw_name not in OBJECT_SPECS:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            width_px = x2 - x1
            height_px = y2 - y1

            dist_feet = calculate_smart_distance(raw_name, width_px, height_px)
            phrase = get_assistive_phrase(dist_feet)

            color = (0, 255, 0) if dist_feet < 2.2 else ((0, 255, 255) if dist_feet < 5.5 else (255, 180, 0))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            label = f"{raw_name}: {dist_feet:.1f} ft ({phrase})"
            cv2.putText(frame, label, (x1, max(25, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    cv2.imshow("Smart SENSE Depth", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()