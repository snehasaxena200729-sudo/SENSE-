import cv2
import numpy as np
import time
from ultralytics import YOLO

# 1. Load YOLO (which already runs smoothly at 30 FPS on your Mac!)
model = YOLO("yolov8s.pt")
camera = cv2.VideoCapture(0)

# Memory of learned people: Name -> Visual Appearance Signature
known_profiles = {}
current_name = "Unknown Person"
latest_person_crop = None

def compute_appearance_profile(person_crop):
    """Generates an orientation-invariant color/texture signature of the person's upper body."""
    # Focus on upper 60% of the person (face, shoulders, upper clothing)
    h, w, _ = person_crop.shape
    upper_body = person_crop[0:int(h * 0.6), :]
    
    # Resize and convert to HSV color space
    resized = cv2.resize(upper_body, (80, 100))
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    
    # Calculate 2D Hue-Saturation color signature
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 20], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist

def match_profile(current_sig, threshold=0.60):
    """Compares the current person against all learned people in memory."""
    if not known_profiles:
        return "Unknown Person"

    best_match = "Unknown Person"
    best_score = -1.0

    for name, saved_sig in known_profiles.items():
        # Correlation comparison: 1.0 = exact match
        score = cv2.compareHist(current_sig, saved_sig, cv2.HISTCMP_CORREL)
        if score > best_score:
            best_score = score
            if score > threshold:
                best_match = name

    return best_match

print("------------------------------------------------------------------")
print("SENSE Person Re-ID & Name Memory Online!")
print("👉 1. Stand/sit in front of camera: SENSE detects 'Unknown Person'.")
print("👉 2. Press 'N' on your keyboard to teach SENSE your name!")
print("👉 3. Step away and return: SENSE will recognize you by name!")
print("👉 Press 'q' to quit.")
print("------------------------------------------------------------------")

while True:
    success, frame = camera.read()
    if not success:
        break

    results = model.track(frame, persist=True, conf=0.30, tracker="bytetrack.yaml", verbose=False)
    boxes = results[0].boxes

    person_detected = False

    if boxes is not None and boxes.id is not None:
        for box in boxes:
            class_id = int(box.cls[0])
            label = model.names[class_id]

            # We only care about people for this module!
            if label == "person":
                person_detected = True
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Make sure coordinates are inside the frame
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                
                person_crop = frame[y1:y2, x1:x2]
                if person_crop.size > 0:
                    latest_person_crop = person_crop

                    # Extract visual signature & match identity
                    sig = compute_appearance_profile(person_crop)
                    current_name = match_profile(sig)

                    # Calculate Distance (from Step 19!)
                    box_height = y2 - y1
                    box_width = x2 - x1
                    aspect_ratio = box_height / max(1, box_width)
                    real_size = 165.0 if aspect_ratio > 2.0 else 75.0
                    dist_feet = (real_size * 600.0) / (box_height * 30.48)

                    # Green for recognized, Orange for unknown
                    color = (0, 255, 120) if current_name != "Unknown Person" else (0, 165, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    display_tag = f"{current_name} ({dist_feet:.1f} ft)"
                    cv2.putText(frame, display_tag, (x1, max(25, y1 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    if not person_detected:
        current_name = "No Person in View"

    # HUD
    cv2.putText(frame, "Press 'N' to name person | 'q' to Quit", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    cv2.imshow("SENSE Person Identity Engine", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key in (ord('n'), ord('N')):
        if latest_person_crop is not None and latest_person_crop.size > 0:
            print("\n------------------------------------------------")
            name = input("👤 Enter name for this person (e.g. Sneha): ").strip()
            if name:
                sig = compute_appearance_profile(latest_person_crop)
                known_people = known_profiles[name] = sig
                print(f"✨ Successfully learned identity for: {name}!")
            print("------------------------------------------------\n")
        else:
            print("\n⚠️ No person visible to learn right now!")

camera.release()
cv2.destroyAllWindows()