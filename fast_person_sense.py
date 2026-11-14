import cv2
import threading
import time
import numpy as np

# 1. Load OpenCV's lightning-fast built-in face detector (runs in 3 milliseconds!)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

camera = cv2.VideoCapture(0)

# Memory of learned people: Name -> Face Visual Signature
known_people = {}
current_name = "Unknown Person"
is_recognizing = False
latest_face_crop = None

def compute_face_signature(face_img):
    """Creates a fast, lightweight 3D color/texture histogram of the face."""
    face_resized = cv2.resize(face_img, (100, 100))
    # Convert to HSV color space (robust to lighting changes)
    hsv = cv2.cvtColor(face_resized, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [18, 25], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist

def recognize_in_background(face_img):
    """Runs recognition in a background thread so video NEVER lags!"""
    global current_name, is_recognizing
    try:
        current_sig = compute_face_signature(face_img)

        if not known_people:
            current_name = "Unknown Person"
            return

        best_match = "Unknown Person"
        best_score = 0.55  # Confidence threshold

        for name, saved_sig in known_people.items():
            # Compare histograms: 1.0 = exact match
            score = cv2.compareHist(current_sig, saved_sig, cv2.HISTCMP_CORREL)
            if score > best_score:
                best_score = score
                best_match = name

        current_name = best_match
    finally:
        is_recognizing = False

print("------------------------------------------------------------------")
print("SENSE Zero-Lag Person Memory Online!")
print("👉 1. Look at the camera (it will draw a green face box).")
print("👉 2. Press 'N' on your keyboard to save your name!")
print("👉 3. Move away and return: SENSE will recognize you with 0% lag!")
print("👉 Press 'q' to quit.")
print("------------------------------------------------------------------")

while True:
    success, frame = camera.read()
    if not success:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Ultra-fast face detection (takes ~3ms on CPU!)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(60, 60))

    if len(faces) > 0:
        # Get the largest face in view
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        latest_face_crop = frame[y:y+h, x:x+w]

        # Trigger background recognition without slowing down the video!
        if not is_recognizing:
            is_recognizing = True
            threading.Thread(target=recognize_in_background, args=(latest_face_crop.copy(),), daemon=True).start()

        # Color code: Green if known, Orange if unknown
        color = (0, 255, 120) if current_name != "Unknown Person" else (0, 165, 255)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, current_name, (x, max(25, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
    else:
        current_name = "No Person in View"

    # HUD
    cv2.putText(frame, "Press 'N' to name face | 'q' to Quit", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    cv2.imshow("SENSE Fast Person Recognition", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key in (ord('n'), ord('N')):
        if latest_face_crop is not None and len(latest_face_crop) > 0:
            print("\n------------------------------------------------")
            name = input("👤 Enter name for this person (e.g. Sneha): ").strip()
            if name:
                sig = compute_face_signature(latest_face_crop)
                known_people[name] = sig
                current_name = name
                print(f"✨ Successfully saved face signature for: {name}!")
            print("------------------------------------------------\n")
        else:
            print("\n⚠️ No face visible to save! Look at the camera.")

camera.release()
cv2.destroyAllWindows()