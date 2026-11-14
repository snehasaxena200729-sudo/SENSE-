import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import cv2
import torch
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1

print("Initializing High-Speed Face Engine...")
device = torch.device('cpu')

# Fast MTCNN settings (smaller image size for speed)
mtcnn = MTCNN(image_size=160, margin=14, keep_all=False, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

camera = cv2.VideoCapture(0)

known_faces = {}
latest_face_embedding = None
current_identity = "Scanning for face..."
frame_count = 0

def get_face_embedding(frame):
    """Downscales the image and extracts face embedding in milliseconds."""
    # Shrink frame to 320x240 for 4x faster processing!
    small_frame = cv2.resize(frame, (320, 240))
    rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_frame)

    face_tensor = mtcnn(pil_img)
    if face_tensor is not None:
        with torch.no_grad():
            embedding = resnet(face_tensor.unsqueeze(0)).numpy()[0]
        return embedding
    return None

def match_face(embedding, threshold=0.60):
    if not known_faces:
        return "Unknown Person"

    best_match = "Unknown Person"
    best_similarity = -1.0

    for name, saved_emb in known_faces.items():
        similarity = np.dot(embedding, saved_emb) / (np.linalg.norm(embedding) * np.linalg.norm(saved_emb))
        if similarity > best_similarity:
            best_similarity = similarity
            if similarity > threshold:
                best_match = name

    return best_match

print("------------------------------------------------------------------")
print("SENSE High-Speed Person Recognition Online!")
print("👉 1. Look at camera (it will say 'Unknown Person').")
print("👉 2. Press 'N' on your keyboard to name this face.")
print("👉 3. Move away and come back to see your name recognized!")
print("👉 Press 'q' to quit.")
print("------------------------------------------------------------------")

while True:
    success, frame = camera.read()
    if not success:
        break

    frame_count += 1

    # -------------------------------------------------------------
    # THE 1-SECOND SAMPLING RULE:
    # Only run the heavy face brain ONCE every 20 frames!
    # All other frames run at smooth, lag-free 30 FPS.
    # -------------------------------------------------------------
    if frame_count % 20 == 0:
        embedding = get_face_embedding(frame)
        if embedding is not None:
            latest_face_embedding = embedding
            current_identity = match_face(embedding)
        else:
            if latest_face_embedding is None:
                current_identity = "No Face Visible"

    # Color code
    if current_identity not in ["No Face Visible", "Scanning for face...", "Unknown Person"]:
        color = (0, 255, 120)  # Green for recognized person!
    elif current_identity == "Unknown Person":
        color = (0, 165, 255)  # Orange for unknown
    else:
        color = (150, 150, 150)

    # Visual HUD
    cv2.putText(frame, f"RECOGNIZED: {current_identity}", (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
    cv2.putText(frame, "Press 'N' to name face | 'q' to Quit", (20, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow("SENSE Person Recognition", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key in (ord('n'), ord('N')):
        if latest_face_embedding is not None:
            print("\n------------------------------------------------")
            name = input("👤 Enter name for this person (e.g. Sneha): ").strip()
            if name:
                known_faces[name] = latest_face_embedding
                print(f"✨ Successfully memorized face for: {name}!")
            print("------------------------------------------------\n")
        else:
            print("\n⚠️ No face detected yet. Look straight at the camera and try again.")

camera.release()
cv2.destroyAllWindows()