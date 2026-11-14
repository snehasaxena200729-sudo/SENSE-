import cv2
import numpy as np

camera = cv2.VideoCapture(0)

# Parameters for Lucas-Kanade Optical Flow (the background dot tracker)
feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=15, blockSize=7)
lk_params = dict(winSize=(15, 15), maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Take the first frame to find background points
ret, old_frame = camera.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

print("------------------------------------------------------------------")
print("SENSE Camera Ego-Motion Tracker Online!")
print("👉 1. Keep laptop still: Notice camera speed is 0.")
print("👉 2. Turn your laptop left and right: Watch SENSE detect YOUR movement!")
print("👉 Press 'q' to quit.")
print("------------------------------------------------------------------")

while True:
    ret, frame = camera.read()
    if not ret:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    camera_dx = 0.0
    camera_dy = 0.0

    # Calculate Optical Flow: Where did our background dots move?
    if p0 is not None and len(p0) > 0:
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        if p1 is not None:
            # Select good points
            good_new = p1[st == 1]
            good_old = p0[st == 1]

            if len(good_new) > 10:
                # Calculate the median movement of ALL background points
                displacements = good_new - good_old
                camera_dx = np.median(displacements[:, 0])
                camera_dy = np.median(displacements[:, 1])

                # Draw the background tracking dots
                for new in good_new:
                    a, b = map(int, new.ravel())
                    cv2.circle(frame, (a, b), 3, (0, 255, 255), -1)

                p0 = good_new.reshape(-1, 1, 2)
            else:
                # If we lost too many dots, re-detect fresh ones
                p0 = cv2.goodFeaturesToTrack(frame_gray, mask=None, **feature_params)
        else:
            p0 = cv2.goodFeaturesToTrack(frame_gray, mask=None, **feature_params)
    else:
        p0 = cv2.goodFeaturesToTrack(frame_gray, mask=None, **feature_params)

    # Prepare for next frame
    old_gray = frame_gray.copy()

    # -------------------------------------------------------------
    # SENSE CAMERA STATE DETERMINATION:
    # Is the user turning the laptop, or is the camera still?
    # -------------------------------------------------------------
    camera_speed = (camera_dx**2 + camera_dy**2)**0.5

    if camera_speed > 3.0:
        direction = "PANNING RIGHT" if camera_dx < -1 else ("PANNING LEFT" if camera_dx > 1 else "MOVING")
        status_text = f"CAMERA ROTATING: {direction} ({camera_speed:.1f} px/frame)"
        status_color = (0, 0, 255)  # Red alert
    else:
        status_text = "CAMERA STATE: STILL (STATIONARY)"
        status_color = (0, 255, 0)  # Green safe

    cv2.putText(frame, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

    cv2.imshow("SENSE Camera Ego-Motion", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()