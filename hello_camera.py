import cv2

camera = cv2.VideoCapture(0)

print("Starting SENSE with overlay... Press 'q' to stop!")

while True:
    success, frame = camera.read()

    if not success:
        print("Could not read from camera.")
        break

    # 1. Write text in the top-left corner
    # cv2.putText(image, text, position, font, size, color, thickness)
    cv2.putText(
        frame, 
        "SENSE: System Online", 
        (30, 50),                      # Position (X=30, Y=50)
        cv2.FONT_HERSHEY_SIMPLEX,      # Font style
        1.0,                           # Font size
        (0, 255, 0),                   # Green color (Blue=0, Green=255, Red=0)
        2                              # Thickness of lines
    )

    # 2. Draw a green target box in the center of the screen
    # cv2.rectangle(image, top_left_corner, bottom_right_corner, color, thickness)
    cv2.rectangle(
        frame, 
        (200, 150),                    # Top-left corner of the box
        (440, 350),                    # Bottom-right corner of the box
        (0, 255, 0),                   # Green color
        2                              # Thickness of the border
    )

    # Show the modified frame!
    cv2.imshow("SENSE Eye", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
print("Camera closed cleanly!")