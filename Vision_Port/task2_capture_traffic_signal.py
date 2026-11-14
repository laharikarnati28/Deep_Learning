
"""
VisionPort - Task 2
Capture an Image of a Traffic Signal Using Camera

Expected Output:
Traffic image captured and saved successfully.
"""

import cv2
import os

print("=" * 60)
print("VISIONPORT - TASK 2: TRAFFIC IMAGE CAPTURE")
print("=" * 60)

# 1. Open the camera
print("\nOpening camera...")

camera = cv2.VideoCapture(0)

# 2. Check camera availability
if not camera.isOpened():
    print("[ERROR] Unable to access camera.")
    print("Check camera permissions and try again.")
    exit()

print("[SUCCESS] Camera opened successfully!")

print("\nCamera Instructions:")
print("Press 'S' to capture and save the image.")
print("Press 'Q' to quit without capturing.")

image_saved = False

# 3. Display camera feed
while True:

    ret, frame = camera.read()

    if not ret:
        print("[ERROR] Unable to read camera frame.")
        break

    cv2.imshow("VisionPort - Traffic Camera", frame)

    # 4. Read keyboard input
    key = cv2.waitKey(1) & 0xFF

    # 5. Capture image when S is pressed
    if key == ord('s'):

        filename = "captured_traffic_signal.jpg"

        success = cv2.imwrite(filename, frame)

        if success:
            print("\n[SUCCESS] Image captured!")
            print("Saved as:", os.path.abspath(filename))
            image_saved = True
        else:
            print("[ERROR] Image could not be saved.")

        break

    # 6. Exit when Q is pressed
    elif key == ord('q'):
        print("\nCamera closed without capturing an image.")
        break

# 7. Release camera resources
camera.release()
cv2.destroyAllWindows()

# 8. Display final result
print("\n" + "=" * 60)

if image_saved:
    print("TASK 2 RESULT: PASSED")
    print("Traffic image captured and saved successfully.")
else:
    print("TASK 2 RESULT: NOT COMPLETED")

print("=" * 60)
