 
"""
VisionPort - Task 3
Read and Display Traffic Images Using OpenCV

Expected Output:
Image display window
"""

import cv2
import os

print("=" * 60)
print("VISIONPORT - TASK 3: READ AND DISPLAY TRAFFIC IMAGE")
print("=" * 60)

# 1. Specify the image filename
image_path = "captured_traffic_signal.jpg"

# 2. Check whether the image exists
if not os.path.exists(image_path):
    print("[ERROR] Image file not found!")
    print("Please complete Task 2 first.")
    exit()

print("\n[SUCCESS] Image file found!")

# 3. Read the image using OpenCV
image = cv2.imread(image_path)

# 4. Check whether the image was read successfully
if image is None:
    print("[ERROR] Unable to read the image.")
    exit()

print("[SUCCESS] Image read successfully!")

# 5. Get image dimensions
height, width, channels = image.shape

print("\nImage Information:")
print("Width:", width, "pixels")
print("Height:", height, "pixels")
print("Channels:", channels)

# 6. Display the image
print("\nDisplaying traffic image...")
print("Press any key inside the image window to close it.")

cv2.imshow("VisionPort - Traffic Image", image)

# 7. Wait until a key is pressed
cv2.waitKey(0)

# 8. Close the image window
cv2.destroyAllWindows()

# 9. Final result
print("\n" + "=" * 60)
print("TASK 3 RESULT: PASSED")
print("Traffic image read and displayed successfully.")
print("=" * 60)
