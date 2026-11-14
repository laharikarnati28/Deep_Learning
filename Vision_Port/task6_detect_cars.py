
"""
VisionPort - Task 6
Detect Cars in a Traffic Video Using YOLOv8

Expected Output:
Car detection with bounding boxes.
"""

import cv2
import os
from ultralytics import YOLO

print("=" * 60)
print("VISIONPORT - TASK 6: YOLO CAR DETECTION")
print("=" * 60)

# 1. Define input files
video_path = "traffic_sample.mp4"
model_path = "yolov8n.pt"

# 2. Check whether required files exist
if not os.path.exists(video_path):
    print("[ERROR] Traffic video not found!")
    exit()

if not os.path.exists(model_path):
    print("[ERROR] YOLO model file not found!")
    exit()

print("\n[SUCCESS] Traffic video found!")
print("[SUCCESS] YOLO model found!")

# 3. Load YOLOv8 model
print("\nLoading YOLOv8 model...")

model = YOLO(model_path)

print("[SUCCESS] YOLOv8 model loaded!")

# 4. Open traffic video
video = cv2.VideoCapture(video_path)

if not video.isOpened():
    print("[ERROR] Unable to open traffic video!")
    exit()

print("[SUCCESS] Traffic video opened!")

# 5. Get video properties
fps = video.get(cv2.CAP_PROP_FPS)
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

print("\nVideo Information:")
print("Width:", width)
print("Height:", height)
print("FPS:", round(fps, 2))

# 6. Initialize counters
frame_count = 0
total_cars_detected = 0

print("\nStarting car detection...")
print("Press Q to stop detection.")

# 7. Process video frame by frame
while True:

    ret, frame = video.read()

    if not ret:
        print("\n[SUCCESS] Video processing completed!")
        break

    frame_count += 1

    # Resize frame to reduce processing time
    frame = cv2.resize(frame, (960, 1707))

    # 8. Run YOLO detection for cars only
    results = model(
        frame,
        classes=[2],
        conf=0.25,
        verbose=False
    )

    # 9. Draw bounding boxes
    annotated_frame = results[0].plot()

    # 10. Count detected cars in this frame
    car_count = len(results[0].boxes)

    total_cars_detected += car_count

    # 11. Display car count
    cv2.putText(
        annotated_frame,
        f"Cars Detected: {car_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # 12. Display processed video
    cv2.imshow(
        "VisionPort - YOLO Car Detection",
        annotated_frame
    )

    # Print progress
    if frame_count % 30 == 0:
        print(
            f"Processed frames: {frame_count} | "
            f"Cars in current frame: {car_count}"
        )

    # 13. Exit when Q is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("\nDetection stopped by user.")
        break

# 14. Release resources
video.release()
cv2.destroyAllWindows()

# 15. Display final result
print("\n" + "=" * 60)
print("TASK 6 FINAL RESULT")
print("=" * 60)

print("Total frames processed:", frame_count)
print("Total car detections across frames:", total_cars_detected)

if frame_count > 0:
    print("\nTASK 6 RESULT: PASSED")
    print("YOLO car detection completed successfully.")
else:
    print("\nTASK 6 RESULT: FAILED")

print("=" * 60)
