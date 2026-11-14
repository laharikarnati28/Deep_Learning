
"""
VisionPort - Task 7
Detect Bikes, Buses and Trucks Using YOLOv8

Expected Output:
Multiple vehicle detection.
"""

import cv2
import os
from collections import Counter
from ultralytics import YOLO

print("=" * 60)
print("VISIONPORT - TASK 7: MULTIPLE VEHICLE DETECTION")
print("=" * 60)

# 1. Define input files
video_path = "traffic_sample.mp4"
model_path = "yolov8n.pt"

# 2. Check required files
if not os.path.exists(video_path):
    print("[ERROR] Traffic video not found!")
    exit()

if not os.path.exists(model_path):
    print("[ERROR] YOLO model not found!")
    exit()

print("\n[SUCCESS] Traffic video found!")
print("[SUCCESS] YOLO model found!")

# 3. Load YOLO model
print("\nLoading YOLOv8 model...")
model = YOLO(model_path)
print("[SUCCESS] YOLOv8 model loaded!")

# 4. Open traffic video
video = cv2.VideoCapture(video_path)

if not video.isOpened():
    print("[ERROR] Unable to open traffic video!")
    exit()

print("[SUCCESS] Traffic video opened!")

# 5. Define vehicle classes
vehicle_classes = {
    1: "Bicycle",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck"
}

selected_classes = list(vehicle_classes.keys())

# 6. Initialize counters
frame_count = 0
total_detections = Counter()

print("\nStarting multiple vehicle detection...")
print("Detecting: Bicycles, Motorcycles, Buses and Trucks")
print("Press Q to stop detection.")

# 7. Read video frame by frame
while True:

    ret, frame = video.read()

    if not ret:
        print("\n[SUCCESS] Video processing completed!")
        break

    frame_count += 1

    # Resize frame while maintaining aspect ratio
    original_height, original_width = frame.shape[:2]
    new_width = 960
    new_height = int(original_height * new_width / original_width)

    frame = cv2.resize(frame, (new_width, new_height))

    # 8. Run YOLO detection
    results = model(
        frame,
        classes=selected_classes,
        conf=0.25,
        verbose=False
    )

    # 9. Count each detected vehicle class
    class_counts = Counter()

    for class_id in results[0].boxes.cls.tolist():
        class_id = int(class_id)
        class_counts[class_id] += 1
        total_detections[class_id] += 1

    # 10. Draw bounding boxes without text labels
    annotated_frame = results[0].plot(
        labels=False,
        conf=False
    )

    # 11. Display vehicle counts
    y_position = 40

    for class_id, vehicle_name in vehicle_classes.items():

        count = class_counts[class_id]

        cv2.putText(
            annotated_frame,
            f"{vehicle_name}: {count}",
            (20, y_position),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        y_position += 35

    # 12. Display processed video
    cv2.imshow(
        "VisionPort - Multiple Vehicle Detection",
        annotated_frame
    )

    # Print progress every 30 frames
    if frame_count % 30 == 0:

        print(f"\nProcessed frames: {frame_count}")

        for class_id, vehicle_name in vehicle_classes.items():
            print(
                f"{vehicle_name}: "
                f"{class_counts[class_id]} in current frame"
            )

    # 13. Stop when Q is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("\nDetection stopped by user.")
        break

# 14. Release resources
video.release()
cv2.destroyAllWindows()

# 15. Display final results
print("\n" + "=" * 60)
print("TASK 7 FINAL RESULT")
print("=" * 60)

print("Total frames processed:", frame_count)

print("\nTotal detections by vehicle type:")

for class_id, vehicle_name in vehicle_classes.items():
    print(
        f"{vehicle_name}: "
        f"{total_detections[class_id]}"
    )

total_count = sum(total_detections.values())

print("\nTotal vehicle detections:", total_count)

if frame_count > 0:
    print("\nTASK 7 RESULT: PASSED")
    print("Multiple vehicle detection completed successfully.")
else:
    print("\nTASK 7 RESULT: FAILED")

print("=" * 60)
