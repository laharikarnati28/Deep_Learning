
"""
VisionPort - Task 9
Count the Number of Vehicles in a Video Frame

Expected Output:
Vehicle count.
"""

import cv2
import os
from collections import Counter
from ultralytics import YOLO

print("=" * 60)
print("VISIONPORT - TASK 9: VEHICLE COUNTING")
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

# 5. Define vehicle categories
vehicle_classes = {
    1: "Bicycle",
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck"
}

selected_classes = list(vehicle_classes.keys())

# 6. Initialize counters
frame_count = 0
total_frame_vehicle_detections = 0

print("\nStarting vehicle counting...")
print("Press Q to stop counting.")

# 7. Process video frame by frame
while True:

    ret, frame = video.read()

    if not ret:
        print("\n[SUCCESS] Video processing completed!")
        break

    frame_count += 1

    # Resize frame while maintaining aspect ratio
    original_height, original_width = frame.shape[:2]

    new_width = 960
    new_height = int(
        original_height * new_width / original_width
    )

    frame = cv2.resize(
        frame,
        (new_width, new_height)
    )

    # 8. Detect vehicles using YOLO
    results = model(
        frame,
        classes=selected_classes,
        conf=0.25,
        verbose=False
    )

    # 9. Count vehicles in the current frame
    class_counts = Counter()

    for class_id in results[0].boxes.cls.tolist():
        class_id = int(class_id)
        class_counts[class_id] += 1

    # Calculate total vehicles in this frame
    total_vehicles = sum(class_counts.values())

    total_frame_vehicle_detections += total_vehicles

    # 10. Draw bounding boxes
    annotated_frame = results[0].plot()

    # 11. Display individual vehicle counts
    y_position = 40

    for class_id, vehicle_name in vehicle_classes.items():

        count = class_counts[class_id]

        cv2.putText(
            annotated_frame,
            f"{vehicle_name}: {count}",
            (20, y_position),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        y_position += 30

    # 12. Display total vehicle count
    cv2.putText(
        annotated_frame,
        f"TOTAL VEHICLES: {total_vehicles}",
        (20, y_position + 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 0, 255),
        2
    )

    # 13. Show processed frame
    cv2.imshow(
        "VisionPort - Vehicle Counting",
        annotated_frame
    )

    # Print count every 30 frames
    if frame_count % 30 == 0:

        print("\nFrame Number:", frame_count)

        for class_id, vehicle_name in vehicle_classes.items():
            print(
                f"{vehicle_name}: {class_counts[class_id]}"
            )

        print("Total Vehicles:", total_vehicles)

    # 14. Stop when Q is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("\nCounting stopped by user.")
        break

# 15. Release resources
video.release()
cv2.destroyAllWindows()

# 16. Display final result
print("\n" + "=" * 60)
print("TASK 9 FINAL RESULT")
print("=" * 60)

print("Total frames processed:", frame_count)
print(
    "Total vehicle detections across frames:",
    total_frame_vehicle_detections
)

if frame_count > 0:
    print("\nTASK 9 RESULT: PASSED")
    print("Vehicle counting completed successfully.")
else:
    print("\nTASK 9 RESULT: FAILED")

print("=" * 60)
