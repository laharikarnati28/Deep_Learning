
"""
VisionPort - Task 10
Traffic Density Calculation

Expected Output:
Low, Medium and High Traffic
"""

import cv2
import os
from collections import Counter
from ultralytics import YOLO

print("=" * 60)
print("VISIONPORT - TASK 10: TRAFFIC DENSITY")
print("=" * 60)

# 1. Input files
video_path = "traffic_sample.mp4"
model_path = "yolov8n.pt"

# 2. Check files
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

# 4. Open video
video = cv2.VideoCapture(video_path)

if not video.isOpened():
    print("[ERROR] Unable to open video!")
    exit()

print("[SUCCESS] Traffic video opened!")

# 5. Vehicle categories
vehicle_classes = {
    1: "Bicycle",
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck"
}

selected_classes = list(vehicle_classes.keys())

# 6. Traffic density thresholds
LOW_LIMIT = 10
MEDIUM_LIMIT = 20

# 7. Initialize counters
frame_count = 0
total_detections = 0

print("\nStarting traffic density calculation...")
print("Press Q to stop.")

# Function to classify traffic density
def calculate_density(vehicle_count):

    if vehicle_count <= LOW_LIMIT:
        return "LOW", (0, 255, 0)

    elif vehicle_count <= MEDIUM_LIMIT:
        return "MEDIUM", (0, 165, 255)

    else:
        return "HIGH", (0, 0, 255)


# 8. Process video frame by frame
while True:

    ret, frame = video.read()

    if not ret:
        print("\n[SUCCESS] Video processing completed!")
        break

    frame_count += 1

    # Resize frame
    original_height, original_width = frame.shape[:2]

    new_width = 960
    new_height = int(
        original_height * new_width / original_width
    )

    frame = cv2.resize(
        frame,
        (new_width, new_height)
    )

    # 9. Detect vehicles
    results = model(
        frame,
        classes=selected_classes,
        conf=0.25,
        verbose=False
    )

    # 10. Count vehicles in current frame
    class_counts = Counter()

    for class_id in results[0].boxes.cls.tolist():

        class_id = int(class_id)
        class_counts[class_id] += 1

    vehicle_count = sum(class_counts.values())

    total_detections += vehicle_count

    # 11. Calculate traffic density
    density, density_color = calculate_density(
        vehicle_count
    )

    # 12. Draw bounding boxes
    annotated_frame = results[0].plot()

    # 13. Display vehicle count
    cv2.putText(
        annotated_frame,
        f"Vehicle Count: {vehicle_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2
    )

    # 14. Display traffic density
    cv2.putText(
        annotated_frame,
        f"Traffic Density: {density}",
        (20, 85),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        density_color,
        3
    )

    # 15. Display video
    cv2.imshow(
        "VisionPort - Traffic Density Monitoring",
        annotated_frame
    )

    # Print result every 30 frames
    if frame_count % 30 == 0:

        print("\nFrame Number:", frame_count)
        print("Vehicle Count:", vehicle_count)
        print("Traffic Density:", density)

    # 16. Stop using Q
    if cv2.waitKey(1) & 0xFF == ord('q'):

        print("\nProcessing stopped by user.")
        break

# 17. Release resources
video.release()
cv2.destroyAllWindows()

# 18. Final result
print("\n" + "=" * 60)
print("TASK 10 FINAL RESULT")
print("=" * 60)

print("Total frames processed:", frame_count)
print("Total vehicle detections:", total_detections)

if frame_count > 0:

    print("\nTASK 10 RESULT: PASSED")
    print("Traffic density calculated successfully.")

else:

    print("\nTASK 10 RESULT: FAILED")

print("=" * 60)
