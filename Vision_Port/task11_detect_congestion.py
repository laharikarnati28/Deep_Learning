
"""
VisionPort - Task 11
Traffic Congestion Detection

Expected Output:
Congestion alerts
"""

import cv2
import os
from ultralytics import YOLO

print("=" * 60)
print("VISIONPORT - TASK 11: TRAFFIC CONGESTION DETECTION")
print("=" * 60)

# 1. Check input files
video_path = "traffic_sample.mp4"
model_path = "yolov8n.pt"

if not os.path.exists(video_path):
    print("[ERROR] Traffic video not found!")
    exit()

if not os.path.exists(model_path):
    print("[ERROR] YOLO model not found!")
    exit()

print("\n[SUCCESS] Traffic video found!")
print("[SUCCESS] YOLO model found!")

# 2. Load YOLO model
print("\nLoading YOLOv8 model...")
model = YOLO(model_path)
print("[SUCCESS] YOLOv8 model loaded!")

# 3. Open video
video = cv2.VideoCapture(video_path)

if not video.isOpened():
    print("[ERROR] Unable to open video!")
    exit()

print("[SUCCESS] Traffic video opened!")

# 4. Vehicle classes
vehicle_classes = [1, 2, 3, 5, 7]

# 5. Congestion settings
CONGESTION_LIMIT = 10
ALERT_FRAMES = 15

# Region of Interest (ROI), as fractions of frame dimensions
ROI_X1 = 0.15
ROI_Y1 = 0.25
ROI_X2 = 0.85
ROI_Y2 = 0.85

frame_count = 0
total_detections = 0
congestion_frames = 0
alert_generated = False

print("\nStarting congestion detection...")
print("Monitoring selected road region.")
print("Press Q to stop detection.")

# 6. Process video
while True:

    ret, frame = video.read()

    if not ret:
        print("\n[SUCCESS] Video processing completed!")
        break

    frame_count += 1

    # Resize video
    height, width = frame.shape[:2]

    new_width = 960
    new_height = int(height * new_width / width)

    frame = cv2.resize(frame, (new_width, new_height))

    height, width = frame.shape[:2]

    # 7. Define monitoring region
    x1 = int(width * ROI_X1)
    y1 = int(height * ROI_Y1)
    x2 = int(width * ROI_X2)
    y2 = int(height * ROI_Y2)

    # 8. Detect vehicles
    results = model(
        frame,
        classes=vehicle_classes,
        conf=0.25,
        verbose=False
    )

    vehicle_count = 0

    # 9. Check vehicle centers inside ROI
    if results[0].boxes is not None:

        boxes = results[0].boxes.xyxy.cpu().tolist()

        for box in boxes:

            bx1, by1, bx2, by2 = box

            center_x = int((bx1 + bx2) / 2)
            center_y = int((by1 + by2) / 2)

            if (x1 <= center_x <= x2 and
                    y1 <= center_y <= y2):

                vehicle_count += 1

    total_detections += vehicle_count

    # 10. Check congestion
    if vehicle_count >= CONGESTION_LIMIT:
        congestion_frames += 1
    else:
        congestion_frames = 0
        alert_generated = False

    if congestion_frames >= ALERT_FRAMES:
        status = "CONGESTION ALERT!"
        color = (0, 0, 255)
        alert_generated = True
    elif vehicle_count >= CONGESTION_LIMIT:
        status = "TRAFFIC BUILDING UP"
        color = (0, 165, 255)
    else:
        status = "NORMAL TRAFFIC"
        color = (0, 255, 0)

    # 11. Draw detections and ROI
    output = results[0].plot()

    cv2.rectangle(
        output,
        (x1, y1),
        (x2, y2),
        (255, 255, 0),
        2
    )

    cv2.putText(
        output,
        f"Vehicles in ROI: {vehicle_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.putText(
        output,
        status,
        (20, 85),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        3
    )

    # 12. Display video
    cv2.imshow(
        "VisionPort - Congestion Detection",
        output
    )

    # Print every 30 frames
    if frame_count % 30 == 0:

        print("\nFrame Number:", frame_count)
        print("Vehicles in monitoring region:", vehicle_count)
        print("Status:", status)

        if alert_generated:
            print("[ALERT] Traffic congestion detected!")

    # 13. Stop using Q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("\nDetection stopped by user.")
        break

# 14. Release resources
video.release()
cv2.destroyAllWindows()

# 15. Final report
print("\n" + "=" * 60)
print("TASK 11 FINAL RESULT")
print("=" * 60)

print("Total frames processed:", frame_count)
print("Total vehicle detections in ROI:", total_detections)

if frame_count > 0:
    print("\nTASK 11 RESULT: PASSED")
    print("Congestion monitoring completed successfully.")
else:
    print("\nTASK 11 RESULT: FAILED")

print("=" * 60)
