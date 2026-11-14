
"""
VisionPort - Task 12
Individual Vehicle Tracking Using YOLOv8 + ByteTrack

Expected Output:
Vehicle tracking IDs
"""

import cv2
import os
from ultralytics import YOLO

print("=" * 60)
print("VISIONPORT - TASK 12: VEHICLE TRACKING")
print("=" * 60)

# 1. Check required files
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

# 3. Open traffic video
video = cv2.VideoCapture(video_path)

if not video.isOpened():
    print("[ERROR] Unable to open traffic video!")
    exit()

print("[SUCCESS] Traffic video opened!")

# COCO vehicle class IDs:
# 1 = Bicycle
# 2 = Car
# 3 = Motorcycle
# 5 = Bus
# 7 = Truck

vehicle_classes = [1, 2, 3, 5, 7]

# 4. Tracking variables
frame_count = 0
total_detections = 0
unique_vehicle_ids = set()

print("\nStarting vehicle tracking...")
print("Tracking vehicles using ByteTrack.")
print("Press Q to stop tracking.")

# 5. Process video frame by frame
while True:

    ret, frame = video.read()

    if not ret:
        print("\n[SUCCESS] Video processing completed!")
        break

    frame_count += 1

    # Resize frame for smoother processing
    height, width = frame.shape[:2]

    new_width = 960
    new_height = int(height * new_width / width)

    frame = cv2.resize(frame, (new_width, new_height))

    # 6. Run YOLO tracking
    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        classes=vehicle_classes,
        conf=0.25,
        verbose=False
    )

    result = results[0]

    # 7. Get tracking IDs
    if result.boxes is not None and result.boxes.id is not None:

        boxes = result.boxes
        tracking_ids = boxes.id.int().cpu().tolist()
        class_ids = boxes.cls.int().cpu().tolist()

        for track_id, class_id in zip(tracking_ids, class_ids):

            unique_vehicle_ids.add(track_id)
            total_detections += 1

        # Display tracking information
        output = result.plot()

    else:
        output = frame.copy()

    # 8. Display tracking statistics
    cv2.putText(
        output,
        f"Frame: {frame_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.putText(
        output,
        f"Unique Vehicle IDs: {len(unique_vehicle_ids)}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    # 9. Display output
    cv2.imshow(
        "VisionPort - Vehicle Tracking",
        output
    )

    # Print progress every 30 frames
    if frame_count % 30 == 0:

        print("\nProcessed Frames:", frame_count)
        print("Unique Vehicle IDs:", len(unique_vehicle_ids))
        print("Tracking detections:", total_detections)

    # 10. Stop when Q is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("\nTracking stopped by user.")
        break

# 11. Release resources
video.release()
cv2.destroyAllWindows()

# 12. Final report
print("\n" + "=" * 60)
print("TASK 12 FINAL RESULT")
print("=" * 60)

print("Total frames processed:", frame_count)
print("Total tracking detections:", total_detections)
print("Unique vehicle IDs:", len(unique_vehicle_ids))

if frame_count > 0 and len(unique_vehicle_ids) > 0:

    print("\nTASK 12 RESULT: PASSED")
    print("Individual vehicle tracking completed successfully.")

else:

    print("\nTASK 12 RESULT: FAILED")
    print("No vehicle tracking IDs were generated.")

print("=" * 60)
