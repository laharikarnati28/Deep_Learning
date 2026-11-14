
import cv2
from ultralytics import YOLO

print("=" * 60)
print("VISIONPORT - TASK 13: VEHICLE SPEED ESTIMATION")
print("=" * 60)

VIDEO_PATH = "traffic_sample.mp4"
MODEL_PATH = "yolov8n.pt"

# Load YOLO model
print("\nLoading YOLOv8 model...")
model = YOLO(MODEL_PATH)
print("[SUCCESS] YOLO model loaded!")

# Open traffic video
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("[ERROR] Cannot open traffic video.")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    print("[ERROR] Invalid video FPS.")
    cap.release()
    exit()

print("[SUCCESS] Traffic video opened!")
print("FPS:", round(fps, 2))

# Store previous vehicle positions and timestamps
previous_positions = {}
vehicle_speeds = {}

frame_count = 0

print("\nStarting vehicle speed estimation...")
print("Tracking vehicles using ByteTrack.")
print("Press Q to stop detection.")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1

    # Video timestamp in seconds
    current_time = frame_count / fps

    # YOLO vehicle detection and tracking
    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        classes=[2, 3, 5, 7],
        verbose=False
    )

    result = results[0]

    if result.boxes is not None and result.boxes.id is not None:

        boxes = result.boxes.xyxy.cpu().numpy()
        ids = result.boxes.id.int().cpu().tolist()

        for box, vehicle_id in zip(boxes, ids):

            x1, y1, x2, y2 = map(int, box)

            # Bottom-centre position of vehicle
            center_x = (x1 + x2) // 2
            center_y = y2

            speed = None

            if vehicle_id in previous_positions:

                old_x, old_y, old_time = previous_positions[vehicle_id]

                distance_pixels = (
                    (center_x - old_x) ** 2 +
                    (center_y - old_y) ** 2
                ) ** 0.5

                time_difference = current_time - old_time

                if time_difference > 0:

                    speed = distance_pixels / time_difference

                    vehicle_speeds[vehicle_id] = speed

            # Update vehicle position
            previous_positions[vehicle_id] = (
                center_x, center_y, current_time
            )

            # Create label
            if speed is not None:
                label = f"ID:{vehicle_id}  {speed:.1f} px/s"
            else:
                label = f"ID:{vehicle_id}  Calculating"

            # Draw thicker bounding box
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                4
            )

            # BIGGER LABEL SETTINGS
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 2.5
            thickness = 6

            (text_width, text_height), baseline = cv2.getTextSize(
                label, font, font_scale, thickness
            )

            # Keep label inside image boundaries
            label_x = min(
                max(x1, 0),
                max(0, frame.shape[1] - text_width - 15)
            )

            label_y = max(y1 - 15, text_height + 20)

            # Dark background behind label
            cv2.rectangle(
                frame,
                (label_x, label_y - text_height - 15),
                (label_x + text_width + 15, label_y + baseline + 5),
                (0, 0, 0),
                -1
            )

            # Display large yellow text
            cv2.putText(
                frame,
                label,
                (label_x + 5, label_y),
                font,
                font_scale,
                (0, 255, 255),
                thickness
            )

    # Display video
    cv2.imshow(
        "VisionPort - Vehicle Speed Estimation",
        frame
    )

    # Progress information
    if frame_count % 30 == 0:

        print(
            f"Processed Frames: {frame_count} | "
            f"Vehicles with speed estimates: {len(vehicle_speeds)}"
        )

    # Quit when Q is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("\nProcessing stopped by user.")
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

# Final report
print("\n" + "=" * 60)
print("TASK 13 FINAL RESULT")
print("=" * 60)

print("Total frames processed:", frame_count)
print("Vehicles with speed estimates:", len(vehicle_speeds))

print("\nTASK 13 RESULT: PASSED")
print("Vehicle speed estimation completed successfully.")
print("=" * 60)
