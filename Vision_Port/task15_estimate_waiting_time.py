
import cv2
from ultralytics import YOLO

print("=" * 60)
print("VISIONPORT - TASK 15: TRAFFIC WAITING TIME ESTIMATION")
print("=" * 60)

VIDEO_PATH = "traffic_signal_video.mp4"
MODEL_PATH = "yolov8n.pt"

# Detection settings
MOVEMENT_THRESHOLD = 8
CONFIRM_STOP_SECONDS = 1.0

# Display settings
DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720

# Load YOLO model
model = YOLO(MODEL_PATH)
print("\n[SUCCESS] YOLO model loaded!")

# Open traffic video
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("[ERROR] Cannot open traffic video.")
    exit()

print("[SUCCESS] Traffic video opened!")

fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    fps = 30

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print("\nVideo Information:")
print("Width:", width)
print("Height:", height)
print("FPS:", round(fps, 2))

# Vehicle classes
vehicle_classes = [2, 3, 5, 7]

class_names = {
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck"
}

# Vehicle tracking information
vehicles = {}

frame_count = 0
total_detections = 0

confirm_frames = max(1, int(fps * CONFIRM_STOP_SECONDS))

print("\nStarting traffic waiting time estimation...")
print("Tracking vehicles using ByteTrack.")
print("Press Q to stop detection.")

while True:

    ret, frame = cap.read()

    if not ret:
        print("\nVideo playback completed.")
        break

    frame_count += 1

    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        classes=vehicle_classes,
        conf=0.35,
        verbose=False
    )

    result = results[0]

    # Store annotations for display
    annotations = []

    if result.boxes is not None and result.boxes.id is not None:

        boxes = result.boxes.xyxy.cpu().numpy()
        ids = result.boxes.id.int().cpu().tolist()
        classes = result.boxes.cls.int().cpu().tolist()

        total_detections += len(ids)

        for box, track_id, class_id in zip(boxes, ids, classes):

            x1, y1, x2, y2 = map(int, box)

            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            current_center = (center_x, center_y)

            if track_id not in vehicles:

                vehicles[track_id] = {
                    "previous_center": current_center,
                    "stopped_since": None,
                    "is_stopped": False,
                    "max_wait_frames": 0,
                    "total_wait_frames": 0,
                    "stop_events": 0
                }

            else:

                vehicle = vehicles[track_id]

                previous_center = vehicle["previous_center"]

                movement = (
                    (center_x - previous_center[0]) ** 2 +
                    (center_y - previous_center[1]) ** 2
                ) ** 0.5

                if movement <= MOVEMENT_THRESHOLD:

                    if vehicle["stopped_since"] is None:
                        vehicle["stopped_since"] = frame_count

                    stopped_frames = (
                        frame_count - vehicle["stopped_since"]
                    )

                    if stopped_frames >= confirm_frames:

                        if not vehicle["is_stopped"]:
                            vehicle["is_stopped"] = True
                            vehicle["stop_events"] += 1

                        vehicle["max_wait_frames"] = max(
                            vehicle["max_wait_frames"],
                            stopped_frames
                        )

                else:

                    # Save completed waiting duration
                    if vehicle["is_stopped"]:

                        completed_frames = (
                            frame_count - vehicle["stopped_since"]
                        )

                        vehicle["total_wait_frames"] += completed_frames

                        vehicle["max_wait_frames"] = max(
                            vehicle["max_wait_frames"],
                            completed_frames
                        )

                    vehicle["stopped_since"] = None
                    vehicle["is_stopped"] = False

                vehicle["previous_center"] = current_center

            vehicle = vehicles[track_id]

            if vehicle["is_stopped"]:

                wait_frames = frame_count - vehicle["stopped_since"]
                wait_time = wait_frames / fps

                label = f"ID:{track_id} WAIT: {wait_time:.1f}s"
                color = (0, 0, 255)

            else:

                label = f"ID:{track_id} MOVING"
                color = (0, 255, 0)

            vehicle_name = class_names.get(class_id, "Vehicle")

            # Scale bounding box coordinates for display
            sx = DISPLAY_WIDTH / width
            sy = DISPLAY_HEIGHT / height

            bx1 = int(x1 * sx)
            by1 = int(y1 * sy)
            bx2 = int(x2 * sx)
            by2 = int(y2 * sy)

            annotations.append(
                (bx1, by1, bx2, by2,
                 f"{vehicle_name} {label}", color)
            )

    # Count currently stopped vehicles
    stopped_count = sum(
        1 for v in vehicles.values() if v["is_stopped"]
    )

    # Resize frame before drawing display elements
    display_frame = cv2.resize(
        frame,
        (DISPLAY_WIDTH, DISPLAY_HEIGHT)
    )

    # Draw vehicle bounding boxes and labels
    for bx1, by1, bx2, by2, label, color in annotations:

        cv2.rectangle(
            display_frame,
            (bx1, by1),
            (bx2, by2),
            color,
            3
        )

        cv2.putText(
            display_frame,
            label,
            (bx1, max(by1 - 12, 25)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            color,
            2,
            cv2.LINE_AA
        )

    # Large information panel
    cv2.rectangle(
        display_frame,
        (10, 10),
        (650, 115),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        display_frame,
        f"STOPPED VEHICLES: {stopped_count}",
        (25, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 255),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        display_frame,
        f"TOTAL TRACKED IDS: {len(vehicles)}",
        (25, 95),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    cv2.imshow(
        "VisionPort - Traffic Waiting Time",
        display_frame
    )

    # Terminal progress
    if frame_count % 30 == 0:

        print(
            f"Frame: {frame_count} | "
            f"Stopped Vehicles: {stopped_count} | "
            f"Tracked Vehicles: {len(vehicles)}"
        )

    if cv2.waitKey(1) & 0xFF == ord('q'):

        print("\nProcessing stopped by user.")
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

# Final report
print("\n" + "=" * 60)
print("TASK 15 FINAL RESULT")
print("=" * 60)

print("Total frames processed:", frame_count)
print("Total tracking detections:", total_detections)
print("Total unique tracked vehicles:", len(vehicles))

stopped_vehicles = 0
total_waiting_seconds = 0
total_stop_events = 0

print("\nVEHICLE WAITING TIME REPORT:")

for track_id, vehicle in vehicles.items():

    completed_frames = vehicle["total_wait_frames"]

    # Include waiting time if vehicle is still stopped
    if vehicle["is_stopped"]:

        current_frames = (
            frame_count - vehicle["stopped_since"]
        )

    else:
        current_frames = 0

    total_frames = completed_frames + current_frames

    max_frames = max(
        vehicle["max_wait_frames"],
        current_frames
    )

    if vehicle["stop_events"] > 0:

        stopped_vehicles += 1
        total_stop_events += vehicle["stop_events"]

        total_wait = total_frames / fps
        max_wait = max_frames / fps

        total_waiting_seconds += total_wait

        average_wait = (
            total_wait / vehicle["stop_events"]
        )

        print(
            f"Vehicle ID: {track_id} | "
            f"Stop Events: {vehicle['stop_events']} | "
            f"Total Wait: {total_wait:.2f}s | "
            f"Maximum Wait: {max_wait:.2f}s | "
            f"Average Wait: {average_wait:.2f}s"
        )

print("\nOVERALL WAITING TIME SUMMARY:")

print("Vehicles that stopped:", stopped_vehicles)
print("Total stopping events:", total_stop_events)

if stopped_vehicles > 0:

    print(
        "Average waiting time per stopped vehicle:",
        round(total_waiting_seconds / stopped_vehicles, 2),
        "seconds"
    )

else:
    print("No confirmed vehicle stops detected.")

print("\nTASK 15 RESULT: PROCESSING COMPLETED")
print("Traffic waiting time estimation completed.")
print("=" * 60)
