

import cv2
import numpy as np
from collections import deque, Counter

print("=" * 60)
print("VISIONPORT - TASK 14: TRAFFIC SIGNAL STATUS DETECTION")
print("=" * 60)

VIDEO_PATH = "traffic_signal_video.mp4"

# Detection settings
MIN_PIXELS = 20
MIN_COLOR_RATIO = 0.005
DOMINANCE_RATIO = 1.25
SMOOTHING_FRAMES = 5

# Open video
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("[ERROR] Cannot open traffic signal video.")
    exit()

print("[SUCCESS] Traffic signal video opened!")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print("\nVideo Information:")
print("Width:", width)
print("Height:", height)
print("FPS:", round(fps, 2))

# Read first frame
ret, first_frame = cap.read()

if not ret:
    print("[ERROR] Cannot read video.")
    cap.release()
    exit()

# Resize preview
scale = min(960 / width, 720 / height, 1.0)

preview = cv2.resize(
    first_frame,
    (int(width * scale), int(height * scale))
)

print("\nSelect ONLY the traffic light area.")
print("Avoid including cars, roads and buildings.")
print("Press ENTER or SPACE to confirm.")

roi = cv2.selectROI(
    "Select Traffic Signal",
    preview,
    False,
    False
)

cv2.destroyWindow("Select Traffic Signal")

rx, ry, rw, rh = roi

if rw == 0 or rh == 0:
    print("[ERROR] No region selected.")
    cap.release()
    exit()

# Convert ROI to original coordinates
rx = int(rx / scale)
ry = int(ry / scale)
rw = int(rw / scale)
rh = int(rh / scale)

print("[SUCCESS] Traffic signal region selected!")

# HSV colour ranges
lower_red1 = np.array([0, 100, 100])
upper_red1 = np.array([10, 255, 255])

lower_red2 = np.array([160, 100, 100])
upper_red2 = np.array([180, 255, 255])

lower_yellow = np.array([15, 100, 100])
upper_yellow = np.array([35, 255, 255])

lower_green = np.array([36, 70, 70])
upper_green = np.array([90, 255, 255])

status_counts = {
    "RED": 0,
    "YELLOW": 0,
    "GREEN": 0,
    "UNKNOWN": 0
}

frame_count = 0

# Store recent predictions
recent_statuses = deque(maxlen=SMOOTHING_FRAMES)

colors = {
    "RED": (0, 0, 255),
    "YELLOW": (0, 255, 255),
    "GREEN": (0, 255, 0),
    "UNKNOWN": (255, 255, 255)
}


def detect_signal_color(signal_roi):

    hsv = cv2.cvtColor(
        signal_roi,
        cv2.COLOR_BGR2HSV
    )

    # Red masks
    red1 = cv2.inRange(
        hsv, lower_red1, upper_red1
    )

    red2 = cv2.inRange(
        hsv, lower_red2, upper_red2
    )

    red_mask = cv2.bitwise_or(red1, red2)

    # Yellow mask
    yellow_mask = cv2.inRange(
        hsv, lower_yellow, upper_yellow
    )

    # Green mask
    green_mask = cv2.inRange(
        hsv, lower_green, upper_green
    )

    # Count matching pixels
    pixels = {
        "RED": cv2.countNonZero(red_mask),
        "YELLOW": cv2.countNonZero(yellow_mask),
        "GREEN": cv2.countNonZero(green_mask)
    }

    roi_area = signal_roi.shape[0] * signal_roi.shape[1]

    # Find highest colour count
    sorted_colors = sorted(
        pixels.items(),
        key=lambda item: item[1],
        reverse=True
    )

    best_color, best_pixels = sorted_colors[0]
    second_pixels = sorted_colors[1][1]

    # Calculate matching colour percentage
    color_percentage = (best_pixels / roi_area) * 100

    # Reject weak detections
    minimum_required = max(
        MIN_PIXELS,
        int(roi_area * MIN_COLOR_RATIO)
    )

    if best_pixels < minimum_required:
        return "UNKNOWN", color_percentage, pixels

    # Check colour dominance
    if second_pixels > 0:
        if best_pixels < second_pixels * DOMINANCE_RATIO:
            return "UNKNOWN", color_percentage, pixels

    return best_color, color_percentage, pixels


print("\nStarting improved traffic signal detection...")
print("Press Q to stop detection.")

# Restart video from beginning
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

while True:

    ret, frame = cap.read()

    if not ret:
        print("\nVideo playback completed.")
        break

    frame_count += 1

    # Extract selected region
    signal_roi = frame[ry:ry+rh, rx:rx+rw]

    if signal_roi.size == 0:
        print("[ERROR] Invalid selected region.")
        break

    # Detect colour
    raw_status, percentage, pixel_counts = detect_signal_color(
        signal_roi
    )

    # Temporal smoothing
    recent_statuses.append(raw_status)

    status_frequency = Counter(recent_statuses)

    detected_status, frequency = status_frequency.most_common(1)[0]

    # Require majority agreement
    if frequency <= len(recent_statuses) // 2:
        detected_status = "UNKNOWN"

    status_counts[detected_status] += 1

    display_color = colors[detected_status]

    # Draw ROI
    cv2.rectangle(
        frame,
        (rx, ry),
        (rx + rw, ry + rh),
        (255, 0, 255),
        3
    )

    # Display information panel
    cv2.rectangle(
        frame,
        (10, 10),
        (700, 160),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        frame,
        "TRAFFIC SIGNAL: " + detected_status,
        (25, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        display_color,
        3
    )

    cv2.putText(
        frame,
        f"Color pixels: {percentage:.2f}%",
        (25, 95),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Frame: {frame_count}",
        (25, 135),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    # Resize for display
    display_frame = cv2.resize(
        frame,
        (int(width * scale), int(height * scale))
    )

    cv2.imshow(
        "VisionPort - Improved Traffic Signal Detection",
        display_frame
    )

    if frame_count % 30 == 0:
        print(
            f"Frame: {frame_count} | "
            f"Status: {detected_status} | "
            f"Colour percentage: {percentage:.2f}%"
        )

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("\nDetection stopped by user.")
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

# Final report
print("\n" + "=" * 60)
print("TASK 14 FINAL RESULT")
print("=" * 60)

print("Total frames processed:", frame_count)

print("\nTraffic Signal Status Counts:")

for status, count in status_counts.items():
    print(f"{status}: {count}")

print("\nTASK 14 RESULT: PROCESSING COMPLETED")
print("Improved traffic signal colour analysis completed.")
print("=" * 60)
