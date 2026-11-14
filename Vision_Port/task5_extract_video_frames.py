
"""
VisionPort - Task 5
Extract Individual Frames from Traffic Videos

Expected Output:
Frames saved as images.
"""

import cv2
import os

print("=" * 60)
print("VISIONPORT - TASK 5: EXTRACT VIDEO FRAMES")
print("=" * 60)

# 1. Specify input video
video_path = "traffic_sample.mp4"

# 2. Specify output folder
output_folder = "extracted_frames"

# 3. Check whether video exists
if not os.path.exists(video_path):
    print("[ERROR] Video file not found!")
    exit()

print("\n[SUCCESS] Video file found!")

# 4. Create output folder if it does not exist
os.makedirs(output_folder, exist_ok=True)

print("[SUCCESS] Output folder ready!")

# 5. Open video
video = cv2.VideoCapture(video_path)

if not video.isOpened():
    print("[ERROR] Unable to open video!")
    exit()

print("[SUCCESS] Video opened successfully!")

# 6. Get video properties
fps = video.get(cv2.CAP_PROP_FPS)
total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

print("\nVideo Information:")
print("FPS:", round(fps, 2))
print("Total Frames:", total_frames)

# 7. Extract approximately one frame per second
frame_interval = max(1, round(fps))

frame_count = 0
saved_count = 0

print("\nExtracting frames...")

while True:

    ret, frame = video.read()

    if not ret:
        break

    # Save one frame approximately every second
    if frame_count % frame_interval == 0:

        filename = f"frame_{saved_count:04d}.jpg"

        image_path = os.path.join(output_folder, filename)

        success = cv2.imwrite(image_path, frame)

        if success:
            saved_count += 1
            print("[SAVED]", filename)

    frame_count += 1

# 8. Release video
video.release()

# 9. Display final result
print("\n" + "=" * 60)
print("TASK 5 FINAL RESULT")
print("=" * 60)

print("Total video frames processed:", frame_count)
print("Total images extracted:", saved_count)
print("Images saved inside:", output_folder)

if saved_count > 0:
    print("\nTASK 5 RESULT: PASSED")
    print("Video frames extracted and saved successfully.")
else:
    print("\nTASK 5 RESULT: FAILED")
    print("No frames were extracted.")

print("=" * 60)
