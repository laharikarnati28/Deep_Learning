
"""
VisionPort - Task 4
Load and Play a Traffic Video Using OpenCV

Expected Output:
Video playback
"""

import cv2
import os

print("=" * 60)
print("VISIONPORT - TASK 4: TRAFFIC VIDEO PLAYBACK")
print("=" * 60)

# 1. Specify video file
video_path = "traffic_sample.mp4"

# 2. Check whether video exists
if not os.path.exists(video_path):
    print("[ERROR] Video file not found!")
    print("Place traffic_sample.mp4 inside the project folder.")
    exit()

print("\n[SUCCESS] Video file found!")

# 3. Open the video
video = cv2.VideoCapture(video_path)

# 4. Check whether video opened
if not video.isOpened():
    print("[ERROR] Unable to open video.")
    exit()

print("[SUCCESS] Video opened successfully!")

# 5. Get video properties
fps = video.get(cv2.CAP_PROP_FPS)
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

duration = total_frames / fps if fps > 0 else 0

print("\nVideo Information:")
print("Width:", width, "pixels")
print("Height:", height, "pixels")
print("FPS:", fps)
print("Total Frames:", total_frames)
print("Duration:", round(duration, 2), "seconds")

print("\nPlaying traffic video...")
print("Press Q to stop playback.")

# 6. Calculate playback delay
delay = max(1, int(1000 / fps)) if fps > 0 else 33

frame_count = 0

# 7. Read and display video frames
while True:

    ret, frame = video.read()

    if not ret:
        print("\n[SUCCESS] Video playback completed!")
        break

    frame_count += 1

    cv2.imshow("VisionPort - Traffic Video Playback", frame)

    # 8. Check keyboard input
    key = cv2.waitKey(delay) & 0xFF

    if key == ord('q'):
        print("\nPlayback stopped by user.")
        break

# 9. Release resources
video.release()
cv2.destroyAllWindows()

# 10. Final result
print("\n" + "=" * 60)
print("TASK 4 RESULT: PASSED")
print("Video loaded and played successfully.")
print("Frames displayed:", frame_count)
print("=" * 60)
