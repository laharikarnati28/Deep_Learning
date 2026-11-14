import cv2
from ultralytics import YOLO

model = YOLO("yolo11n.pt")

video = cv2.VideoCapture("traffic_sample.mp4")

while video.isOpened():

    ret, frame = video.read()

    if not ret:
        break

    results = model(
        frame,
        classes=[2, 3, 5, 7],
        verbose=False
    )

    for result in results:

        for box in result.boxes:

            class_id = int(box.cls[0])

            confidence = float(box.conf[0])

            name = model.names[class_id]

            print(
                name,
                round(confidence * 100, 2),
                "%"
            )

    output = results[0].plot()

    cv2.imshow("Vehicle Labels and Confidence", output)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()