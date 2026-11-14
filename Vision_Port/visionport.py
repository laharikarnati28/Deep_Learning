
import io
import cv2
import numpy as np

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from ultralytics import YOLO


# 1. Initialize FastAPI
app = FastAPI(title="Indian Traffic Detection API")


# 2. Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5500",
        "http://127.0.0.1:5500"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 3. Load YOLOv8 Nano model
print("Loading YOLOv8 model...")
model = YOLO("yolov8n.pt")
print("YOLOv8 model loaded successfully!")


# 4. Image detection endpoint
@app.post("/detect/")
async def detect_objects(file: UploadFile = File(...)):

    # Read uploaded image
    contents = await file.read()

    # Convert image bytes into OpenCV format
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return {"error": "Invalid image file"}

    # Run YOLO object detection
    results = model(img)

    # Draw bounding boxes and labels
    annotated_img = results[0].plot()

    # Convert processed image into JPEG
    success, encoded_img = cv2.imencode(
        ".jpg", annotated_img
    )

    if not success:
        return {"error": "Image encoding failed"}

    # Return processed image to frontend
    return StreamingResponse(
        io.BytesIO(encoded_img.tobytes()),
        media_type="image/jpeg"
    )


# 5. Run the backend server
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000
    )
