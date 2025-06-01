import cv2
import base64
import requests
import time
from fastapi import FastAPI
from pydantic import BaseModel
app = FastAPI()

class CaptionRequest(BaseModel):
    trigger: str

@app.post("/caption")
def describe_camera(req: CaptionRequest):
    if req.trigger != "describe_camera":
        return {"status": "ignored"}

    # 1. Capture webcam frame
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return {"status": "error", "message": "Webcam capture failed"}

    _, buffer = cv2.imencode('.jpg', frame)
    base64_image = base64.b64encode(buffer).decode("utf-8")

    # 2. Prepare payload for SmolVLM
    payload = {
        "max_tokens": 100,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What do you see?"},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }}
                ]
            }
        ]
    }

    # 3. Send request to local SmolVLM API
    try:
        response = requests.post("http://localhost:8080/v1/chat/completions", json=payload)
        if response.ok:
            caption = response.json()["choices"][0]["message"]["content"]
            return {"status": "ok", "caption": caption}
        else:
            return {"status": "error", "message": f"SmolVLM failed: {response.text}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}
