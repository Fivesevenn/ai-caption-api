# caption_api.py
import base64
import cv2
from fastapi import FastAPI
from pydantic import BaseModel
import requests

app = FastAPI()

# Constants
API_URL = "http://localhost:8080/v1/chat/completions"
INSTRUCTION = "What do you see?"

class CaptionRequest(BaseModel):
    trigger: str  # expected to be "describe_camera"

def capture_frame():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError("Webcam capture failed.")
    _, buffer = cv2.imencode('.jpg', frame)
    return base64.b64encode(buffer.tobytes()).decode("utf-8")

@app.post("/caption")
def describe_camera(req: CaptionRequest):
    if req.trigger != "describe_camera":
        return {"status": "ignored"}

    image_base64 = capture_frame()
    payload = {
        "max_tokens": 100,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": INSTRUCTION},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/jpeg;base64,{image_base64}"
                    }}
                ]
            }
        ]
    }

    response = requests.post(API_URL, json=payload)
    if response.ok:
        caption = response.json()["choices"][0]["message"]["content"]
        return {"caption": caption}
    else:
        return {"error": response.text}
