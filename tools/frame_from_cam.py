import cv2
import os
from datetime import datetime

stream_url = "https://streamingwebcams.mtu.edu:1935/rtplive/camera004.stream/playlist.m3u8"

cap = cv2.VideoCapture(stream_url)

if not cap.isOpened():
    raise RuntimeError("Failed to open the stream")

# Sometimes the first frame is empty — we read a few.
frame = None
for _ in range(30):
    ret, frame = cap.read()
    if ret:
        break

cap.release()

if frame is None:
    raise RuntimeError("Failed to retrieve the frame count.")

os.makedirs("frames", exist_ok=True)
filename = f"frames/camera004_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png"
cv2.imwrite(filename, frame)

print("Saved:", filename)
