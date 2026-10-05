import cv2
import os
from datetime import datetime


# ------------------------------------------------------------
# This script captures a single frame from the live SDC webcam.
#
# It is mainly used to:
#   - grab a reference frame for curtain detection
#   - grab a frame to manually inspect ROI coordinates
#   - grab a frame to verify camera alignment
#
# It does NOT run YOLO, does NOT detect people, and does NOT
# perform any analysis. It simply saves one image.
# ------------------------------------------------------------

# URL of the live stream from the SDC camera
stream_url = "https://streamingwebcams.mtu.edu:1935/rtplive/camera004.stream/playlist.m3u8"

# Open the video stream
cap = cv2.VideoCapture(stream_url)

if not cap.isOpened():
    raise RuntimeError("Failed to open the stream")

# ------------------------------------------------------------
# Sometimes the first few frames from a network stream are empty
# or corrupted. To avoid saving a blank frame, we read up to 30
# frames and stop as soon as we get a valid one.
# ------------------------------------------------------------
frame = None
for _ in range(30):
    ret, frame = cap.read()
    if ret:     # ret=True means a valid frame was captured
        break

# Close the stream
cap.release()

# If we still don't have a valid frame, something is wrong
if frame is None:
    raise RuntimeError("Failed to retrieve the frame count.")

# ------------------------------------------------------------
# Save the captured frame to the "frames" folder.
# The filename includes a timestamp so each capture is unique.
# ------------------------------------------------------------
os.makedirs("frames", exist_ok=True)

filename = f"frames/camera004_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png"

cv2.imwrite(filename, frame)

print("Saved:", filename)
