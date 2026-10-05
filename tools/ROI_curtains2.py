import cv2
import numpy as np
import csv

# ------------------------------------------------------------
# This script analyzes a prerecorded video and determines
# whether each court curtain is open or closed over time.
#
# It does this by comparing each ROI (Region of Interest)
# in the current frame to the same ROI in a reference image.
# ------------------------------------------------------------

video_path = "short_video.mp4"

# Reference image should be a frame where ALL curtains are open.
# This is used as the baseline to compare against.
reference_image = "..frames/multi.png"

# Open the video file
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise RuntimeError("Cannot open video")

# Get video metadata
fps = int(cap.get(cv2.CAP_PROP_FPS))    # frames per second
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))   # total number of frames 

# ------------------------------------------------------------
# ROI definitions:
# Each ROI is a rectangle (x, y, width, height)
# These coordinates correspond to the curtain areas in the frame.
# ------

rois = [
    ((144, 108, 37, 84)),  # court 1
    (286, 117, 272, 242),  # court 2
    (1328, 208, 347, 245), # court 3
    (1752, 123, 40, 66) # court 4
]

# Load the reference image and convert to grayscale
ref = cv2.imread(reference_image)
ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)

# ------------------------------------------------------------
# Output CSV:
# time_sec = timestamp in seconds from start of video
# court1..court4 = 1=open, 0=closed
# ------------------------------------------------------------
with open("courts_state.csv", "w", newline="") as f:

    writer = csv.writer(f)
    writer.writerow(["time_sec","court1","court2","court3","court4"])

    frame_number = 0

    # How often to check curtain state (in seconds)
    check_interval = 5

    # ------------------------------------------------------------
    # Main loop: iterate through video at fixed time intervals
    # ------------------------------------------------------------
    while frame_number < total_frames:

        # Jump to the desired frame number
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

        ret, frame = cap.read()
        if not ret:
            break

        # Convert current frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        states = [] # stores open/closed state for each court

        # ------------------------------------------------------------
        # For each ROI:
        #   - Extract the same region from current frame and reference frame
        #   - Compute absolute difference
        #   - If difference is large → curtain is closed
        #   - If difference is small → curtain is open
        # ------------------------------------------------------------
        for i,(x,y,w,h) in enumerate(rois):

            roi_current = gray[y:y+h, x:x+w]
            roi_ref = ref_gray[y:y+h, x:x+w]

            diff = cv2.absdiff(roi_current, roi_ref)

            score = np.mean(diff)

            # Threshold determines open vs closed
            if score > 25:
                state = 0   # curtain closed
            else:
                state = 1   # curtain open

            states.append(state)

        # Convert frame number to seconds
        time_sec = frame_number / fps
        
        # Write row to CSV
        writer.writerow([int(time_sec)] + states)

        # Move forward by N seconds worth of frames
        frame_number += fps * check_interval

cap.release()

print("Processing complete.")