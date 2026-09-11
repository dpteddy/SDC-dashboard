import cv2
import numpy as np
import csv

video_path = "short_video.mp4"
reference_image = "frames/multi.png"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise RuntimeError("Cannot open video")

fps = int(cap.get(cv2.CAP_PROP_FPS))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# ROI (вставь свои координаты)
rois = [
    ((144, 108, 37, 84)),  # court 1
    (286, 117, 272, 242),  # court 2
    (1328, 208, 347, 245), # court 3
    (1752, 123, 40, 66) # court 4
]
# загрузить reference кадр
ref = cv2.imread(reference_image)
ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)

with open("courts_state.csv", "w", newline="") as f:

    writer = csv.writer(f)
    writer.writerow(["time_sec","court1","court2","court3","court4"])

    frame_number = 0
    check_interval = 5  # секунд

    while frame_number < total_frames:

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        states = []

        for i,(x,y,w,h) in enumerate(rois):

            roi_current = gray[y:y+h, x:x+w]
            roi_ref = ref_gray[y:y+h, x:x+w]

            diff = cv2.absdiff(roi_current, roi_ref)

            score = np.mean(diff)

            if score > 25:
                state = 0   # curtain closed
            else:
                state = 1   # curtain open

            states.append(state)

        time_sec = frame_number / fps
        writer.writerow([int(time_sec)] + states)

        frame_number += fps * check_interval

cap.release()

print("Processing complete.")