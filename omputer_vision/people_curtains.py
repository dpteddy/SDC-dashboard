#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import os
import time
import csv
import numpy as np
from datetime import datetime
from ultralytics import YOLO

# ================= SETTINGS =================

MODEL_PATH = "yolov8s.pt"
STREAM_URL = "https://streamingwebcams.mtu.edu:1935/rtplive/camera004.stream/playlist.m3u8"

CAPTURE_INTERVAL = 5  # seconds

BOXES_CSV = "people_boxes.csv"
TOTAL_CSV = "people_total.csv"
CURTAIN_CSV = "courts_state.csv"

REFERENCE_IMAGE = "frames/multi.png"

# ROI coordinates (x, y, width, height)
rois = [
    (144,108,37,84),
    (286,117,272,242),
    (1328,208,347,245),
    (1752,123,40,66)
]

record_days = [0,1,2,3,4,5,6]

start_hour,start_min = 6,00
end_hour,end_min = 22,30

DIFF_THRESHOLD = 25

# ================= CSV SETUP =================

if not os.path.exists(BOXES_CSV):
    with open(BOXES_CSV,"w",newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp","x1","y1","x2","y2"])

if not os.path.exists(TOTAL_CSV):
    with open(TOTAL_CSV,"w",newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp","total_people"])

if not os.path.exists(CURTAIN_CSV):
    with open(CURTAIN_CSV,"w",newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp","court1","court2","court3","court4"])

# ================= TIME CHECK =================

def within_recording_time():

    now = datetime.now()

    if now.weekday() in record_days:

        current_minutes = now.hour*60 + now.minute
        start_minutes = start_hour*60 + start_min
        end_minutes = end_hour*60 + end_min

        return start_minutes <= current_minutes < end_minutes

    return False

# ================= MAIN PROCESS =================

def run_monitoring():

    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(STREAM_URL)

    if not cap.isOpened():
        print("Cannot open stream")
        return

    # load reference image
    ref = cv2.imread(REFERENCE_IMAGE)
    ref_gray = cv2.cvtColor(ref,cv2.COLOR_BGR2GRAY)

    print("Monitoring started")

    last_capture_time = 0

    try:

        while within_recording_time():

            ret,frame = cap.read()

            if not ret:
                print("Frame grab failed")
                continue

            current_time = time.time()

            if current_time - last_capture_time >= CAPTURE_INTERVAL:

                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                # ================= PEOPLE DETECTION =================

                results = model(frame,conf=0.4,imgsz=1280,verbose=False)

                count = 0
                boxes_to_write = []

                for box in results[0].boxes:

                    class_id = int(box.cls.item())

                    if model.names[class_id] == "person":

                        x1,y1,x2,y2 = box.xyxy[0]

                        boxes_to_write.append([
                            timestamp,
                            float(x1),
                            float(y1),
                            float(x2),
                            float(y2)
                        ])

                        count += 1

                if boxes_to_write:

                    with open(BOXES_CSV,"a",newline="") as f:
                        writer = csv.writer(f)
                        writer.writerows(boxes_to_write)

                with open(TOTAL_CSV,"a",newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([timestamp,count])

                # ================= CURTAIN DETECTION =================

                gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

                states = []

                for (x,y,w,h) in rois:

                    roi_current = gray[y:y+h,x:x+w]
                    roi_ref = ref_gray[y:y+h,x:x+w]

                    diff = cv2.absdiff(roi_current,roi_ref)

                    score = np.mean(diff)

                    if score > DIFF_THRESHOLD:
                        state = 0
                    else:
                        state = 1

                    states.append(state)

                with open(CURTAIN_CSV,"a",newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([timestamp] + states)

                print(f"[{timestamp}] People: {count} | Curtains: {states}")

                last_capture_time = time.time()

            cv2.waitKey(1)

    except KeyboardInterrupt:

        print("Stopped manually")

    finally:

        cap.release()
        cv2.destroyAllWindows()

# ================= MAIN LOOP =================

def main():

    print("Script running. Waiting for recording time.")

    while True:

        if within_recording_time():

            run_monitoring()

        else:

            print("Outside recording hours")
            time.sleep(60)

if __name__ == "__main__":
    main()