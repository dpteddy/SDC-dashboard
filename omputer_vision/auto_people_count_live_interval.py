#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 24 16:09:29 2026

@author: draughon
@author: noah zabinski
This program is meant to automatically run during the set days and keep count of the number of people it detects. 
The campus activity project will be using this in the SDC to properly get data during the working hours. 
"""

import cv2
import os
import time
import csv
from datetime import datetime
from ultralytics import YOLO

# === SETTINGS ===
MODEL_PATH = "yolov8s.pt"
CAPTURE_INTERVAL = 5  # seconds between captures (adjust as needed)
BOXES_CSV = "people_boxes.csv"
TOTAL_CSV = "people_total.csv"
STREAM_URL = "https://streamingwebcams.mtu.edu:1935/rtplive/camera004.stream/playlist.m3u8"

record_days = [0, 1, 2, 3, 4, 5, 6]  # 0=Monday, 6=Sunday
start_hour, start_min = 16, 35    # 6 = 6:00 AM
end_hour, end_min = 18, 35       # 22 = 10:00 PM

# === SETUP CSV FILES (only write headers if file doesn't exist) ===
if not os.path.exists(BOXES_CSV):
    with open(BOXES_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "x1", "y1", "x2", "y2"])

if not os.path.exists(TOTAL_CSV):
    with open(TOTAL_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "total_people"])

# === SCHEDULING ===
def within_recording_time():
    now = datetime.now()
    if now.weekday() in record_days:
        current_minutes = now.hour * 60 + now.minute
        start_minutes = start_hour * 60 + start_min
        end_minutes = end_hour * 60 + end_min
        return start_minutes <= current_minutes < end_minutes
    return False

# === PEOPLE COUNTING ===
def run_people_counting():
    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(STREAM_URL)

    if not cap.isOpened():
        print("Error: Cannot open stream.")
        return

    print(f"Started monitoring at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    last_capture_time = 0
    annotated = None

    try:
        while within_recording_time():
            ret, frame = cap.read()
            if not ret:
                print("Frame grab failed, retrying...")
                continue

            current_time = time.time()

            if current_time - last_capture_time >= CAPTURE_INTERVAL:

                results = model(frame, conf=0.4, imgsz=1280, verbose=False)
                last_capture_time = time.time()  # Set AFTER model runs

                annotated = results[0].plot()
                cv2.imshow("People Counter", annotated)

                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                count = 0
                boxes_to_write = []

                for box in results[0].boxes:
                    class_id = int(box.cls.item())
                    if model.names[class_id] == "person":
                        x1, y1, x2, y2 = box.xyxy[0]
                        boxes_to_write.append([
                            timestamp,
                            float(x1),
                            float(y1),
                            float(x2),
                            float(y2)
                        ])
                        count += 1

                # Record detected boxes
                if boxes_to_write:
                    with open(BOXES_CSV, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerows(boxes_to_write)

                # Record total count
                with open(TOTAL_CSV, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([timestamp, count])

                print(f"[{timestamp}] Detected people: {count}")

            else:
                if annotated is not None:
                    cv2.imshow("People Counter", annotated)
                else:
                    cv2.imshow("People Counter", frame)

            cv2.waitKey(1)

    except KeyboardInterrupt:
        raise  # Pass it up to the main loop to handle

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"Stopped monitoring at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# === MAIN CONTROL ===
def main():
    print("Script started. Monitoring schedule...")
    print(f"Will run {start_hour}:{start_min:02d} AM - {end_hour}:{end_min:02d} PM")

    try:
        while True:
            if within_recording_time():
                run_people_counting()
            else:
                print(f"Outside recording time. Sleeping... ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
                time.sleep(60)  # Check again every minute

    except KeyboardInterrupt:
        print("Script manually stopped.")

if __name__ == "__main__":
    main()