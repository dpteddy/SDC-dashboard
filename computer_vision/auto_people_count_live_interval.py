#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author dragon 
@author Noah
This script:
    - Connects to the live SDC camera stream
    - Runs YOLO every CAPTURE_INTERVAL seconds
    - Counts people
    - Logs bounding boxes + total counts to CSV
    - Runs only during scheduled hours
    - Does NOT record video (privacy-safe)
    - Does NOT detect curtains
    - Does NOT assign people to courts (yet)

This is the simplest and safest production script.
"""

import cv2
import os
import time
import csv
from datetime import datetime
from ultralytics import YOLO

# ================= SETTINGS =================

MODEL_PATH = "yolov11s.pt"   # YOLO model file
CAPTURE_INTERVAL = 5         # seconds between YOLO detections

# CSV output files
BOXES_CSV = "people_boxes.csv"   # bounding box logs
TOTAL_CSV = "people_total.csv"   # total people count per interval

# Live SDC camera stream
STREAM_URL = "https://streamingwebcams.mtu.edu:1935/rtplive/camera004.stream/playlist.m3u8"

# Days of the week to record (0=Monday, 6=Sunday)
record_days = [0, 1, 2, 3, 4, 5, 6]

# Recording window (24-hour time)
start_hour, start_min = 16, 35   # 4:35 PM
end_hour, end_min = 18, 35       # 6:35 PM

# ================= CSV SETUP =================
# Create CSV files if they do not already exist.

if not os.path.exists(BOXES_CSV):
    with open(BOXES_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "x1", "y1", "x2", "y2"])

if not os.path.exists(TOTAL_CSV):
    with open(TOTAL_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "total_people"])

# ================= SCHEDULING =================

def within_recording_time():
    """
    Returns True if the current time is within the allowed
    recording window AND today is one of the allowed days.
    """
    now = datetime.now()

    # Check day of week
    if now.weekday() in record_days:

        # Convert time to minutes for easy comparison
        current_minutes = now.hour * 60 + now.minute
        start_minutes = start_hour * 60 + start_min
        end_minutes = end_hour * 60 + end_min

        return start_minutes <= current_minutes < end_minutes

    return False

# ================= PEOPLE COUNTING =================

def run_people_counting():
    """
    Main loop:
      - Connect to camera stream
      - Run YOLO every CAPTURE_INTERVAL seconds
      - Count people
      - Write bounding boxes + total count to CSV
      - Display annotated frame (optional)
    """

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

            # Run YOLO only every CAPTURE_INTERVAL seconds
            if current_time - last_capture_time >= CAPTURE_INTERVAL:

                # Run YOLO
                results = model(frame, conf=0.4, imgsz=1280, verbose=False)

                # Update capture time AFTER YOLO finishes
                last_capture_time = time.time()

                # Generate YOLO-annotated frame for display
                annotated = results[0].plot()
                cv2.imshow("People Counter", annotated)

                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                count = 0
                boxes_to_write = []

                # Loop through YOLO detections
                for box in results[0].boxes:

                    class_id = int(box.cls.item())

                    # Only count people
                    if model.names[class_id] == "person":
                        x1, y1, x2, y2 = box.xyxy[0]

                        # Save bounding box
                        boxes_to_write.append([
                            timestamp,
                            float(x1),
                            float(y1),
                            float(x2),
                            float(y2)
                        ])

                        count += 1

                # Write bounding boxes
                if boxes_to_write:
                    with open(BOXES_CSV, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerows(boxes_to_write)

                # Write total people count
                with open(TOTAL_CSV, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([timestamp, count])

                print(f"[{timestamp}] Detected people: {count}")

            else:
                # Show last annotated frame or raw frame
                if annotated is not None:
                    cv2.imshow("People Counter", annotated)
                else:
                    cv2.imshow("People Counter", frame)

            cv2.waitKey(1)

    except KeyboardInterrupt:
        raise  # allow main loop to handle shutdown

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"Stopped monitoring at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ================= MAIN CONTROL =================

def main():
    """
    Continuously checks whether it is recording time.
    If yes → run YOLO monitoring loop.
    If no → sleep and check again.
    """

    print("Script started. Monitoring schedule...")
    print(f"Will run {start_hour}:{start_min:02d} AM - {end_hour}:{end_min:02d} PM")

    try:
        while True:
            if within_recording_time():
                run_people_counting()
            else:
                print(f"Outside recording time. Sleeping... ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
                time.sleep(60)  # check again every minute

    except KeyboardInterrupt:
        print("Script manually stopped.")

if __name__ == "__main__":
    main()
