import cv2
import os
import time
import csv
import numpy as np
from datetime import datetime
from ultralytics import YOLO

# ------------------------------------------------------------
# This script:
#   - Connects to the live SDC camera stream
#   - Runs YOLO every CAPTURE_INTERVAL seconds
#   - Counts people
#   - Detects curtain open/closed state
#   - Records BOTH raw video and annotated video
#   - Writes people + curtain data to CSV
#
# This is the MOST FEATURE-COMPLETE script in your project.
# It is NOT privacy-safe because it records video.
# ------------------------------------------------------------

# ================= SETTINGS =================

MODEL_PATH = "yolov11s.pt"   # YOLO model file
STREAM_URL = "https://streamingwebcams.mtu.edu:1935/rtplive/camera004.stream/playlist.m3u8"

CAPTURE_INTERVAL = 5  # seconds between YOLO detections

# CSV output files
BOXES_CSV = "trackPrac_boxes.csv"          # bounding box logs
TOTAL_CSV = "trackPrac_total.csv"          # total people count
CURTAIN_CSV = "trackPrac_courts_state.csv" # curtain open/closed state

# Folder where raw + annotated video recordings are saved
OUTPUT_FOLDER = "trackPrac_recordings"

# Reference image used for curtain comparison
REFERENCE_IMAGE = "frames/multi.png"

# NOTE: OUTPUT_FOLDER is overwritten below — likely a mistake
OUTPUT_FOLDER = "recordings"

# ------------------------------------------------------------
# ROI definitions:
# Each ROI is a rectangle (x, y, width, height)
# These correspond to the curtain areas for each court.
# ------------------------------------------------------------
rois = [
    (144,108,37,84),     # Court 1 curtain region
    (286,117,272,242),   # Court 2 curtain region
    (1328,208,347,245),  # Court 3 curtain region
    (1752,123,40,66)     # Court 4 curtain region
]

# Days of the week to record (0=Monday, 6=Sunday)
record_days = [0,1,2,3,4,5,6]

# Recording window (24-hour time)
start_hour, start_min = 15, 30   # 3:30 PM
end_hour, end_min = 17, 30       # 5:30 PM

# Threshold for curtain detection
DIFF_THRESHOLD = 25

# ================= CSV SETUP =================
# Create CSV files if they do not already exist.

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

# Ensure output folder exists
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ================= TIME CHECK =================

def within_recording_time():
    """
    Returns True if the current time is within the allowed
    recording window AND today is one of the allowed days.
    """
    now = datetime.now()

    if now.weekday() in record_days:
        current_minutes = now.hour * 60 + now.minute
        start_minutes = start_hour * 60 + start_min
        end_minutes = end_hour * 60 + end_min

        return start_minutes <= current_minutes < end_minutes

    return False

# ================= MAIN PROCESS =================

def run_monitoring():
    """
    Main loop:
      - Connect to camera stream
      - Record raw video continuously
      - Run YOLO every CAPTURE_INTERVAL seconds
      - Count people
      - Detect curtain states
      - Write annotated video
      - Write CSV logs
    """

    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(STREAM_URL)

    if not cap.isOpened():
        print("Cannot open stream")
        return

    # Load reference image for curtain comparison
    ref = cv2.imread(REFERENCE_IMAGE)
    ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)

    # ------------------------------------------------------------
    # Get video properties (resolution, FPS)
    # If FPS is unknown, default to 20
    # ------------------------------------------------------------
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1920)
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1080)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps > 0 else 20

    # ------------------------------------------------------------
    # Create timestamped filenames for raw + annotated recordings
    # ------------------------------------------------------------
    timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    raw_path = os.path.join(OUTPUT_FOLDER, f"raw_{timestamp_str}.mp4")
    annotated_path = os.path.join(OUTPUT_FOLDER, f"annotated_{timestamp_str}.mp4")

    # ------------------------------------------------------------
    # Setup video writers
    # raw_writer      → raw frames
    # annotated_writer → YOLO-annotated frames
    # ------------------------------------------------------------
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    raw_writer = cv2.VideoWriter(raw_path, fourcc, fps, (frame_width, frame_height))
    annotated_writer = cv2.VideoWriter(annotated_path, fourcc, fps, (frame_width, frame_height))

    print(f"Monitoring started")
    print(f"Raw recording:       {raw_path}")
    print(f"Annotated recording: {annotated_path}")

    last_capture_time = 0
    last_annotated_frame = None

    try:
        while within_recording_time():

            ret, frame = cap.read()
            if not ret:
                print("Frame grab failed")
                continue

            current_time = time.time()

            # Always write raw frame (privacy issue)
            raw_writer.write(frame)

            # Run YOLO every CAPTURE_INTERVAL seconds
            if current_time - last_capture_time >= CAPTURE_INTERVAL:

                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                # ================= PEOPLE DETECTION =================
                results = model(frame, conf=0.4, imgsz=1280, verbose=False)

                count = 0
                boxes_to_write = []

                # Loop through YOLO detections
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

                # Write bounding boxes
                if boxes_to_write:
                    with open(BOXES_CSV,"a",newline="") as f:
                        writer = csv.writer(f)
                        writer.writerows(boxes_to_write)

                # Write total people count
                with open(TOTAL_CSV,"a",newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([timestamp, count])

                # ================= CURTAIN DETECTION =================
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                states = []

                for (x, y, w, h) in rois:
                    roi_current = gray[y:y+h, x:x+w]
                    roi_ref = ref_gray[y:y+h, x:x+w]

                    diff = cv2.absdiff(roi_current, roi_ref)
                    score = np.mean(diff)

                    state = 0 if score > DIFF_THRESHOLD else 1
                    states.append(state)

                # Write curtain states
                with open(CURTAIN_CSV,"a",newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([timestamp] + states)

                print(f"[{timestamp}] People: {count} | Curtains: {states}")

                # Save YOLO-annotated frame
                last_annotated_frame = results[0].plot()

                last_capture_time = time.time()

            # Write annotated frame (use last YOLO result)
            if last_annotated_frame is not None:
                annotated_writer.write(last_annotated_frame)
            else:
                annotated_writer.write(frame)

            cv2.waitKey(1)

    except KeyboardInterrupt:
        print("Stopped manually")

    finally:
        cap.release()
        raw_writer.release()
        annotated_writer.release()
        cv2.destroyAllWindows()
        print(f"Recordings saved to {OUTPUT_FOLDER}")

# ================= MAIN LOOP =================

def main():
    """
    Continuously checks whether it is recording time.
    If yes → run monitoring loop.
    If no → sleep and check again.
    """

    print("Script running. Waiting for recording time.")

    while True:
        if within_recording_time():
            run_monitoring()
        else:
            print("Outside recording hours")
            time.sleep(60)

if __name__ == "__main__":
    main()
