import cv2
import time
import csv
from datetime import datetime
from ultralytics import YOLO

# Configuration
MODEL_PATH = "yolov8s.pt"
CAPTURE_INTERVAL = 1  # seconds between captures
BOXES_CSV = "people_boxes1.csv"
TOTAL_CSV = "people_total1.csv"

model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture("https://streamingwebcams.mtu.edu:1935/rtplive/camera004.stream/playlist.m3u8")

# Creating CSV files with headers if they don't exist
with open(BOXES_CSV, "a", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["timestamp", "x1", "y1", "x2", "y2"])

with open(TOTAL_CSV, "a", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["timestamp", "total_people"])

print("Started monitoring... Press Ctrl+C to stop.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        results = model(frame, conf=0.4, imgsz=1280, verbose=False)

        annotated = results[0].plot()
        cv2.imshow("Debug", annotated)
        cv2.waitKey(1)

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

        print(f"Detected people: {count}")

        time.sleep(CAPTURE_INTERVAL)

except KeyboardInterrupt:
    print("Stopping...")

cap.release()
cv2.destroyAllWindows()