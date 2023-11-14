# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 13:21:10 2023

@author: sethuri
"""


import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from datetime import datetime

# Initialize YOLO model
model = YOLO("yolov8n.pt")

area1_pts = np.array([[50, 50], [250, 50], [250, 350], [50, 350]], np.int32)
area2_pts = np.array([[250, 50], [450, 50], [450, 350], [250, 350]], np.int32)
area3_pts = np.array([[450, 50], [650, 50], [650, 350], [450, 350]], np.int32)
area4_pts = np.array([[650, 50], [850, 50], [850, 350], [650, 350]], np.int32)

area1_pts = area1_pts.reshape((-1, 1, 2))
area2_pts = area2_pts.reshape((-1, 1, 2))
area3_pts = area3_pts.reshape((-1, 1, 2))
area4_pts = area4_pts.reshape((-1, 1, 2))


# Create a DeepSort tracker
tracker = DeepSort(max_age=10)
count = 0
cap = cv2.VideoCapture(1)

my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")

# Create dictionaries to track entry and exit times for each person in the two areas
area1_tracker = {}
area2_tracker = {}
area3_tracker = {}
area4_tracker = {}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    count += 1
    if count % 3 != 0:
        continue

    frame = cv2.resize(frame, (1020, 500))

    # Draw polygons for both areas on the frame
    cv2.polylines(frame, [area1_pts], isClosed=True, color=(0, 0, 255), thickness=2)
    cv2.polylines(frame, [area2_pts], isClosed=True, color=(0, 0, 255), thickness=2)
    cv2.polylines(frame, [area3_pts], isClosed=True, color=(0, 0, 255), thickness=2)
    cv2.polylines(frame, [area4_pts], isClosed=True, color=(0, 0, 255), thickness=2)

    results = model.predict(frame)
    a = results[0].boxes.data
    px = pd.DataFrame(a.cpu()).astype("float")
    detections = []

    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])

        c = class_list[d]
        if 'person' in c:
            # Append detections in the format expected by DeepSort
            detections.append(([x1, y1, x2 - x1, y2 - y1], row[4], c))

    # Update tracks using DeepSort
    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        bbox = track.to_tlbr()
        cx = int((bbox[0] + bbox[2]) / 2)
        cy = int((bbox[1] + bbox[3]) / 2)
        color = (0, 255, 0)

        # Check if a person is in area 1
        if cv2.pointPolygonTest(area1_pts, (cx, cy), False) >= 0:
            if track_id not in area1_tracker:
                area1_tracker[track_id] = {"entry_time": datetime.now(), "exit_time": None}
        elif cv2.pointPolygonTest(area2_pts, (cx, cy), False) >= 0:
            if track_id not in area2_tracker:
                area2_tracker[track_id] = {"entry_time": datetime.now(), "exit_time": None}
                
        elif cv2.pointPolygonTest(area3_pts, (cx, cy), False) >= 0:
            if track_id not in area3_tracker:
                area3_tracker[track_id] = {"entry_time": datetime.now(), "exit_time": None}

        elif cv2.pointPolygonTest(area4_pts, (cx, cy), False) >= 0:
            if track_id not in area4_tracker:
                area4_tracker[track_id] = {"entry_time": datetime.now(), "exit_time": None}

    # Check if people have exited the areas
    for track_id in list(area1_tracker.keys()):
        if track_id not in [track.track_id for track in tracks]:
            if area1_tracker[track_id]["exit_time"] is None:
                area1_tracker[track_id]["exit_time"] = datetime.now()

    for track_id in list(area2_tracker.keys()):
        if track_id not in [track.track_id for track in tracks]:
            if area2_tracker[track_id]["exit_time"] is None:
                area2_tracker[track_id]["exit_time"] = datetime.now()
                
    for track_id in list(area3_tracker.keys()):
        if track_id not in [track.track_id for track in tracks]:
            if area3_tracker[track_id]["exit_time"] is None:
                area3_tracker[track_id]["exit_time"] = datetime.now()
                
    for track_id in list(area4_tracker.keys()):
        if track_id not in [track.track_id for track in tracks]:
            if area4_tracker[track_id]["exit_time"] is None:
                area4_tracker[track_id]["exit_time"] = datetime.now()

    # Display the person counts for both areas on the frame
    area1_count = len([track_id for track_id, times in area1_tracker.items() if times["exit_time"] is None])
    area2_count = len([track_id for track_id, times in area2_tracker.items() if times["exit_time"] is None])
    area3_count = len([track_id for track_id, times in area3_tracker.items() if times["exit_time"] is None])
    area4_count = len([track_id for track_id, times in area4_tracker.items() if times["exit_time"] is None])
    
    cv2.putText(frame, f'Area 1 Count: {area1_count}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, f'Area 2 Count: {area2_count}', (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, f'Area 3 Count: {area3_count}', (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, f'Area 4 Count: {area4_count}', (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Export results to a text file in the desired format
with open("results.txt", "w") as text_file:
    text_file.write("personid\tentry time\texit time\tarea#\n")
    
    for track_id, times in area1_tracker.items():
        entry_time = times["entry_time"]
        exit_time = times["exit_time"]
        area = "Area 1"
        text_file.write(f"{track_id}\t{entry_time}\t{exit_time}\t{area}\n")

    for track_id, times in area2_tracker.items():
        entry_time = times["entry_time"]
        exit_time = times["exit_time"]
        area = "Area 2"
        text_file.write(f"{track_id}\t{entry_time}\t{exit_time}\t{area}\n")

    for track_id, times in area3_tracker.items():
        entry_time = times["entry_time"]
        exit_time = times["exit_time"]
        area = "Area 3"
        text_file.write(f"{track_id}\t{entry_time}\t{exit_time}\t{area}\n")
        
    for track_id, times in area4_tracker.items():
        entry_time = times["entry_time"]
        exit_time = times["exit_time"]
        area = "Area 4"
        text_file.write(f"{track_id}\t{entry_time}\t{exit_time}\t{area}\n")