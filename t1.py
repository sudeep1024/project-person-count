!pip install ultralytics
!pip install deep-sort-realtime opencv-python-headless numpy

import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from google.colab.patches import cv2_imshow

VIDEO_SOURCE = "/content/testing.mp4"
MODEL_PATH = "yolov8n-pose.pt"
CONFIDENCE_THRESHOLD = 0.5
PERSON_CLASS_ID = 0
RUNNING_SPEED_THRESHOLD = 10  # Start with 10 and adjust as needed

track_histories = {}
unique_person_ids = set()

cap = cv2.VideoCapture(VIDEO_SOURCE)
if not cap.isOpened():
    print(f"Error: Could not open video source: {VIDEO_SOURCE}")
    exit()

model = YOLO(MODEL_PATH)
tracker = DeepSort(embedder="mobilenet", half=True, bgr=True, embedder_gpu=True)

frame_num = 0

while True:
    ret, frame = cap.read()
    if not ret or frame_num > 100000000:  # Only process up to a very high frame number
        print("End of video stream or frame limit reached.")
        break

    results = model(frame, stream=True, verbose=False)
    detections_for_tracker = []
    keypoints_per_box = {}

    for result in results:
        boxes = result.boxes
        keypoints = result.keypoints
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box.xyxy[0]
            confidence = box.conf[0]
            class_id = int(box.cls[0])
            if class_id == PERSON_CLASS_ID and confidence > CONFIDENCE_THRESHOLD:
                w, h = x2 - x1, y2 - y1
                bbox = [int(x1), int(y1), int(w), int(h)]
                detections_for_tracker.append((bbox, confidence, 'person'))
                keypoints_per_box[tuple(bbox)] = keypoints[i].xy[0]

    tracks = tracker.update_tracks(detections_for_tracker, frame=frame)
    person_count = 0
    active_track_ids = []

    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        unique_person_ids.add(track_id)  # Add track_id for unique counting
        active_track_ids.append(track_id)
        ltrb = track.to_ltrb()
        bbox = (int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3]))

        person_count += 1
        is_running = False

        original_w, original_h = ltrb[2] - ltrb[0], ltrb[3] - ltrb[1]
        original_bbox_for_key = [int(ltrb[0]), int(ltrb[1]), int(original_w), int(original_h)]
        kps = keypoints_per_box.get(tuple(original_bbox_for_key))

        speed = 0
        if kps is not None and len(kps) > 12:
            left_hip = kps[11]
            right_hip = kps[12]
            torso_center_x = int((left_hip[0] + right_hip[0]) / 2)
            torso_center_y = int((left_hip[1] + right_hip[1]) / 2)
            current_pos = (torso_center_x, torso_center_y)

            if track_id in track_histories:
                last_pos = track_histories[track_id]
                speed = np.sqrt((current_pos[0] - last_pos[0])**2 + (current_pos[1] - last_pos[1])**2)
                if speed > RUNNING_SPEED_THRESHOLD:
                    is_running = True

            track_histories[track_id] = current_pos

        box_color = (255, 0, 0) if is_running else (0, 255, 0)
        label = f"ID: {track_id}"

        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), box_color, 2)
        if is_running:
            alert_text = "ALERT: RUNNING"
            text_size, _ = cv2.getTextSize(alert_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame, (bbox[0], bbox[1] - 35 - text_size[1]), (bbox[0] + text_size[0], bbox[1] - 35), box_color, -1)
            cv2.putText(frame, alert_text, (bbox[0], bbox[1] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.putText(frame, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
        print(f"Frame {frame_num}, Track {track_id}: Speed={speed:.2f}, Running={is_running}")

    stale_tracks = set(track_histories.keys()) - set(active_track_ids)
    for track_id in stale_tracks:
        del track_histories[track_id]

    cv2.putText(frame, f"Person Count: {person_count}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 6)
    cv2.putText(frame, f"Total Unique: {len(unique_person_ids)}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 6)
    cv2_imshow(frame)

    frame_num += 1

cap.release()

print("Total unique person count in the video:", len(unique_person_ids))