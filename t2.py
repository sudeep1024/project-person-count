import time
import cv2
import numpy as np
from picamera2 import Picamera2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

MODEL_PATH = "yolov8n-pose.pt"
CONFIDENCE_THRESHOLD = 0.5
PERSON_CLASS_ID = 0
RUNNING_SPEED_THRESHOLD = 10

track_histories = {}
unique_person_ids = set()

# Initialize Picamera2
picam2 = Picamera2()
camera_config = picam2.create_preview_configuration(main={"size": (640, 480), "format": "RGB888"})
picam2.configure(camera_config)
picam2.start()
time.sleep(2)  # allow camera to warm up

model = YOLO(MODEL_PATH)
tracker = DeepSort(embedder="mobilenet", half=True, bgr=True, embedder_gpu=True)

frame_num = 0
max_frames = 10000  # adjust as needed or remove for infinite

while frame_num < max_frames:
    # Capture a frame from Pi Camera
    frame = picam2.capture_array()

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
        unique_person_ids.add(track_id)
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

    # Display (will show window on Pi desktop, use with HDMI display)
    cv2.imshow("Pi Camera Person Counter", frame)
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        print("Exit requested.")
        break

    frame_num += 1

cap.release()
cv2.destroyAllWindows()
print("Total unique person count in the session:", len(unique_person_ids))
