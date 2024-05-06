import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
from collections import defaultdict

track_history = defaultdict(lambda: [])
model = YOLO("yolov8n-seg.pt")  # segmentation model
cap = cv2.VideoCapture(0)

max_det = 50
conf = 0.2

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Inference
    results = model(frame_rgb, max_det=max_det, conf=conf)
    annotator = Annotator(frame, line_width=2)
    results = model.track(frame, persist=True)

    if results[0].boxes.id is not None and results[0].masks is not None:
        masks = results[0].masks.xy
        track_ids = results[0].boxes.id.int().cpu().tolist()
        classes = results[0].boxes.cls.int().cpu().tolist()

        for mask, track_id, cls in zip(masks, track_ids, classes):
            annotator.seg_bbox(mask=mask, mask_color=colors(track_id, True), track_label=f"{model.names[cls]} {track_id}")

    annotated_frame = annotator.result()
    cv2.imshow("Instance Segmentation with Object Tracking", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

