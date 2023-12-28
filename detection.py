from ultralytics import YOLO
# from ultralytics.yolo.v8.detect.predict import DetectionPrector

import cv2

previous_result = None
model = YOLO("best (1).pt")
results = model.predict(source=0, show=True)

for r in results:
    for c in r.boxes.cls:
        if model.names[int(c)] != previous_result:
            print(model.names[int(c)])
            previous_result = model.names[int(c)]