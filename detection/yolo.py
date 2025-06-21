from ultralytics import YOLO

from detection import BaseDetector
from typing import List
from utils.types import Detection


class YOLODetector(BaseDetector):
    def __init__(self, model="yolov8n.pt") -> None:
        self.model = YOLO(model)

    def detect(self, batch):
        results = self.model.predict(source=batch, classes=[2], save=False)

        all_detections: List[Detection] = []

        for result in results:
            boxes = result.boxes
            detections: List[Detection] = []

            for box, conf in zip(boxes.xyxy, boxes.conf):
                confidence = float(conf.item())
                class_name = "car"
                detection = Detection(
                    box=box, confidence=confidence, class_name=class_name
                )
                detections.append(detection)

            all_detections.append(detections)

        return all_detections
    
    def preprocess(self, data):
        return
