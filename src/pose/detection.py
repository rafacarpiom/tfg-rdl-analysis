
from __future__ import annotations

from ultralytics import YOLO


class YoloPersonDetector:

    def __init__(
        self,
        weights_path: str,
        device: str = "cpu",
        conf_threshold: float = 0.25,
    ) -> None:
        self.model = YOLO(weights_path)
        self.model.to(device)
        self.device = device
        self.conf_threshold = float(conf_threshold)

    def detect(self, frame) -> list[list[float]]:
        results = self.model.predict(frame, device=self.device, verbose=False)
        detections = results[0]

        person_boxes: list[list[float]] = []
        for box in detections.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            if cls == 0 and conf >= self.conf_threshold:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                person_boxes.append([float(x1), float(y1), float(x2), float(y2)])
        return person_boxes
