from pathlib import Path
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO


class CarPartDetector:

    def __init__(self, model_path: str | Path):
        self.model = YOLO(model_path)

    def detect(self, image: np.ndarray) -> list[dict[str, Any]]:
        if image.ndim != 3:
            raise ValueError("La imagen debe tener tres canales de color.")

        results = self.model(image)

        detections: list[dict[str, Any]] = []
        height, width = image.shape[:2]

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            masks = result.masks
            for index, box in enumerate(boxes):
                cls_id = int(box.cls[0])
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                mask = None
                if masks is not None and index < len(masks.data):
                    mask_data = masks.data[index].detach().cpu().numpy()
                    mask = cv2.resize(
                        mask_data,
                        (width, height),
                        interpolation=cv2.INTER_NEAREST,
                    ) > 0.5

                detections.append(
                    {
                        "part": result.names[cls_id],
                        "confidence": confidence,
                        "bbox": [
                            int(x1),
                            int(y1),
                            int(x2),
                            int(y2),
                        ],
                        "mask": mask,
                    }
                )

        return detections
