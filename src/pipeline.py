"""Orquestación de la segmentación de piezas y clasificación de daños."""

from typing import Any

import cv2
import numpy as np
from PIL import Image

from src.classifier import DamageClassifier
from src.detector import CarPartDetector


class CarDamagePipeline:
    def __init__(
        self,
        detector: CarPartDetector,
        classifier: DamageClassifier,
    ):
        self.detector = detector
        self.classifier = classifier

    def analyze(self, image: np.ndarray) -> list[dict[str, Any]]:
        detections = self.detector.detect(image)
        height, width = image.shape[:2]
        analyzed: list[dict[str, Any]] = []

        for detection in detections:
            x1, y1, x2, y2 = detection["bbox"]
            x1 = max(0, min(x1, width))
            x2 = max(0, min(x2, width))
            y1 = max(0, min(y1, height))
            y2 = max(0, min(y2, height))

            if x2 <= x1 or y2 <= y1:
                raise ValueError(
                    f"La detección de '{detection['part']}' tiene "
                    "coordenadas inválidas."
                )

            crop = cv2.cvtColor(
                image[y1:y2, x1:x2],
                cv2.COLOR_BGR2RGB,
            )
            damage = self.classifier.predict(Image.fromarray(crop))
            analyzed.append({**detection, **damage})

        return analyzed
