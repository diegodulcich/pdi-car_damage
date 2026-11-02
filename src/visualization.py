from typing import Any

import cv2
import numpy as np


BOX_COLOR = (0, 255, 0)
TEXT_COLOR = (255, 255, 255)


def draw_detections(
    image: np.ndarray,
    detections: list[dict[str, Any]],
) -> np.ndarray:

    output = image.copy()

    for detection in detections:
        mask = detection.get("mask")
        if mask is not None:
            overlay = np.zeros_like(output)
            overlay[mask] = BOX_COLOR
            output = cv2.addWeighted(output, 1.0, overlay, 0.3, 0)

        x1, y1, x2, y2 = detection["bbox"]

        label = (
            f"{detection['part']} "
            f"{detection['confidence']:.2f}"
        )
        if "damage" in detection:
            label += (
                f" | {detection['damage']} "
                f"{detection['damage_confidence']:.2f}"
            )

        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            BOX_COLOR,
            2,
        )

        cv2.putText(
            output,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            TEXT_COLOR,
            2,
        )

    return output
