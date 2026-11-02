import unittest
from unittest.mock import Mock, patch

import numpy as np

from src.detector import CarPartDetector
from src.pipeline import CarDamagePipeline


class CarDamagePipelineTests(unittest.TestCase):
    def test_classifies_each_detected_part_crop(self):
        image = np.zeros((10, 12, 3), dtype=np.uint8)
        detection = {
            "part": "door",
            "confidence": 0.9,
            "bbox": [-2, 1, 8, 7],
            "mask": None,
        }
        detector = Mock()
        detector.detect.return_value = [detection]
        classifier = Mock()
        classifier.predict.return_value = {
            "damage": "Dent",
            "damage_confidence": 0.8,
        }

        result = CarDamagePipeline(detector, classifier).analyze(image)

        self.assertEqual(result[0]["damage"], "Dent")
        classified_image = classifier.predict.call_args.args[0]
        self.assertEqual(classified_image.size, (8, 6))
        self.assertIn("mask", result[0])

    def test_rejects_invalid_detection_coordinates(self):
        detector = Mock()
        detector.detect.return_value = [
            {"part": "door", "bbox": [5, 1, 4, 7]}
        ]
        classifier = Mock()
        image = np.zeros((10, 12, 3), dtype=np.uint8)

        with self.assertRaisesRegex(ValueError, "coordenadas inválidas"):
            CarDamagePipeline(detector, classifier).analyze(image)

        classifier.predict.assert_not_called()


class CarPartDetectorTests(unittest.TestCase):
    def test_returns_segmentation_mask_with_each_detection(self):
        class MaskData:
            def detach(self):
                return self

            def cpu(self):
                return self

            def numpy(self):
                return np.array([[0.0, 1.0], [1.0, 1.0]], dtype=np.float32)

        box = Mock()
        box.cls = [0]
        box.conf = [0.95]
        coordinates = Mock()
        coordinates.tolist.return_value = [0.0, 0.0, 3.0, 3.0]
        box.xyxy = [coordinates]
        result = Mock()
        result.boxes = [box]
        result.masks.data = [MaskData()]
        result.names = {0: "door"}
        model = Mock(return_value=[result])

        with patch("src.detector.YOLO", return_value=model):
            detector = CarPartDetector("model.pt")
            detections = detector.detect(
                np.zeros((4, 4, 3), dtype=np.uint8)
            )

        self.assertEqual(detections[0]["part"], "door")
        self.assertEqual(detections[0]["mask"].shape, (4, 4))
        self.assertTrue(detections[0]["mask"].dtype == np.bool_)


if __name__ == "__main__":
    unittest.main()
