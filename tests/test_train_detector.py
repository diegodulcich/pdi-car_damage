import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import train_detector


class TrainDetectorTests(unittest.TestCase):
    def test_resume_loads_last_checkpoint_and_copies_best_weights(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            checkpoint = root / "runs" / "weights" / "last.pt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"checkpoint")
            best_weights = root / "runs" / "weights" / "best.pt"
            best_weights.write_bytes(b"best weights")
            destination = root / "models" / "carparts-seg.pt"

            model = Mock()
            model.train.return_value = SimpleNamespace(
                save_dir=str(root / "runs")
            )
            with (
                patch("train_detector.YOLO", return_value=model) as yolo,
                patch.object(train_detector, "LAST_CHECKPOINT", checkpoint),
                patch.object(train_detector, "PART_MODEL_PATH", destination),
            ):
                train_detector.main(["--resume"])

            yolo.assert_called_once_with(checkpoint)
            model.train.assert_called_once_with(resume=True)
            self.assertEqual(destination.read_bytes(), b"best weights")

    def test_resume_without_checkpoint_reports_error(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint = Path(temporary_directory) / "last.pt"
            with (
                patch.object(train_detector, "LAST_CHECKPOINT", checkpoint),
                patch("train_detector.YOLO") as yolo,
                self.assertRaisesRegex(FileNotFoundError, "checkpoint"),
            ):
                train_detector.main(["--resume"])

            yolo.assert_not_called()


if __name__ == "__main__":
    unittest.main()
