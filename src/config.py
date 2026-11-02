"""Configuración compartida de la aplicación."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PART_MODEL_PATH = PROJECT_ROOT / "models" / "yolo" / "carparts-seg.pt"
DAMAGE_MODEL_ID = "beingamit99/car_damage_detection"

SEGMENTATION_DATASET = "carparts-seg.yaml"
SEGMENTATION_BASE_MODEL = "yolov8n-seg.pt"
