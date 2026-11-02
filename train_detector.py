import argparse
from pathlib import Path
import shutil
from typing import Sequence

from ultralytics import YOLO

from src.config import (
    PART_MODEL_PATH,
    PROJECT_ROOT,
    SEGMENTATION_BASE_MODEL,
    SEGMENTATION_DATASET,
)


RUN_PROJECT = PROJECT_ROOT / "runs" / "segment"
RUN_NAME = "carparts-seg"
LAST_CHECKPOINT = RUN_PROJECT / RUN_NAME / "weights" / "last.pt"


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Entrena o reanuda el detector de piezas Carparts-Seg."
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="reanuda el entrenamiento desde el último checkpoint guardado",
    )
    args = parser.parse_args(argv)

    if args.resume:
        if not LAST_CHECKPOINT.is_file():
            raise FileNotFoundError(
                f"No se encontró el checkpoint para reanudar: "
                f"{LAST_CHECKPOINT}. Iniciá un entrenamiento nuevo sin "
                "--resume."
            )
        model = YOLO(LAST_CHECKPOINT)
        results = model.train(resume=True)
    else:
        model = YOLO(SEGMENTATION_BASE_MODEL)
        results = model.train(
            data=SEGMENTATION_DATASET,
            task="segment",
            epochs=25,
            imgsz=640,
            project=RUN_PROJECT,
            name=RUN_NAME,
            exist_ok=True,
        )

    best_weights = Path(results.save_dir) / "weights" / "best.pt"
    if not best_weights.is_file():
        raise FileNotFoundError(
            f"Ultralytics no generó el archivo esperado: {best_weights}"
        )

    PART_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_weights, PART_MODEL_PATH)
    print(f"Modelo de segmentación guardado en: {PART_MODEL_PATH}")


if __name__ == "__main__":
    main()
