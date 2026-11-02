"""Clasificación de daños en imágenes de piezas detectadas."""

from typing import Any

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification


class DamageClassifier:
    def __init__(self, model_id: str):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModelForImageClassification.from_pretrained(model_id)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, image: Image.Image) -> dict[str, Any]:
        inputs = self.processor(images=image.convert("RGB"), return_tensors="pt")
        inputs = {
            name: tensor.to(self.device)
            for name, tensor in inputs.items()
        }

        with torch.inference_mode():
            logits = self.model(**inputs).logits
            probabilities = torch.softmax(logits, dim=-1)[0]
            confidence, class_id = torch.max(probabilities, dim=0)

        predicted_id = int(class_id.item())
        return {
            "damage": self.model.config.id2label[predicted_id],
            "damage_confidence": float(confidence.item()),
        }
