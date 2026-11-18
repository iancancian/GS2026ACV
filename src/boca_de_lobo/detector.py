"""Deteccao com pesos YOLO, isolada para que testes usem um detector falso."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

Detection = dict[str, Any]
Detector = Callable[[Path], list[Detection]]


def load_detector(weights: Path, confidence: float) -> Detector:
    """Carrega os pesos uma vez e devolve uma funcao imagem -> deteccoes."""
    from ultralytics import YOLO  # importacao lenta; so quando ha modelo real

    model = YOLO(str(weights))

    def detect(image: Path) -> list[Detection]:
        result = model.predict(source=str(image), conf=confidence, verbose=False)[0]
        return [
            {
                "class_id": int(box.cls.item()),
                "class_name": str(result.names[int(box.cls.item())]),
                "confidence": round(float(box.conf.item()), 6),
                "bbox_xyxy_pixels": [round(float(value), 2) for value in box.xyxy[0].tolist()],
            }
            for box in result.boxes
        ]

    return detect
