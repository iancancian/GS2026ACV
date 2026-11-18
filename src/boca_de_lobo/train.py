"""Treinamento YOLO com verificacao previa e parametros explicitos."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Sequence

from boca_de_lobo.dataset import validate_dataset
from boca_de_lobo.paths import PROJECT_ROOT

LOGGER = logging.getLogger("boca_de_lobo.train")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Treina o detector Boca de Lobo.")
    parser.add_argument("--model", default="yolo11n.pt", help="Pesos iniciais ou caminho local.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=None, help="Ex.: cpu, 0; omitido para selecao do Ultralytics.")
    parser.add_argument("--name", default="train")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if min(args.epochs, args.imgsz, args.batch) <= 0:
        parser.error("--epochs, --imgsz e --batch devem ser positivos.")

    dataset_root = PROJECT_ROOT / "dataset"
    errors = validate_dataset(dataset_root)
    if errors:
        LOGGER.error("Dataset invalido: %d problema(s).", len(errors))
        for error in errors:
            LOGGER.error("%s", error)
        return 1

    image_count = sum(
        1
        for split in ("train", "val", "test")
        for image in (dataset_root / "images" / split).iterdir()
        if image.is_file() and image.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if image_count == 0:
        LOGGER.error("Dataset vazio. Colete, anote e valide imagens antes do treino.")
        return 1

    try:
        from ultralytics import YOLO
    except ImportError:
        LOGGER.error("Dependencia ausente. Instale com: pip install -r requirements.txt")
        return 2

    model = YOLO(args.model)
    train_options: dict[str, object] = {
        "data": str(dataset_root / "data.yaml"),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "project": str(PROJECT_ROOT / "runs" / "boca_de_lobo"),
        "name": args.name,
        "patience": 20,
        "seed": 42,
        "deterministic": True,
        "workers": 4,
        "val": True,
    }
    if args.device is not None:
        train_options["device"] = args.device

    model.train(**train_options)
    LOGGER.info("Treinamento concluido. Avalie o conjunto test antes de qualquer uso operacional.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
