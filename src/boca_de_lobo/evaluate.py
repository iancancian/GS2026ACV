"""Avalia pesos treinados no conjunto test, sem ajusta-los nesse conjunto."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Sequence

from boca_de_lobo.dataset import validate_dataset
from boca_de_lobo.paths import PROJECT_ROOT

LOGGER = logging.getLogger("boca_de_lobo.evaluate")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Avalia o modelo; por padrao no split test isolado.")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=None, help="Ex.: cpu, 0; omitido para selecao do Ultralytics.")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "runs" / "boca_de_lobo" / "evaluation" / "test_metrics.json",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if min(args.imgsz, args.batch) <= 0:
        parser.error("--imgsz e --batch devem ser positivos.")
    if not args.weights.is_file():
        LOGGER.error("Pesos nao encontrados: %s", args.weights)
        return 2

    dataset_root = PROJECT_ROOT / "dataset"
    errors = validate_dataset(dataset_root)
    if errors:
        LOGGER.error("Dataset invalido: %d problema(s).", len(errors))
        for error in errors:
            LOGGER.error("%s", error)
        return 1

    try:
        from ultralytics import YOLO
    except ImportError:
        LOGGER.error("Dependencia ausente. Instale com: pip install -r requirements.txt")
        return 2

    model = YOLO(str(args.weights))
    evaluation_options: dict[str, object] = {
        "data": str(dataset_root / "data.yaml"),
        "split": args.split,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "plots": True,
    }
    if args.device is not None:
        evaluation_options["device"] = args.device

    results = model.val(**evaluation_options)
    per_class = {}
    for position, class_index in enumerate(results.box.ap_class_index):
        precision, recall, map50, map50_95 = results.box.class_result(position)
        per_class[str(results.names[int(class_index)])] = {
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "map50": round(float(map50), 4),
            "map50_95": round(float(map50_95), 4),
        }
    report = {
        "weights": args.weights.name,
        "dataset_split": args.split,
        "imgsz": args.imgsz,
        "metrics": {name: round(float(value), 4) for name, value in results.results_dict.items()},
        "per_class": per_class,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    LOGGER.info("Metricas (%s) salvas em %s.", args.split, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
