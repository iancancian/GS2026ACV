"""Executa inferencia YOLO e exporta observacao e triagem em JSON."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from boca_de_lobo.decision import assess_inspection
from boca_de_lobo.paths import PROJECT_ROOT

LOGGER = logging.getLogger("boca_de_lobo.predict")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspeciona uma foto com pesos YOLO treinados.")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--asset-id", required=True, help="Identificador interno do bueiro, sem endereco.")
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--strong-rain", action="store_true")
    parser.add_argument("--risk-area", action="store_true")
    parser.add_argument("--report", type=Path, help="Salva relatorio legivel em texto UTF-8.")
    parser.add_argument("--annotated-image", type=Path, help="Salva uma copia da foto com caixas e confiancas.")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "runs" / "predictions" / "inspection.json",
    )
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _parse_args()
    destinations = [path for path in (args.output, args.report, args.annotated_image) if path is not None]
    if len({path.resolve() for path in destinations}) != len(destinations):
        LOGGER.error("Os arquivos de saida devem ter caminhos diferentes.")
        return 2
    if {args.image.resolve(), args.weights.resolve()} & {path.resolve() for path in destinations}:
        LOGGER.error("Um arquivo de saida nao pode substituir a imagem de entrada ou os pesos.")
        return 2
    if not 0 < args.conf <= 1:
        LOGGER.error("--conf deve estar no intervalo (0, 1].")
        return 2
    if not args.asset_id.strip():
        LOGGER.error("--asset-id nao pode ser vazio.")
        return 2
    if not args.image.is_file():
        LOGGER.error("Imagem nao encontrada: %s", args.image)
        return 2
    if not args.weights.is_file():
        LOGGER.error("Pesos nao encontrados: %s", args.weights)
        return 2

    try:
        from boca_de_lobo.detector import load_detector

        detections = load_detector(args.weights, args.conf)(args.image)
    except ImportError:
        LOGGER.error("Dependencia ausente. Instale com: pip install -r requirements.txt")
        return 2

    assessment = assess_inspection(
        detections=detections,
        strong_rain=args.strong_rain,
        risk_area=args.risk_area,
    )
    observed_at = datetime.now(timezone.utc)
    payload = {
        "schema_version": "1.0",
        "inspection_id": str(uuid4()),
        "asset_id": args.asset_id,
        "observed_at": observed_at.isoformat(),
        "image_name": args.image.name,
        "model": {
            "weights": args.weights.name,
            "confidence_threshold": args.conf,
        },
        "detections": detections,
        "assessment": assessment,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    LOGGER.info("Inspecao exportada para %s (%d deteccao(oes)).", args.output, len(detections))
    if args.report or args.annotated_image:
        from boca_de_lobo.inspection_report import format_report, save_annotated_image

        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(format_report(payload), encoding="utf-8")
            LOGGER.info("Relatorio salvo em %s.", args.report)
        if args.annotated_image:
            save_annotated_image(args.image, args.annotated_image, detections)
            LOGGER.info("Imagem anotada salva em %s.", args.annotated_image)
    return 0


if __name__ == "__main__":
    sys.exit(main())
