"""Gera demonstracao silenciosa a partir de inferencias reais em duas fotos revisadas."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = (
    "test_0033_IMG_20250627_182743_jpg.rf.4b00e5cc87833788d3f95ea1a24a1ecf.jpg",
    "test_0027_IMG_20250627_181511_jpg.rf.ff8b598c4c3d014c27cebc825a7f19f6.jpg",
)
SIZE = (1280, 720)
FPS = 10


def frame(title: str, lines: list[str], photo: Path | None = None) -> np.ndarray:
    canvas = np.full((SIZE[1], SIZE[0], 3), (30, 25, 23), dtype=np.uint8)
    cv2.putText(canvas, title, (40, 65), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    y = 110
    for line in lines:
        cv2.putText(canvas, line, (40, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1, cv2.LINE_AA)
        y += 30
    if photo is not None:
        image = cv2.imread(str(photo))
        if image is None:
            raise ValueError(f"Nao foi possivel abrir a imagem: {photo}")
        scale = min(1100 / image.shape[1], 470 / image.shape[0])
        image = cv2.resize(image, None, fx=scale, fy=scale)
        top = max(y + 5, 720 - image.shape[0] - 25)
        left = (1280 - image.shape[1]) // 2
        canvas[top:top + image.shape[0], left:left + image.shape[1]] = image
    return canvas


def main() -> None:
    output = ROOT / "output" / "demonstracao_visao"
    output.mkdir(parents=True, exist_ok=True)
    stages = [
        (frame("Boca de Lobo - Applied Computer Vision", [
            "Triagem preventiva de imagens de drenagem urbana.",
            "YOLO11n treinado para bueiro, lixo, folha, pedra e sedimento.",
            "Demonstracao automatica sem audio. Resultados reais; revisao humana obrigatoria.",
        ]), 5)
    ]
    for index, name in enumerate(SAMPLES, 1):
        image = ROOT / "dataset" / "images" / "test" / name
        json_file = output / f"inspecao_{index}.json"
        report_file = output / f"inspecao_{index}.txt"
        annotated = output / f"inspecao_{index}.jpg"
        command = [
            sys.executable, "-m", "boca_de_lobo.predict",
            "--image", str(image), "--weights", str(ROOT / "models" / "best.pt"),
            "--asset-id", f"DEMO-{index:03d}", "--output", str(json_file),
            "--report", str(report_file), "--annotated-image", str(annotated),
        ]
        subprocess.run(command, cwd=ROOT, check=True)
        result = json.loads(json_file.read_text(encoding="utf-8"))
        inspection = result["assessment"]
        details = [
            f"Deteccoes: {len(result['detections'])} | Bueiros: {inspection['drain_count']}",
        ]
        for drain in inspection["drains"][:3]:
            details.append(
                f"Bueiro {drain['drain_index'] + 1}: {drain['state']} | "
                f"cobertura estimada {drain['obstruction_coverage']:.1%}"
            )
        if not inspection["drains"]:
            details.append("Nenhum bueiro detectado: revisao manual.")
        details.append(f"Prioridade sugerida: {inspection['maintenance_priority']}")
        stages.extend([
            (frame(f"Inspecao {index}: imagem de entrada", [
                "Comando: python -m boca_de_lobo.predict --image FOTO --weights models/best.pt",
                "A inferencia usa o modelo treinado; a foto abaixo e do conjunto de teste.",
            ], image), 5),
            (frame(f"Inspecao {index}: resultado real da inferencia", details, annotated), 8),
        ])
    stages.append((frame("Interpretacao e limites", [
        "Abaixo de 10%: sem obstrucao visivel; 10% a menos de 80%: parcial.",
        "A partir de 80%: entupido provavel. Limiares experimentais nao calibrados.",
        "Caixas nao medem volume, area real de residuos ou obstrucao interna.",
        "A ausencia de deteccao nao garante bueiro livre. Confirmar em campo.",
        "Teste: mAP50 geral 0,393; folha e sedimento apresentam recall baixo.",
        "Codigo, pesos e metodologia: README e relatorio de Applied Computer Vision.",
    ]), 9))
    video_path = ROOT / "docs" / "Demonstracao_Applied_Computer_Vision.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, SIZE)
    if not writer.isOpened():
        raise RuntimeError(f"Nao foi possivel criar video: {video_path}")
    try:
        for image, seconds in stages:
            for _ in range(seconds * FPS):
                writer.write(image)
    finally:
        writer.release()
    print(f"Video silencioso gerado: {video_path}")


if __name__ == "__main__":
    main()
