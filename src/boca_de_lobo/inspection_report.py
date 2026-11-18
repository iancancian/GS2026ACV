"""Apresentacao legivel de uma inspecao e desenho das deteccoes."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageOps

from boca_de_lobo.classes import OBSTRUCTION_NAMES

COLORS = {
    "bueiro": "#00cc66",
    "lixo": "#ff4444",
    "folha": "#ffcc00",
    "pedra": "#00aaff",
    "sedimento": "#ff88ff",
}


def format_report(payload: dict[str, Any]) -> str:
    assessment = payload["assessment"]
    counts = Counter(d["class_name"] for d in payload["detections"] if d["class_name"] in OBSTRUCTION_NAMES)
    lines = [
        "RELATORIO DE INSPECAO VISUAL - BOCA DE LOBO",
        f"Inspecao: {payload['inspection_id']}",
        f"Ativo: {payload['asset_id']}",
        f"Data UTC: {payload['observed_at']}",
        f"Imagem: {payload['image_name']}",
        f"Modelo: {payload['model']['weights']} | confianca minima: {payload['model']['confidence_threshold']}",
        f"Bueiros detectados: {assessment['drain_count']}",
        "Deteccoes de material na imagem: "
        + (", ".join(f"{name}: {count}" for name, count in sorted(counts.items())) if counts else "nenhuma"),
        f"Deteccoes de material fora das caixas de bueiro: {assessment['obstruction_outside_drain_count']}",
    ]
    for drain in assessment["drains"]:
        materials = ", ".join(drain["materials"]) or "nenhum detectado"
        lines.append(
            f"Bueiro {drain['drain_index'] + 1}: {drain['state']} | "
            f"cobertura estimada: {drain['obstruction_coverage']:.1%} | materiais sobrepostos: {materials}"
        )
    if not assessment["drains"]:
        lines.append("Estado: indeterminado; nenhum bueiro detectado, requer revisao manual.")
    lines.extend(
        [
            f"Prioridade sugerida: {assessment['maintenance_priority']}",
            "Contexto informado: chuva forte = "
            f"{assessment['reason']['strong_rain']}; area de risco = {assessment['reason']['risk_area']}.",
            "Regra experimental: abaixo de 10% sem obstrucao visivel; de 10% a menos de 80% "
            "parcialmente obstruido; a partir de 80% entupido provavel.",
            "Cobertura e uniao das caixas de materiais dentro da caixa do bueiro, nao area real de sujeira.",
            "Contagens sao deteccoes, nao volume ou peso. Materiais proximos podem nao tocar a caixa.",
            "A ausencia de deteccao nao comprova que o bueiro esta livre; o modelo pode falhar.",
            "Os limiares nao foram calibrados; a obstrucao interna nao e visivel. Confirmar em campo.",
            "Triagem experimental: aprovacao humana obrigatoria antes de qualquer acao.",
        ]
    )
    return "\n".join(lines) + "\n"


def save_annotated_image(image_path: Path, output_path: Path, detections: list[dict[str, Any]]) -> None:
    with Image.open(image_path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    draw = ImageDraw.Draw(image)
    width = max(2, round(min(image.size) / 300))
    for detection in detections:
        box = tuple(detection["bbox_xyxy_pixels"])
        color = COLORS[detection["class_name"]]
        draw.rectangle(box, outline=color, width=width)
        label = f"{detection['class_name']} {detection['confidence']:.2f}"
        left, top = box[:2]
        text_box = draw.textbbox((left, max(0, top - 14)), label)
        draw.rectangle(text_box, fill="black")
        draw.text(text_box[:2], label, fill=color)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
