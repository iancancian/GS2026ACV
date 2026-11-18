"""Regras explicaveis de triagem; nao substituem decisao de agente publico."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from boca_de_lobo.classes import CLASS_NAMES, DRAIN_NAME, OBSTRUCTION_NAMES

# Limiares iniciais da convencao do projeto; calibrar com fotos locais.
PARTIAL_COVERAGE = 0.10
BLOCKED_COVERAGE = 0.80

STATE_FREE = "sem_obstrucao_visivel"
STATE_PARTIAL = "parcialmente_obstruido"
STATE_BLOCKED = "entupido_provavel"
_STATE_LEVEL = {STATE_FREE: 0, STATE_PARTIAL: 1, STATE_BLOCKED: 2}

Box = tuple[float, float, float, float]


def _area(box: Box) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _intersection(first: Box, second: Box) -> Box | None:
    left, top = max(first[0], second[0]), max(first[1], second[1])
    right, bottom = min(first[2], second[2]), min(first[3], second[3])
    return (left, top, right, bottom) if right > left and bottom > top else None


def _union_area(boxes: Sequence[Box]) -> float:
    """Area da uniao de retangulos, sem contar sobreposicoes duas vezes."""
    xs = sorted({coordinate for box in boxes for coordinate in (box[0], box[2])})
    area = 0.0
    for left, right in zip(xs, xs[1:]):
        spans = sorted((box[1], box[3]) for box in boxes if box[0] <= left and box[2] >= right)
        covered, end = 0.0, float("-inf")
        for top, bottom in spans:
            if bottom > end:
                covered += bottom - max(top, end)
                end = bottom
        area += (right - left) * covered
    return area


def drain_coverage(drain: Box, obstructions: Sequence[Box]) -> float:
    """Fracao da caixa do bueiro coberta por caixas de obstrucao (0 a 1)."""
    drain_area = _area(drain)
    if drain_area == 0:
        return 0.0
    overlaps = [
        overlap for box in obstructions if (overlap := _intersection(drain, box)) is not None
    ]
    return min(1.0, _union_area(overlaps) / drain_area)


def classify_coverage(coverage: float) -> str:
    if coverage >= BLOCKED_COVERAGE:
        return STATE_BLOCKED
    if coverage >= PARTIAL_COVERAGE:
        return STATE_PARTIAL
    return STATE_FREE


def _parse_detection(detection: dict[str, Any]) -> tuple[str, Box]:
    name = detection.get("class_name")
    if name not in CLASS_NAMES:
        raise ValueError(f"Classe desconhecida: {name}")
    values = detection.get("bbox_xyxy_pixels")
    if not isinstance(values, Sequence) or len(values) != 4:
        raise ValueError("bbox_xyxy_pixels deve conter quatro valores.")
    x1, y1, x2, y2 = (float(value) for value in values)
    return str(name), (x1, y1, x2, y2)


def assess_inspection(
    detections: Sequence[dict[str, Any]],
    strong_rain: bool,
    risk_area: bool,
) -> dict[str, object]:
    """Deriva estado do bueiro e prioridade a partir das deteccoes e do contexto."""
    parsed = [_parse_detection(detection) for detection in detections]
    drains = [box for name, box in parsed if name == DRAIN_NAME]
    obstructions = [(name, box) for name, box in parsed if name in OBSTRUCTION_NAMES]

    drain_reports: list[dict[str, object]] = []
    for index, drain in enumerate(drains):
        overlapping = [(name, box) for name, box in obstructions if _intersection(drain, box)]
        coverage = drain_coverage(drain, [box for _, box in overlapping])
        drain_reports.append(
            {
                "drain_index": index,
                "obstruction_coverage": round(coverage, 3),
                "state": classify_coverage(coverage),
                "materials": sorted({name for name, _ in overlapping}),
            }
        )

    outside_count = sum(
        1 for _, box in obstructions if not any(_intersection(drain, box) for drain in drains)
    )
    level = max((_STATE_LEVEL[str(report["state"])] for report in drain_reports), default=0)
    if outside_count:
        level = max(level, 1)

    if not drains:
        priority = "revisao_manual"
    elif level == 0:
        priority = "sem_acao_sugerida"
    elif strong_rain:
        priority = "alta" if level == 2 or risk_area else "media"
    else:
        priority = "programada"

    limitations = (
        "Nenhum bueiro detectado; confirmar em campo. Ausencia de deteccao nao significa bueiro livre."
        if not drains
        else "Cobertura estimada por caixas 2D; nao mede a obstrucao interna. Confirmar em campo."
    )
    return {
        "maintenance_priority": priority,
        "resident_warning_recommended": strong_rain and risk_area,
        "human_approval_required": True,
        "drain_count": len(drains),
        "drains": drain_reports,
        "obstruction_outside_drain_count": outside_count,
        "reason": {
            "strong_rain": strong_rain,
            "risk_area": risk_area,
            "limitations": limitations,
        },
    }
