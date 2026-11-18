"""Automacao RPA: lote de inspecoes -> deteccao -> decisao -> fila de acoes para o MVP."""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import os
import sys
import tempfile
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from PIL import Image

from boca_de_lobo.decision import assess_inspection
from boca_de_lobo.detector import Detector
from boca_de_lobo.paths import PROJECT_ROOT
from boca_de_lobo.weather import (
    STRONG_RAIN_MM_PER_HOUR,
    Forecast,
    ForecastUnavailable,
    fetch_forecast,
)

LOGGER = logging.getLogger("boca_de_lobo.automation")

SCHEMA_VERSION = "1.0"
PRIORITY_ORDER = {"alta": 0, "media": 1, "programada": 2, "revisao_manual": 3, "sem_acao_sugerida": 4}
INSPECTION_COLUMNS = {"asset_id", "image_file", "area_id"}
AREA_COLUMNS = {"area_id", "area_name", "high_risk", "latitude", "longitude"}
TRUE_VALUES = {"true", "1", "sim", "yes"}
FALSE_VALUES = {"false", "0", "nao", "não", "no"}
TASK_COLUMNS = (
    "asset_id",
    "area_id",
    "priority",
    "state",
    "obstruction_coverage",
    "materials",
    "forecast_status",
    "human_approval_required",
)

ForecastProvider = Callable[[float, float], Forecast]


class AutomationError(Exception):
    """Falha que impede o lote inteiro (entrada ilegivel ou sem areas validas)."""


@dataclass(frozen=True)
class Area:
    area_id: str
    name: str
    high_risk: bool
    latitude: float
    longitude: float


@dataclass(frozen=True)
class RunResult:
    payload: dict[str, Any]
    output_files: dict[str, Path]


def _read_csv(path: Path, required: set[str]) -> list[dict[str, str]]:
    try:
        with path.open(encoding="utf-8-sig", newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                raise AutomationError(f"{path.name}: cabecalho deve conter {', '.join(sorted(required))}.")
            return list(reader)
    except (OSError, UnicodeDecodeError, csv.Error) as error:
        raise AutomationError(f"{path.name}: nao foi possivel ler ({error}).") from error


def load_areas(path: Path) -> dict[str, Area]:
    """Le areas de risco; linhas invalidas ou duplicadas sao registradas e ignoradas."""
    areas: dict[str, Area] = {}
    for line_number, row in enumerate(_read_csv(path, AREA_COLUMNS), start=2):
        area_id = (row.get("area_id") or "").strip()
        risk_text = (row.get("high_risk") or "").strip().casefold()
        try:
            latitude, longitude = float(row["latitude"]), float(row["longitude"])
            valid_coordinates = -90 <= latitude <= 90 and -180 <= longitude <= 180
        except (TypeError, ValueError):
            valid_coordinates = False
        if not area_id or risk_text not in TRUE_VALUES | FALSE_VALUES or not valid_coordinates:
            LOGGER.warning("%s:%d area invalida ignorada (%s).", path.name, line_number, area_id or "sem id")
            continue
        if area_id in areas:
            LOGGER.warning("%s:%d area duplicada ignorada (%s).", path.name, line_number, area_id)
            continue
        areas[area_id] = Area(
            area_id=area_id,
            name=(row.get("area_name") or area_id).strip(),
            high_risk=risk_text in TRUE_VALUES,
            latitude=latitude,
            longitude=longitude,
        )
    return areas


def _collect_forecasts(
    areas: dict[str, Area], provider: ForecastProvider, rain_threshold: float
) -> dict[str, dict[str, Any]]:
    """Uma consulta por area; falha vira status 'indisponivel', nunca 'sem chuva'."""
    forecasts: dict[str, dict[str, Any]] = {}
    for area in areas.values():
        try:
            forecast = provider(area.latitude, area.longitude)
            forecasts[area.area_id] = {
                "status": "ok",
                "max_hourly_mm": forecast.max_hourly_mm,
                "hours": forecast.hours,
                "strong_rain": forecast.is_strong_rain(rain_threshold),
            }
            LOGGER.info(
                "Previsao %s: max %.1f mm/h em %dh (chuva forte: %s).",
                area.area_id,
                forecast.max_hourly_mm,
                forecast.hours,
                forecasts[area.area_id]["strong_rain"],
            )
        except (ForecastUnavailable, ValueError) as error:
            LOGGER.error("Previsao %s indisponivel: %s", area.area_id, error)
            forecasts[area.area_id] = {
                "status": "indisponivel",
                "max_hourly_mm": None,
                "hours": 0,
                "strong_rain": False,
            }
    return forecasts


def _resolve_image(images_dir: Path, image_file: str) -> Path | None:
    """Impede que o CSV aponte para fora da pasta de imagens."""
    root = images_dir.resolve()
    candidate = (root / image_file).resolve()
    return candidate if root in candidate.parents else None


def _is_readable_image(path: Path) -> bool:
    try:
        with Image.open(path) as image:
            image.verify()
    except (OSError, SyntaxError, Image.DecompressionBombError):
        return False
    return True


def _item_summary(item: dict[str, Any]) -> tuple[str, float | str, str]:
    drains = item["assessment"]["drains"]
    if not drains:
        return "bueiro_nao_detectado", "", ""
    worst = max(drains, key=lambda drain: drain["obstruction_coverage"])
    return worst["state"], worst["obstruction_coverage"], "|".join(worst["materials"])


def _write_atomic(path: Path, content: str) -> None:
    """Grava via arquivo temporario para o MVP nunca ler um arquivo pela metade."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="", dir=path.parent, suffix=".tmp", delete=False
    ) as temp_file:
        temp_file.write(content)
    os.replace(temp_file.name, path)


def _tasks_csv(items: list[dict[str, Any]]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=TASK_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for item in items:
        priority = item["assessment"]["maintenance_priority"]
        if priority == "sem_acao_sugerida":
            continue
        state, coverage, materials = _item_summary(item)
        writer.writerow(
            {
                "asset_id": item["asset_id"],
                "area_id": item["area_id"],
                "priority": priority,
                "state": state,
                "obstruction_coverage": coverage,
                "materials": materials,
                "forecast_status": item["forecast_status"],
                "human_approval_required": True,
            }
        )
    return buffer.getvalue()


def run_automation(
    *,
    inspections_csv: Path,
    areas_csv: Path,
    images_dir: Path,
    output_dir: Path,
    detector: Detector,
    forecast_provider: ForecastProvider = fetch_forecast,
    rain_threshold: float = STRONG_RAIN_MM_PER_HOUR,
) -> RunResult:
    run_id = str(uuid4())
    LOGGER.info("Execucao %s iniciada.", run_id)
    areas = load_areas(areas_csv)
    if not areas:
        raise AutomationError("Nenhuma area de risco valida; abortando.")
    forecasts = _collect_forecasts(areas, forecast_provider, rain_threshold)
    rows = _read_csv(inspections_csv, INSPECTION_COLUMNS)

    items: list[dict[str, Any]] = []
    rejected: list[dict[str, str]] = []
    seen: dict[str, tuple[str, str]] = {}

    def reject(position: int, asset_id: str, reason: str, detail: str = "") -> None:
        rejected.append({"line": str(position + 1), "asset_id": asset_id, "reason": reason, "detail": detail})
        LOGGER.warning(
            "[%d/%d] %s rejeitado: %s%s",
            position,
            len(rows),
            asset_id or "-",
            reason,
            f" {detail}" if detail else "",
        )

    for position, row in enumerate(rows, start=1):
        asset_id = (row.get("asset_id") or "").strip()
        image_file = (row.get("image_file") or "").strip()
        area_id = (row.get("area_id") or "").strip()

        if not (asset_id and image_file and area_id):
            reject(position, asset_id, "registro_incompleto")
            continue
        if area_id not in areas:
            reject(position, asset_id, "area_desconhecida", area_id)
            continue
        if asset_id in seen:
            same = seen[asset_id] == (image_file, area_id)
            reject(position, asset_id, "ativo_duplicado" if same else "ativo_inconsistente")
            continue
        seen[asset_id] = (image_file, area_id)

        image_path = _resolve_image(images_dir, image_file)
        if image_path is None:
            reject(position, asset_id, "caminho_invalido", image_file)
            continue
        if not image_path.is_file():
            reject(position, asset_id, "imagem_ausente", image_file)
            continue
        if not _is_readable_image(image_path):
            reject(position, asset_id, "imagem_invalida", image_file)
            continue

        try:
            detections = detector(image_path)
            forecast = forecasts[area_id]
            assessment = assess_inspection(detections, forecast["strong_rain"], areas[area_id].high_risk)
        except Exception as error:  # um item defeituoso nao pode derrubar o lote
            reject(position, asset_id, "falha_deteccao", type(error).__name__)
            continue

        item = {
            "asset_id": asset_id,
            "area_id": area_id,
            "image_file": image_file,
            "forecast_status": forecast["status"],
            "detections": detections,
            "assessment": assessment,
        }
        items.append(item)
        state, coverage, _ = _item_summary(item)
        LOGGER.info(
            "[%d/%d] %s processado: prioridade=%s estado=%s cobertura=%s",
            position,
            len(rows),
            asset_id,
            assessment["maintenance_priority"],
            state,
            coverage if coverage != "" else "-",
        )

    items.sort(key=lambda item: (PRIORITY_ORDER[item["assessment"]["maintenance_priority"]], item["asset_id"]))
    alerts = [
        {
            "area_id": area.area_id,
            "area_name": area.name,
            "status": "pendente_aprovacao",
            "forecast_max_hourly_mm": forecasts[area.area_id]["max_hourly_mm"],
            "message": (
                f"Previsao de chuva forte para {area.name}. "
                "Evite passar por trechos baixos e areas que costumam alagar."
            ),
        }
        for area in areas.values()
        if forecasts[area.area_id]["strong_rain"] and area.high_risk
    ]
    priorities = Counter(item["assessment"]["maintenance_priority"] for item in items)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "rain_threshold_mm_per_hour": rain_threshold,
        "summary": {
            "total_registros": len(rows),
            "processados": len(items),
            "rejeitados": len(rejected),
            "areas_sem_previsao": sum(1 for f in forecasts.values() if f["status"] != "ok"),
            "por_prioridade": dict(priorities),
        },
        "forecasts": forecasts,
        "items": items,
        "rejected": rejected,
        "resident_alerts": alerts,
    }

    output_files = {
        "action_queue": output_dir / "action_queue.json",
        "cleaning_tasks": output_dir / "tarefas_limpeza.csv",
        "resident_alerts": output_dir / "avisos_moradores.json",
    }
    _write_atomic(output_files["action_queue"], json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    _write_atomic(output_files["cleaning_tasks"], _tasks_csv(items))
    _write_atomic(
        output_files["resident_alerts"],
        json.dumps(
            {"schema_version": SCHEMA_VERSION, "generated_at": payload["generated_at"], "alerts": alerts},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
    )
    LOGGER.info(
        "Execucao %s concluida: %d processados, %d rejeitados, %d aviso(s) pendente(s).",
        run_id,
        len(items),
        len(rejected),
        len(alerts),
    )
    return RunResult(payload=payload, output_files=output_files)


def load_action_queue(path: Path) -> dict[str, Any]:
    """Lado consumidor (MVP): valida o contrato antes de usar a fila de acoes."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Fila de acoes ilegivel: {error}") from error
    if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("schema_version incompativel.")
    for key in ("run_id", "summary", "items", "rejected", "resident_alerts"):
        if key not in payload:
            raise ValueError(f"Campo obrigatorio ausente: {key}")
    for item in payload["items"]:
        assessment = item.get("assessment", {})
        if assessment.get("maintenance_priority") not in PRIORITY_ORDER:
            raise ValueError(f"Prioridade invalida em {item.get('asset_id')}.")
        if assessment.get("human_approval_required") is not True:
            raise ValueError(f"Item sem aprovacao humana: {item.get('asset_id')}.")
    return payload


def _configure_logging(log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_file, encoding="utf-8")],
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Executa a automacao de inspecao de bocas de lobo.")
    parser.add_argument("--inspections", type=Path, default=PROJECT_ROOT / "data" / "inspecoes_exemplo.csv")
    parser.add_argument("--areas", type=Path, default=PROJECT_ROOT / "data" / "areas_risco.csv")
    parser.add_argument("--images-dir", type=Path, default=PROJECT_ROOT / "dataset" / "images" / "test")
    parser.add_argument(
        "--weights",
        type=Path,
        default=PROJECT_ROOT / "models" / "best.pt",
    )
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--rain-threshold", type=float, default=STRONG_RAIN_MM_PER_HOUR)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "output")
    parser.add_argument("--log-file", type=Path, default=PROJECT_ROOT / "logs" / "automation.log")
    args = parser.parse_args(argv)
    _configure_logging(args.log_file)

    if not 0 < args.conf <= 1 or args.rain_threshold <= 0:
        LOGGER.error("--conf deve estar em (0, 1] e --rain-threshold deve ser positivo.")
        return 2
    if not args.weights.is_file():
        LOGGER.error("Pesos nao encontrados: %s. Treine antes com boca_de_lobo.train.", args.weights.name)
        return 2

    try:
        from boca_de_lobo.detector import load_detector

        detector = load_detector(args.weights, args.conf)
        result = run_automation(
            inspections_csv=args.inspections,
            areas_csv=args.areas,
            images_dir=args.images_dir,
            output_dir=args.output_dir,
            detector=detector,
            rain_threshold=args.rain_threshold,
        )
    except ImportError:
        LOGGER.error("Dependencia ausente. Instale com: pip install -e .")
        return 2
    except AutomationError as error:
        LOGGER.error("Execucao abortada: %s", error)
        return 2

    return 0 if result.payload["summary"]["processados"] else 1


if __name__ == "__main__":
    sys.exit(main())
