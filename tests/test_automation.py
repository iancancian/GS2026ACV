from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from boca_de_lobo.automation import (
    AutomationError,
    load_action_queue,
    load_areas,
    run_automation,
)
from boca_de_lobo.weather import Forecast, ForecastUnavailable

DRAIN_BOX = [0.0, 0.0, 100.0, 100.0]


def detection(name: str, box: list[float]) -> dict[str, object]:
    return {"class_id": 0, "class_name": name, "confidence": 0.9, "bbox_xyxy_pixels": box}


def fake_detector(image: Path) -> list[dict[str, object]]:
    name = image.name
    if name == "blocked.jpg":
        return [detection("bueiro", DRAIN_BOX), detection("lixo", DRAIN_BOX)]
    if name == "clean.jpg":
        return [detection("bueiro", DRAIN_BOX)]
    if name == "partial.jpg":
        return [detection("bueiro", DRAIN_BOX), detection("folha", [0.0, 0.0, 20.0, 100.0])]
    if name == "boom.jpg":
        raise RuntimeError("modelo falhou com dados internos")
    return []


def fake_forecast(latitude: float, longitude: float) -> Forecast:
    if latitude == -23.0:
        return Forecast(max_hourly_mm=12.0, hours=24)
    if latitude == -24.0:
        return Forecast(max_hourly_mm=0.4, hours=24)
    raise ForecastUnavailable("sem rede")


class AutomationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.images = self.root / "images"
        self.images.mkdir()
        for name in ("blocked", "clean", "none", "boom", "partial"):
            Image.new("RGB", (4, 4)).save(self.images / f"{name}.jpg")
        (self.images / "corrupt.jpg").write_bytes(b"nao e imagem")
        self.areas_csv = self.root / "areas.csv"
        self._write(
            self.areas_csv,
            ["area_id", "area_name", "high_risk", "latitude", "longitude"],
            [
                ["AREA-A", "Setor A", "true", "-23.0", "-46.0"],
                ["AREA-B", "Setor B", "false", "-24.0", "-46.0"],
                ["AREA-C", "Setor C", "true", "-25.0", "-46.0"],
            ],
        )
        self.output = self.root / "output"

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    @staticmethod
    def _write(path: Path, header: list[str], rows: list[list[str]]) -> None:
        with path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.writer(file, lineterminator="\n")
            writer.writerow(header)
            writer.writerows(rows)

    def _run(self, rows: list[list[str]], provider=fake_forecast):  # noqa: ANN001, ANN202
        inspections = self.root / "inspections.csv"
        self._write(inspections, ["asset_id", "image_file", "area_id"], rows)
        return run_automation(
            inspections_csv=inspections,
            areas_csv=self.areas_csv,
            images_dir=self.images,
            output_dir=self.output,
            detector=fake_detector,
            forecast_provider=provider,
        )

    def test_success_and_decision_produce_integrated_outputs(self) -> None:
        result = self._run(
            [
                ["B-1", "blocked.jpg", "AREA-A"],
                ["B-2", "clean.jpg", "AREA-B"],
                ["B-3", "none.jpg", "AREA-A"],
            ]
        )
        priorities = {item["asset_id"]: item["assessment"]["maintenance_priority"] for item in result.payload["items"]}
        self.assertEqual(priorities, {"B-1": "alta", "B-2": "sem_acao_sugerida", "B-3": "revisao_manual"})
        self.assertEqual([item["asset_id"] for item in result.payload["items"]], ["B-1", "B-3", "B-2"])

        queue = load_action_queue(result.output_files["action_queue"])
        self.assertEqual(queue["summary"]["processados"], 3)

        tasks = result.output_files["cleaning_tasks"].read_text(encoding="utf-8")
        self.assertIn("B-1,AREA-A,alta,entupido_provavel", tasks)
        self.assertNotIn("B-2", tasks)

        alerts = json.loads(result.output_files["resident_alerts"].read_text(encoding="utf-8"))["alerts"]
        self.assertEqual([alert["area_id"] for alert in alerts], ["AREA-A"])
        self.assertEqual(alerts[0]["status"], "pendente_aprovacao")
        self.assertEqual(list(self.output.glob("*.tmp")), [])

    def test_item_failures_are_recorded_and_batch_continues(self) -> None:
        with self.assertLogs("boca_de_lobo.automation", level="WARNING") as logs:
            result = self._run(
                [
                    ["OK-1", "clean.jpg", "AREA-A"],
                    ["X-1", "ausente.jpg", "AREA-A"],
                    ["X-2", "clean.jpg", "AREA-ZZ"],
                    ["OK-1", "clean.jpg", "AREA-A"],
                    ["OK-1", "blocked.jpg", "AREA-B"],
                    ["X-3", "../fora.jpg", "AREA-A"],
                    ["X-4", "corrupt.jpg", "AREA-A"],
                    ["X-5", "boom.jpg", "AREA-A"],
                    ["", "clean.jpg", "AREA-A"],
                ]
            )
        reasons = [entry["reason"] for entry in result.payload["rejected"]]
        self.assertEqual(
            reasons,
            [
                "imagem_ausente",
                "area_desconhecida",
                "ativo_duplicado",
                "ativo_inconsistente",
                "caminho_invalido",
                "imagem_invalida",
                "falha_deteccao",
                "registro_incompleto",
            ],
        )
        self.assertEqual(result.payload["summary"]["processados"], 1)
        joined = "\n".join(logs.output)
        self.assertIn("falha_deteccao RuntimeError", joined)
        self.assertNotIn("dados internos", joined)
        self.assertNotIn(str(self.root), joined)

    def test_forecast_outage_is_explicit_and_never_means_no_rain(self) -> None:
        result = self._run([["C-1", "blocked.jpg", "AREA-C"]])
        item = result.payload["items"][0]
        self.assertEqual(item["forecast_status"], "indisponivel")
        self.assertEqual(result.payload["summary"]["areas_sem_previsao"], 1)
        alerted_areas = [alert["area_id"] for alert in result.payload["resident_alerts"]]
        self.assertNotIn("AREA-C", alerted_areas)
        self.assertEqual(item["assessment"]["maintenance_priority"], "programada")

    def test_rain_threshold_is_configurable(self) -> None:
        inspections = self.root / "inspections.csv"
        self._write(inspections, ["asset_id", "image_file", "area_id"], [["P-1", "partial.jpg", "AREA-B"]])

        def run(threshold: float) -> str:
            result = run_automation(
                inspections_csv=inspections,
                areas_csv=self.areas_csv,
                images_dir=self.images,
                output_dir=self.output,
                detector=fake_detector,
                forecast_provider=fake_forecast,
                rain_threshold=threshold,
            )
            return result.payload["items"][0]["assessment"]["maintenance_priority"]

        self.assertEqual(run(7.6), "programada")
        self.assertEqual(run(0.3), "media")

    def test_fatal_input_errors_abort_the_batch(self) -> None:
        with self.assertRaises(AutomationError):
            run_automation(
                inspections_csv=self.root / "nao_existe.csv",
                areas_csv=self.areas_csv,
                images_dir=self.images,
                output_dir=self.output,
                detector=fake_detector,
                forecast_provider=fake_forecast,
            )
        self._write(self.areas_csv, ["area_id", "area_name", "high_risk", "latitude", "longitude"], [["A", "x", "talvez", "0", "0"]])
        with self.assertRaisesRegex(AutomationError, "Nenhuma area"):
            self._run([["B-1", "clean.jpg", "A"]])

    def test_invalid_area_rows_are_skipped_not_fatal(self) -> None:
        self._write(
            self.areas_csv,
            ["area_id", "area_name", "high_risk", "latitude", "longitude"],
            [["A", "ok", "sim", "-23.5", "-46.6"], ["A", "dup", "sim", "0", "0"], ["B", "ruim", "sim", "999", "0"]],
        )
        self.assertEqual(list(load_areas(self.areas_csv)), ["A"])

    def test_consumer_rejects_incompatible_or_unapproved_queue(self) -> None:
        result = self._run([["B-1", "clean.jpg", "AREA-B"]])
        path = result.output_files["action_queue"]
        payload = json.loads(path.read_text(encoding="utf-8"))

        payload["schema_version"] = "9.9"
        path.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "schema_version"):
            load_action_queue(path)

        payload["schema_version"] = "1.0"
        payload["items"][0]["assessment"]["human_approval_required"] = False
        path.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "aprovacao humana"):
            load_action_queue(path)


if __name__ == "__main__":
    unittest.main()
