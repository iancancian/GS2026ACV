from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from boca_de_lobo.decision import assess_inspection
from boca_de_lobo.inspection_report import format_report, save_annotated_image
from boca_de_lobo.predict import main


def detection(name: str, box: list[int]) -> dict:
    return {"class_name": name, "confidence": 0.9, "bbox_xyxy_pixels": box}


def payload(detections: list[dict]) -> dict:
    return {
        "inspection_id": "id",
        "asset_id": "BUEIRO-001",
        "observed_at": "2026-01-01",
        "image_name": "foto.jpg",
        "model": {"weights": "best.pt", "confidence_threshold": 0.35},
        "detections": detections,
        "assessment": assess_inspection(detections, False, False),
    }


class InspectionReportTests(unittest.TestCase):
    def test_multiple_states_and_materials(self) -> None:
        detections = [
            detection("bueiro", [0, 0, 100, 100]),
            detection("bueiro", [200, 0, 300, 100]),
            detection("lixo", [0, 0, 20, 100]),
            detection("folha", [200, 0, 300, 100]),
            detection("pedra", [400, 0, 450, 50]),
        ]
        report = format_report(payload(detections))
        self.assertIn("Bueiro 1: parcialmente_obstruido | cobertura estimada: 20.0%", report)
        self.assertIn("Bueiro 2: entupido_provavel | cobertura estimada: 100.0%", report)
        self.assertIn("fora das caixas de bueiro: 1", report)
        self.assertIn("nao comprova", report)

    def test_no_drain_is_indeterminate(self) -> None:
        report = format_report(payload([detection("lixo", [0, 0, 20, 20])]))
        self.assertIn("Estado: indeterminado", report)
        self.assertIn("Prioridade sugerida: revisao_manual", report)

    def test_annotated_image_preserves_original(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "original.png"
            output = Path(directory) / "annotated.png"
            Image.new("RGB", (100, 100), "white").save(source)
            save_annotated_image(source, output, [detection("bueiro", [10, 20, 80, 90])])
            with Image.open(source) as original, Image.open(output) as annotated:
                self.assertEqual(original.getpixel((10, 20)), (255, 255, 255))
                self.assertNotEqual(annotated.getpixel((10, 20)), (255, 255, 255))

    def test_command_writes_all_outputs_and_protects_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, weights = root / "foto.png", root / "best.pt"
            Image.new("RGB", (100, 100), "white").save(source)
            weights.write_bytes(b"fake")
            output, report, annotated = root / "result.json", root / "result.txt", root / "result.png"
            arguments = [
                "predict", "--image", str(source), "--weights", str(weights), "--asset-id", "BUEIRO-001",
                "--output", str(output), "--report", str(report), "--annotated-image", str(annotated),
            ]
            detections = [detection("bueiro", [0, 0, 100, 100]), detection("lixo", [0, 0, 20, 100])]
            with patch("sys.argv", arguments), patch("boca_de_lobo.detector.load_detector", return_value=lambda _: detections):
                self.assertEqual(main(), 0)
            self.assertTrue(output.is_file())
            self.assertTrue(annotated.is_file())
            self.assertIn("parcialmente_obstruido", report.read_text(encoding="utf-8"))
            arguments[-1] = str(weights)
            with patch("sys.argv", arguments):
                self.assertEqual(main(), 2)
            self.assertEqual(weights.read_bytes(), b"fake")


if __name__ == "__main__":
    unittest.main()
