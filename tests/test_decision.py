from __future__ import annotations

import unittest

from boca_de_lobo.decision import (
    STATE_BLOCKED,
    STATE_FREE,
    STATE_PARTIAL,
    assess_inspection,
    classify_coverage,
    drain_coverage,
)

DRAIN = (0.0, 0.0, 100.0, 100.0)


def detection(name: str, box: tuple[float, float, float, float]) -> dict[str, object]:
    return {"class_name": name, "confidence": 0.9, "bbox_xyxy_pixels": list(box)}


class CoverageTests(unittest.TestCase):
    def test_overlapping_obstructions_are_not_counted_twice(self) -> None:
        half = (0.0, 0.0, 50.0, 100.0)
        self.assertAlmostEqual(drain_coverage(DRAIN, [half, half]), 0.5)

    def test_obstruction_outside_drain_adds_no_coverage(self) -> None:
        self.assertEqual(drain_coverage(DRAIN, [(200.0, 0.0, 300.0, 100.0)]), 0.0)

    def test_state_thresholds(self) -> None:
        self.assertEqual(classify_coverage(0.09), STATE_FREE)
        self.assertEqual(classify_coverage(0.10), STATE_PARTIAL)
        self.assertEqual(classify_coverage(0.80), STATE_BLOCKED)


class InspectionDecisionTests(unittest.TestCase):
    def test_blocked_drain_under_strong_rain_is_high_priority(self) -> None:
        result = assess_inspection(
            [detection("bueiro", DRAIN), detection("lixo", DRAIN)], True, False
        )
        self.assertEqual(result["drains"][0]["state"], STATE_BLOCKED)
        self.assertEqual(result["drains"][0]["materials"], ["lixo"])
        self.assertEqual(result["maintenance_priority"], "alta")
        self.assertTrue(result["human_approval_required"])

    def test_partial_obstruction_escalates_only_in_risk_area(self) -> None:
        detections = [detection("bueiro", DRAIN), detection("folha", (0.0, 0.0, 20.0, 100.0))]
        self.assertEqual(assess_inspection(detections, True, False)["maintenance_priority"], "media")
        self.assertEqual(assess_inspection(detections, True, True)["maintenance_priority"], "alta")
        self.assertEqual(
            assess_inspection(detections, False, True)["maintenance_priority"], "programada"
        )

    def test_debris_near_but_outside_drain_is_scheduled_for_cleaning(self) -> None:
        result = assess_inspection(
            [detection("bueiro", DRAIN), detection("lixo", (200.0, 0.0, 260.0, 40.0))], False, False
        )
        self.assertEqual(result["drains"][0]["state"], STATE_FREE)
        self.assertEqual(result["obstruction_outside_drain_count"], 1)
        self.assertEqual(result["maintenance_priority"], "programada")

    def test_clean_drain_needs_no_action_and_warning_needs_rain_and_risk(self) -> None:
        clean = [detection("bueiro", DRAIN)]
        self.assertEqual(assess_inspection(clean, False, False)["maintenance_priority"], "sem_acao_sugerida")
        self.assertTrue(assess_inspection(clean, True, True)["resident_warning_recommended"])
        self.assertFalse(assess_inspection(clean, True, False)["resident_warning_recommended"])

    def test_no_drain_requires_manual_review(self) -> None:
        result = assess_inspection([detection("lixo", DRAIN)], True, True)
        self.assertEqual(result["maintenance_priority"], "revisao_manual")
        self.assertIn("nao significa bueiro livre", result["reason"]["limitations"])

    def test_rejects_unknown_class_and_malformed_box(self) -> None:
        with self.assertRaises(ValueError):
            assess_inspection([detection("tampa", DRAIN)], False, False)
        with self.assertRaises(ValueError):
            assess_inspection([{"class_name": "bueiro", "bbox_xyxy_pixels": [0, 0, 1]}], False, False)


if __name__ == "__main__":
    unittest.main()
