from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from boca_de_lobo.simulate import build_manifest


class SimulateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        self.images, self.labels = root / "images", root / "labels"
        self.images.mkdir()
        self.labels.mkdir()
        for index, label in enumerate(["0 0.5 0.5 0.2 0.2\n", "1 0.5 0.5 0.2 0.2\n", "", "1 0.1 0.1 0.1 0.1\n0 0.5 0.5 0.3 0.3\n"]):
            Image.new("RGB", (2, 2)).save(self.images / f"img{index}.jpg")
            (self.labels / f"img{index}.txt").write_text(label, encoding="utf-8")

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_only_photos_with_a_drain_are_selected_and_areas_rotate(self) -> None:
        rows = build_manifest(self.images, ["A", "B"], 10, 1, False, self.labels)
        self.assertEqual(sorted(row["image_file"] for row in rows), ["img0.jpg", "img3.jpg"])
        self.assertEqual({row["area_id"] for row in rows}, {"A", "B"})
        self.assertEqual([row["asset_id"] for row in rows], ["BUEIRO-001", "BUEIRO-002"])

    def test_same_seed_gives_same_selection(self) -> None:
        first = build_manifest(self.images, ["A"], 1, 7, False)
        self.assertEqual(first, build_manifest(self.images, ["A"], 1, 7, False))

    def test_inject_faults_appends_four_defective_rows(self) -> None:
        clean = build_manifest(self.images, ["A"], 2, 1, False, self.labels)
        faulty = build_manifest(self.images, ["A"], 2, 1, True, self.labels)
        self.assertEqual(len(faulty), len(clean) + 4)
        self.assertEqual(faulty[-1], faulty[0])

    def test_fails_when_nothing_is_eligible(self) -> None:
        for path in self.labels.glob("*.txt"):
            path.write_text("2 0.5 0.5 0.1 0.1\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            build_manifest(self.images, ["A"], 3, 1, False, self.labels)


if __name__ == "__main__":
    unittest.main()
