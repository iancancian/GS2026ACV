from __future__ import annotations

import csv
import re
import tempfile
import unittest
import zlib
from pathlib import Path

from PIL import Image

from boca_de_lobo.classes import CLASS_NAMES
from boca_de_lobo.dataset import SPLITS, validate_dataset


class DatasetValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        for split in SPLITS:
            (self.root / "images" / split).mkdir(parents=True)
            (self.root / "labels" / split).mkdir(parents=True)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _add_image(
        self,
        split: str,
        name: str,
        label: str = "0 0.5 0.5 0.4 0.4\n",
        site_group: str = "quadra-01",
        asset_id: str | None = None,
        color: tuple[int, int, int] | None = None,
    ) -> None:
        image_name = f"{name}.png"
        image_path = self.root / "images" / split / image_name
        seed = zlib.crc32(name.encode())
        pixel = color or (seed & 255, (seed >> 8) & 255, (seed >> 16) & 255)
        Image.new("RGB", (2, 2), color=pixel).save(image_path)
        (self.root / "labels" / split / f"{name}.txt").write_text(label, encoding="utf-8")
        metadata_path = self.root / "metadata.csv"
        write_header = not metadata_path.exists()
        with metadata_path.open("a", encoding="utf-8", newline="") as metadata_file:
            writer = csv.DictWriter(
                metadata_file,
                fieldnames=[
                    "image_path",
                    "site_group",
                    "asset_id",
                    "captured_at",
                    "lighting",
                    "weather",
                    "annotator_code",
                ],
            )
            if write_header:
                writer.writeheader()
            writer.writerow(
                {
                    "image_path": f"images/{split}/{image_name}",
                    "site_group": site_group,
                    "asset_id": asset_id or name,
                    "captured_at": "2026-10-01T10:00:00-03:00",
                    "lighting": "dia",
                    "weather": "seco",
                    "annotator_code": "A1",
                }
            )

    def test_empty_scaffold_is_valid_before_collection(self) -> None:
        self.assertEqual(validate_dataset(self.root), [])

    def test_valid_image_label_and_metadata_pass(self) -> None:
        for index, split in enumerate(SPLITS, start=1):
            self._add_image(split, f"asset-00{index}", site_group=f"setor-{index}")
        self.assertEqual(validate_dataset(self.root), [])

    def test_yolo_yaml_class_ids_match_shared_taxonomy(self) -> None:
        config_path = Path(__file__).resolve().parents[1] / "dataset" / "data.yaml"
        configured_names = re.findall(r"^\s+(\d+):\s+([a-z_]+)$", config_path.read_text(), re.M)
        self.assertEqual(
            tuple(name for _, name in sorted(configured_names, key=lambda pair: int(pair[0]))),
            CLASS_NAMES,
        )

    def test_rejects_out_of_range_class_and_coordinates(self) -> None:
        self._add_image("train", "asset-001", "5 1.2 0.5 0.4 0.4\n")
        errors = validate_dataset(self.root)
        self.assertTrue(any("class_id 5" in error for error in errors))
        self.assertTrue(any("normalizadas" in error for error in errors))

    def test_unknown_site_group_is_allowed_across_splits_but_asset_id_is_not(self) -> None:
        self._add_image("train", "a-001", site_group="desconhecido", asset_id="shared")
        self._add_image("val", "a-002", site_group="desconhecido", asset_id="other")
        self._add_image("test", "a-003", site_group="desconhecido", asset_id="third")
        self.assertEqual(validate_dataset(self.root), [])
        self._add_image("test", "a-004", site_group="desconhecido", asset_id="shared")
        self.assertTrue(any("asset_id shared aparece" in e for e in validate_dataset(self.root)))

    def test_rejects_identical_image_content_in_different_splits(self) -> None:
        self._add_image("train", "a-001", site_group="setor-1", color=(1, 2, 3))
        self._add_image("val", "a-002", site_group="setor-2", color=(1, 2, 3))
        self._add_image("test", "a-003", site_group="setor-3")
        errors = validate_dataset(self.root)
        self.assertTrue(any("conteudo identico" in error for error in errors))

    def test_rejects_same_site_in_multiple_splits(self) -> None:
        self._add_image("train", "asset-001", site_group="setor-a")
        self._add_image("val", "asset-002", site_group="setor-a")
        errors = validate_dataset(self.root)
        self.assertTrue(any("site_group setor-a aparece" in error for error in errors))

    def test_rejects_missing_metadata_for_collected_image(self) -> None:
        Image.new("RGB", (2, 2)).save(self.root / "images" / "test" / "asset-003.png")
        (self.root / "labels" / "test" / "asset-003.txt").write_text(
            "2 0.5 0.5 0.5 0.5\n", encoding="utf-8"
        )
        errors = validate_dataset(self.root)
        self.assertTrue(any("metadata.csv: obrigatorio" in error for error in errors))

    def test_rejects_unsupported_image_format(self) -> None:
        image_dir = self.root / "images" / "train"
        image_dir.joinpath("asset-001.bmp").write_bytes(b"not used by this pipeline")
        errors = validate_dataset(self.root)
        self.assertTrue(any("formato nao suportado" in error for error in errors))

    def test_rejects_corrupt_image_content(self) -> None:
        image_path = self.root / "images" / "train" / "asset-001.png"
        image_path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 20)
        (self.root / "labels" / "train" / "asset-001.txt").write_text(
            "0 0.5 0.5 0.4 0.4\n", encoding="utf-8"
        )
        errors = validate_dataset(self.root)
        self.assertTrue(any("arquivo vazio, corrompido" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
