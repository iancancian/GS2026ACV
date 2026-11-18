from __future__ import annotations

import tempfile
import unittest
import zipfile
from pathlib import Path

from PIL import Image

from boca_de_lobo.classes import CLASS_NAMES
from boca_de_lobo.dataset import SPLITS, validate_dataset
from boca_de_lobo.import_roboflow import import_dataset

SOURCE_NAMES = ["drainage", "leaf", "rock", "silt", "trash"]
SOURCE_SPLIT_DIRS = {"train": "train", "val": "valid", "test": "test"}


class RoboflowImportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.source = self.root / "export"
        self.target = self.root / "dataset"
        for split in SPLITS:
            (self.source / SOURCE_SPLIT_DIRS[split] / "images").mkdir(parents=True)
            (self.source / SOURCE_SPLIT_DIRS[split] / "labels").mkdir(parents=True)
            (self.target / "images" / split).mkdir(parents=True)
            (self.target / "labels" / split).mkdir(parents=True)
        names = "\n".join(f"  - {name}" for name in SOURCE_NAMES)
        (self.source / "data.yaml").write_text(f"names:\n{names}\n", encoding="utf-8")

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _add(self, split: str, stem: str, label: str | None, shade: int) -> None:
        split_dir = self.source / SOURCE_SPLIT_DIRS[split]
        Image.new("RGB", (4, 4), color=(shade, 10, 20)).save(split_dir / "images" / f"{stem}.jpg")
        if label is not None:
            (split_dir / "labels" / f"{stem}.txt").write_text(label, encoding="utf-8")

    def _populate(self) -> None:
        self._add("train", "a_jpg.rf.h1", "0 0.5 0.5 0.4 0.4\n4 0.2 0.2 0.1 0.1\n", 10)
        self._add("val", "b_jpg.rf.h2", "1 0.5 0.5 0.4 0.4\n", 20)
        self._add("test", "c_jpg.rf.h3", None, 30)

    def test_maps_classes_by_name_not_by_source_id(self) -> None:
        self._populate()
        counts = import_dataset(self.source, self.target)

        label = next((self.target / "labels" / "train").glob("*.txt")).read_text()
        self.assertEqual(label.splitlines()[0].split()[0], str(CLASS_NAMES.index("bueiro")))
        self.assertEqual(label.splitlines()[1].split()[0], str(CLASS_NAMES.index("lixo")))
        self.assertEqual(counts["folha"], 1)
        self.assertEqual(counts["imagens_sem_anotacao"], 1)
        self.assertEqual(validate_dataset(self.target), [])
        self.assertIn("CC BY 4.0", (self.target / "metadata.csv").read_text(encoding="utf-8"))

    def test_polygon_becomes_clamped_bounding_box(self) -> None:
        self._populate()
        self._add("train", "poly_jpg.rf.h4", "2 0.1 0.2 0.5 0.2 0.3 1.2\n", 40)
        import_dataset(self.source, self.target)

        labels = [path.read_text() for path in (self.target / "labels" / "train").glob("*poly*.txt")]
        class_id, x_center, y_center, width, height = labels[0].split()
        self.assertEqual(class_id, str(CLASS_NAMES.index("pedra")))
        self.assertEqual(
            (float(x_center), float(y_center), float(width), float(height)),
            (0.3, 0.6, 0.4, 0.8),
        )
        self.assertEqual(validate_dataset(self.target), [])

    def test_augmented_copies_share_asset_id(self) -> None:
        self._populate()
        self._add("train", "a_jpg.rf.h9", "0 0.5 0.5 0.4 0.4\n", 50)
        counts = import_dataset(self.source, self.target)

        metadata = (self.target / "metadata.csv").read_text(encoding="utf-8")
        self.assertEqual(metadata.count("roboflow:a_jpg"), 2)
        self.assertEqual(counts["imagens_train"], 2)
        self.assertEqual(counts["originais_train"], 1)

    def test_imports_directly_from_zip(self) -> None:
        self._populate()
        archive_path = self.root / "export.zip"
        with zipfile.ZipFile(archive_path, "w") as archive:
            for path in self.source.rglob("*"):
                if path.is_file():
                    archive.write(path, path.relative_to(self.source).as_posix())

        counts = import_dataset(archive_path, self.target)
        self.assertEqual(sum(counts[f"imagens_{split}"] for split in SPLITS), 3)
        self.assertEqual(validate_dataset(self.target), [])

    def test_unsafe_zip_names_are_never_written_to_disk(self) -> None:
        self._populate()
        archive_path = self.root / "evil.zip"
        with zipfile.ZipFile(archive_path, "w") as archive:
            for path in self.source.rglob("*"):
                if path.is_file():
                    archive.write(path, path.relative_to(self.source).as_posix())
            archive.writestr("../escape.txt", "x")

        import_dataset(archive_path, self.target)
        self.assertFalse((self.root / "escape.txt").exists())
        self.assertEqual(validate_dataset(self.target), [])

    def test_large_images_are_downscaled_and_labels_stay_normalized(self) -> None:
        self._populate()
        large = self.source / "train" / "images" / "big_jpg.rf.h5.jpg"
        Image.new("RGB", (3000, 2000), color=(80, 10, 20)).save(large)
        (self.source / "train" / "labels" / "big_jpg.rf.h5.txt").write_text(
            "0 0.5 0.5 0.4 0.4\n", encoding="utf-8"
        )

        counts = import_dataset(self.source, self.target, max_side=1280)

        with Image.open(next((self.target / "images" / "train").glob("*big*"))) as image:
            self.assertEqual(image.size, (1280, 853))
        self.assertEqual(counts["imagens_reduzidas"], 1)
        self.assertEqual(validate_dataset(self.target), [])

    def test_max_side_zero_keeps_original_size(self) -> None:
        self._populate()
        large = self.source / "train" / "images" / "big_jpg.rf.h5.jpg"
        Image.new("RGB", (3000, 2000), color=(80, 10, 20)).save(large)
        (self.source / "train" / "labels" / "big_jpg.rf.h5.txt").write_text("", encoding="utf-8")

        import_dataset(self.source, self.target, max_side=0)

        with Image.open(next((self.target / "images" / "train").glob("*big*"))) as image:
            self.assertEqual(image.size, (3000, 2000))

    def test_creates_missing_target_folders_for_fresh_clones(self) -> None:
        self._populate()
        fresh = self.root / "fresh_dataset"
        fresh.mkdir()
        import_dataset(self.source, fresh)
        self.assertEqual(validate_dataset(fresh), [])

    def test_unmapped_class_aborts_without_copying(self) -> None:
        self._populate()
        (self.source / "data.yaml").write_text("names: [drainage, fake manhole]\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "sem mapeamento"):
            import_dataset(self.source, self.target)
        self.assertFalse(
            any(path for split in SPLITS for path in (self.target / "images" / split).iterdir())
        )
        self.assertFalse((self.target / "metadata.csv").exists())

    def test_missing_split_aborts_and_nonempty_target_is_protected(self) -> None:
        self._populate()
        (self.target / "images" / "train" / "keep.png").write_bytes(b"keep")
        with self.assertRaises(FileExistsError):
            import_dataset(self.source, self.target)
        (self.target / "images" / "train" / "keep.png").unlink()

        for child in (self.source / "test").rglob("*"):
            if child.is_file():
                child.unlink()
        for directory in ((self.source / "test" / "images"), (self.source / "test" / "labels"), (self.source / "test")):
            directory.rmdir()
        with self.assertRaisesRegex(ValueError, "Split 'test' ausente"):
            import_dataset(self.source, self.target)


if __name__ == "__main__":
    unittest.main()
