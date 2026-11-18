"""Importa uma exportacao YOLO do Roboflow (ZIP ou pasta) para a pasta dataset/.

O ZIP e lido em memoria, sem extracao: nenhum nome de arquivo do ZIP vira
caminho em disco, e nao ha copia temporaria de varios GB.
"""

from __future__ import annotations

import argparse
import csv
import io
import logging
import re
import sys
import zipfile
from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Any

import yaml
from PIL import Image

from boca_de_lobo.classes import CLASS_NAMES
from boca_de_lobo.dataset import IMAGE_EXTENSIONS, SPLITS, UNKNOWN_SITE_GROUP
from boca_de_lobo.paths import PROJECT_ROOT

LOGGER = logging.getLogger("boca_de_lobo.import_roboflow")

SOURCE_TO_TARGET = {"drainage": 0, "trash": 1, "leaf": 2, "rock": 3, "silt": 4}
SPLIT_ALIASES = {"train": ("train",), "val": ("valid", "val"), "test": ("test",)}
SOURCE_INFO = {
    "source_dataset": "wqwdas/drainage-dataset",
    "source_license": "CC BY 4.0",
    "source_url": "https://universe.roboflow.com/wqwdas/drainage-dataset",
    "source_author": "wqwdas",
}
METADATA_FIELDS = (
    "image_path",
    "site_group",
    "asset_id",
    "captured_at",
    "lighting",
    "weather",
    "annotator_code",
    *SOURCE_INFO,
)
BOX_EPSILON = 1e-6
DEFAULT_MAX_SIDE = 1280
JPEG_QUALITY = 92

FileReaders = dict[str, Callable[[], bytes]]


def _normalize(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.casefold())


def _safe_stem(stem: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", stem)


def _asset_id(stem: str) -> str:
    """Copias aumentadas do Roboflow compartilham o nome antes de '.rf.'."""
    return stem.split(".rf.")[0]


@contextmanager
def _open_export(source: Path) -> Iterator[FileReaders]:
    """Mapeia caminho relativo (posix) para uma funcao que le os bytes do arquivo."""
    if source.is_file() and source.suffix.lower() == ".zip":
        with zipfile.ZipFile(source) as archive:
            yield {
                info.filename.replace("\\", "/"): (lambda entry=info: archive.read(entry))
                for info in archive.infolist()
                if not info.is_dir()
            }
    elif source.is_dir():
        yield {
            path.relative_to(source).as_posix(): path.read_bytes
            for path in source.rglob("*")
            if path.is_file()
        }
    else:
        raise ValueError(f"Origem deve ser um .zip ou uma pasta existente: {source}")


def _find_prefix(files: FileReaders) -> str:
    candidates = sorted(
        (name for name in files if PurePosixPath(name).name == "data.yaml"),
        key=lambda name: name.count("/"),
    )
    if not candidates:
        raise ValueError("data.yaml nao encontrado na origem. Exporte no formato YOLOv8.")
    return candidates[0][: -len("data.yaml")]


def _read_class_map(files: FileReaders, prefix: str) -> dict[int, int]:
    config: Any = yaml.safe_load(files[f"{prefix}data.yaml"]().decode("utf-8"))
    names = config.get("names") if isinstance(config, dict) else None
    if isinstance(names, list):
        names = dict(enumerate(names))
    if not isinstance(names, dict) or not names:
        raise ValueError("data.yaml: campo 'names' ausente ou invalido.")

    class_map: dict[int, int] = {}
    unmapped: list[str] = []
    for class_id, name in names.items():
        target = SOURCE_TO_TARGET.get(_normalize(str(name)))
        if target is None:
            unmapped.append(str(name))
        else:
            class_map[int(class_id)] = target
    if unmapped:
        raise ValueError(
            "Classes sem mapeamento: " + ", ".join(unmapped) + ". Revise a taxonomia antes de importar."
        )
    return class_map


def _split_images(files: FileReaders, prefix: str, split: str) -> tuple[str, list[str]]:
    for alias in SPLIT_ALIASES[split]:
        base = f"{prefix}{alias}/images/"
        images = sorted(
            name
            for name in files
            if name.startswith(base)
            and "/" not in name[len(base) :]
            and PurePosixPath(name).suffix.lower() in IMAGE_EXTENSIONS
        )
        if images:
            return alias, images
    raise ValueError(
        f"Split '{split}' ausente ou sem imagens. Gere a versao no Roboflow com train/valid/test (70/20/10)."
    )


def _convert_label(text: str, location: str, class_map: dict[int, int], counts: Counter[str]) -> str:
    """Converte para YOLO detection; poligonos viram caixas envolventes."""
    lines: list[str] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        fields = line.split()
        if not fields:
            continue
        where = f"{location}:{line_number}"
        try:
            source_id = int(fields[0])
            values = [float(value) for value in fields[1:]]
        except ValueError as error:
            raise ValueError(f"{where}: valores nao numericos.") from error
        if source_id not in class_map:
            raise ValueError(f"{where}: class_id {source_id} nao existe em data.yaml.")

        if len(values) == 4:
            x, y, width, height = values
            x1, y1, x2, y2 = x - width / 2, y - height / 2, x + width / 2, y + height / 2
        elif len(values) >= 6 and len(values) % 2 == 0:
            xs, ys = values[0::2], values[1::2]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        else:
            raise ValueError(f"{where}: formato de anotacao nao reconhecido.")

        x1, y1, x2, y2 = (min(max(value, 0.0), 1.0) for value in (x1, y1, x2, y2))
        if x2 - x1 <= BOX_EPSILON or y2 - y1 <= BOX_EPSILON:
            counts["caixas_descartadas"] += 1
            continue

        target = class_map[source_id]
        counts[CLASS_NAMES[target]] += 1
        lines.append(
            f"{target} {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} {x2 - x1:.6f} {y2 - y1:.6f}"
        )
    return "\n".join(lines) + ("\n" if lines else "")


def _write_image(data: bytes, target: Path, max_side: int, counts: Counter[str]) -> None:
    """Grava a imagem; reduz o maior lado a max_side (rotulos normalizados nao mudam)."""
    with Image.open(io.BytesIO(data)) as image:
        image.load()
        if max_side and max(image.size) > max_side:
            scale = max_side / max(image.size)
            size = (round(image.width * scale), round(image.height * scale))
            resized = image.convert("RGB").resize(size, Image.Resampling.LANCZOS)
            if target.suffix.lower() == ".png":
                resized.save(target, format="PNG")
            else:
                resized.save(target, format="JPEG", quality=JPEG_QUALITY)
            counts["imagens_reduzidas"] += 1
            return
    target.write_bytes(data)


def _check_target_is_empty(target_root: Path) -> None:
    if (target_root / "metadata.csv").exists():
        raise FileExistsError(f"Ja existe {target_root / 'metadata.csv'}; nada foi alterado.")
    for split in SPLITS:
        for kind in ("images", "labels"):
            directory = target_root / kind / split
            directory.mkdir(parents=True, exist_ok=True)  # pastas vazias nao sao versionadas pelo Git
            if any(path.name != ".gitkeep" for path in directory.iterdir()):
                raise FileExistsError(f"Destino nao esta vazio: {directory}")


def import_dataset(
    source: Path, target_root: Path, max_side: int = DEFAULT_MAX_SIDE
) -> Counter[str]:
    """Importa tudo ou nada; recusa destino que ja contenha dados."""
    if max_side < 0:
        raise ValueError("max_side nao pode ser negativo.")
    target_root = target_root.resolve()
    _check_target_is_empty(target_root)

    counts: Counter[str] = Counter()
    with _open_export(source) as files:
        prefix = _find_prefix(files)
        class_map = _read_class_map(files, prefix)

        plan: list[tuple[str, str, str]] = []
        for split in SPLITS:
            alias, images = _split_images(files, prefix, split)
            for name in images:
                label_name = f"{prefix}{alias}/labels/{PurePosixPath(name).stem}.txt"
                if label_name in files:
                    label = _convert_label(
                        files[label_name]().decode("utf-8"), label_name, class_map, counts
                    )
                else:
                    label = ""
                    counts["imagens_sem_anotacao"] += 1
                counts[f"imagens_{split}"] += 1
                plan.append((split, name, label))
            counts[f"originais_{split}"] = len(
                {_asset_id(PurePosixPath(name).stem) for name in images}
            )

        rows: list[dict[str, str]] = []
        created: list[Path] = []
        split_indexes: Counter[str] = Counter()
        try:
            for split, name, label in plan:
                split_indexes[split] += 1
                posix = PurePosixPath(name)
                stem = f"{split}_{split_indexes[split]:04d}_{_safe_stem(posix.stem)}"
                image_target = target_root / "images" / split / f"{stem}{posix.suffix.lower()}"
                label_target = target_root / "labels" / split / f"{stem}.txt"
                _write_image(files[name](), image_target, max_side, counts)
                created.append(image_target)
                label_target.write_text(label, encoding="utf-8")
                created.append(label_target)
                rows.append(
                    {
                        "image_path": image_target.relative_to(target_root).as_posix(),
                        "site_group": UNKNOWN_SITE_GROUP,
                        "asset_id": f"roboflow:{_asset_id(posix.stem)}",
                        "captured_at": "",
                        "lighting": "",
                        "weather": "",
                        "annotator_code": "roboflow",
                        **SOURCE_INFO,
                    }
                )

            metadata_path = target_root / "metadata.csv"
            with metadata_path.open("w", encoding="utf-8", newline="") as metadata_file:
                writer = csv.DictWriter(metadata_file, fieldnames=METADATA_FIELDS)
                writer.writeheader()
                writer.writerows(rows)
        except (OSError, ValueError):
            for path in created:
                path.unlink(missing_ok=True)
            raise
    return counts


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Importa o dataset Roboflow (formato YOLOv8).")
    parser.add_argument("--source", type=Path, required=True, help="ZIP baixado ou pasta extraida.")
    parser.add_argument("--target", type=Path, default=PROJECT_ROOT / "dataset")
    parser.add_argument(
        "--max-side",
        type=int,
        default=DEFAULT_MAX_SIDE,
        help="Maior lado em pixels apos a importacao; 0 mantem o tamanho original.",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    try:
        counts = import_dataset(args.source, args.target, args.max_side)
    except (OSError, ValueError, zipfile.BadZipFile, yaml.YAMLError) as error:
        LOGGER.error("Importacao abortada: %s", error)
        return 1

    for key in sorted(counts):
        LOGGER.info("%s: %d", key, counts[key])
    for split in SPLITS:
        if counts[f"imagens_{split}"] > counts[f"originais_{split}"]:
            LOGGER.warning(
                "O split %s tem copias aumentadas (%d imagens para %d originais). "
                "Prefira um fork sem augmentation e verifique se as imagens estao em escala de cinza.",
                split,
                counts[f"imagens_{split}"],
                counts[f"originais_{split}"],
            )
    LOGGER.warning(
        "Revise amostras: remova imagens com pessoas e nao publique as fotos. "
        "O dataset nao informa locais; confirme duplicatas com python -m boca_de_lobo.dataset."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
