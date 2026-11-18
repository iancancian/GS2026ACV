"""Validacao local de imagens, anotacoes YOLO e separacao por local."""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import sys
from collections.abc import Sequence
from pathlib import Path

from PIL import Image

from boca_de_lobo.classes import CLASS_NAMES
from boca_de_lobo.paths import PROJECT_ROOT
SPLITS = ("train", "val", "test")
IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png"})
METADATA_COLUMNS = frozenset({"image_path", "site_group", "asset_id"})
# Datasets publicos nao informam o local; asset_id e hash continuam separando os splits.
UNKNOWN_SITE_GROUP = "desconhecido"
BOX_TOLERANCE = 1e-6  # absorve arredondamento de 6 casas das exportacoes YOLO


def _is_supported_image(path: Path) -> bool:
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            image.load()
    except (OSError, SyntaxError, Image.DecompressionBombError):
        return False

    return True


def _validate_label(path: Path, errors: list[str]) -> None:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        errors.append(f"{path}: nao foi possivel ler anotacao ({error}).")
        return

    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        fields = line.split()
        location = f"{path}:{line_number}"
        if len(fields) != 5:
            errors.append(f"{location}: esperado class_id x_center y_center width height.")
            continue

        try:
            class_id = int(fields[0])
            coordinates = tuple(float(value) for value in fields[1:])
        except ValueError:
            errors.append(f"{location}: classe e coordenadas devem ser numericas.")
            continue

        if not 0 <= class_id < len(CLASS_NAMES):
            errors.append(f"{location}: class_id {class_id} fora de 0..{len(CLASS_NAMES) - 1}.")
        if not all(math.isfinite(value) for value in coordinates):
            errors.append(f"{location}: coordenadas devem ser finitas.")
            continue

        x_center, y_center, width, height = coordinates
        if not all(0 <= value <= 1 for value in coordinates):
            errors.append(f"{location}: coordenadas YOLO devem estar normalizadas entre 0 e 1.")
        if width <= 0 or height <= 0:
            errors.append(f"{location}: largura e altura devem ser maiores que zero.")
        if x_center - width / 2 < -BOX_TOLERANCE or x_center + width / 2 > 1 + BOX_TOLERANCE:
            errors.append(f"{location}: caixa ultrapassa os limites horizontais da imagem.")
        if y_center - height / 2 < -BOX_TOLERANCE or y_center + height / 2 > 1 + BOX_TOLERANCE:
            errors.append(f"{location}: caixa ultrapassa os limites verticais da imagem.")


def _validate_metadata(
    root: Path,
    image_splits: dict[str, str],
    errors: list[str],
) -> None:
    metadata_path = root / "metadata.csv"
    if not metadata_path.is_file():
        errors.append(f"{metadata_path}: obrigatorio quando ha imagens no dataset.")
        return

    seen_paths: set[str] = set()
    group_splits: dict[tuple[str, str], str] = {}
    try:
        with metadata_path.open(encoding="utf-8-sig", newline="") as metadata_file:
            reader = csv.DictReader(metadata_file)
            if reader.fieldnames is None or not METADATA_COLUMNS.issubset(reader.fieldnames):
                errors.append(
                    f"{metadata_path}: cabecalho deve conter "
                    + ", ".join(sorted(METADATA_COLUMNS))
                    + "."
                )
                return

            for row_number, row in enumerate(reader, start=2):
                image_key = (row.get("image_path") or "").replace("\\", "/").strip()
                site_group = (row.get("site_group") or "").strip()
                asset_id = (row.get("asset_id") or "").strip()
                location = f"{metadata_path}:{row_number}"
                if not image_key or not site_group or not asset_id:
                    errors.append(f"{location}: image_path, site_group e asset_id sao obrigatorios.")
                    continue
                if image_key in seen_paths:
                    errors.append(f"{location}: image_path duplicado: {image_key}.")
                    continue
                seen_paths.add(image_key)
                if image_key not in image_splits:
                    errors.append(f"{location}: imagem nao encontrada no split: {image_key}.")
                    continue

                split = image_splits[image_key]
                for key_type, key in (("site_group", site_group), ("asset_id", asset_id)):
                    if key_type == "site_group" and key == UNKNOWN_SITE_GROUP:
                        continue
                    previous_split = group_splits.setdefault((key_type, key), split)
                    if previous_split != split:
                        errors.append(
                            f"{location}: {key_type} {key} aparece em {previous_split} e {split}."
                        )

    except OSError as error:
        errors.append(f"{metadata_path}: nao foi possivel ler metadata ({error}).")
        return

    for image_key in sorted(image_splits.keys() - seen_paths):
        errors.append(f"{metadata_path}: falta metadado para {image_key}.")


def validate_dataset(root: Path) -> list[str]:
    """Retorna erros; uma estrutura vazia e valida antes da coleta."""
    root = root.resolve()
    errors: list[str] = []
    image_splits: dict[str, str] = {}
    image_hashes: dict[str, tuple[str, str]] = {}

    for split in SPLITS:
        image_dir = root / "images" / split
        label_dir = root / "labels" / split
        if not image_dir.is_dir():
            errors.append(f"Diretorio ausente: {image_dir}.")
            continue
        if not label_dir.is_dir():
            errors.append(f"Diretorio ausente: {label_dir}.")
            continue

        image_files = sorted(
            path
            for path in image_dir.iterdir()
            if path.is_file() and path.name != ".gitkeep"
        )
        images = [path for path in image_files if path.suffix.lower() in IMAGE_EXTENSIONS]
        for unsupported_path in image_files:
            if unsupported_path.suffix.lower() not in IMAGE_EXTENSIONS:
                errors.append(
                    f"{unsupported_path}: formato nao suportado; use JPG, JPEG ou PNG."
                )

        labels = {path.stem: path for path in label_dir.glob("*.txt") if path.is_file()}
        image_stems: set[str] = set()

        for image_path in images:
            if image_path.stem in image_stems:
                errors.append(f"{image_path}: mais de uma imagem com o mesmo nome-base no split.")
                continue
            image_stems.add(image_path.stem)
            image_key = image_path.relative_to(root).as_posix()
            image_splits[image_key] = split
            if not _is_supported_image(image_path):
                errors.append(f"{image_path}: arquivo vazio, corrompido ou formato nao suportado.")
            else:
                digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
                previous_split, previous_key = image_hashes.setdefault(digest, (split, image_key))
                if previous_split != split:
                    errors.append(f"{image_key}: conteudo identico a {previous_key} em outro split.")
            label_path = labels.get(image_path.stem)
            if label_path is None:
                errors.append(f"{image_path}: anotacao correspondente ausente.")
            else:
                _validate_label(label_path, errors)

        for label_stem in sorted(labels.keys() - image_stems):
            errors.append(f"{labels[label_stem]}: nao ha imagem correspondente.")

    if image_splits:
        populated_splits = set(image_splits.values())
        for split in SPLITS:
            if split not in populated_splits:
                errors.append(f"Split {split} vazio; cada split deve conter imagens antes do treino.")
        _validate_metadata(root, image_splits, errors)

    return errors


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Valida o dataset YOLO Boca de Lobo.")
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT / "dataset")
    args = parser.parse_args(argv)

    errors = validate_dataset(args.root)
    if errors:
        print(f"Dataset invalido: {len(errors)} problema(s).", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    image_count = sum(
        1
        for split in SPLITS
        for path in (args.root / "images" / split).iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if image_count == 0:
        print("Estrutura valida; ainda nao ha imagens para treinar.")
    else:
        print(f"Dataset valido: {image_count} imagem(ns), classes: {', '.join(CLASS_NAMES)}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
