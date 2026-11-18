"""Gera dados SIMULADOS de inspecao a partir das fotos do split test.

Justificativa: nao ha, neste momento, um inventario real de bueiros nem acesso a
fotos de campo. As fotos publicas de teste fazem o papel de inspecoes e recebem
identificadores e areas ficticios (semente fixa, reprodutivel).
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from collections.abc import Sequence
from pathlib import Path

from boca_de_lobo.classes import CLASS_NAMES, DRAIN_NAME
from boca_de_lobo.dataset import IMAGE_EXTENSIONS
from boca_de_lobo.paths import PROJECT_ROOT

FIELDS = ("asset_id", "image_file", "area_id")


def _has_drain(label_path: Path) -> bool:
    drain_id = str(CLASS_NAMES.index(DRAIN_NAME))
    try:
        return any(line.split()[:1] == [drain_id] for line in label_path.read_text(encoding="utf-8").splitlines())
    except OSError:
        return False


def build_manifest(
    images_dir: Path,
    area_ids: Sequence[str],
    count: int,
    seed: int,
    inject_faults: bool,
    labels_dir: Path | None = None,
) -> list[dict[str, str]]:
    """Escolhe fotos de teste; com labels_dir, so as que tem bueiro (uma inspecao fotografa um bueiro)."""
    images = sorted(
        path for path in images_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if labels_dir is not None:
        images = [path for path in images if _has_drain(labels_dir / f"{path.stem}.txt")]
    names = [path.name for path in images]
    if not names or not area_ids:
        raise ValueError("E necessario ter imagens elegiveis e ao menos uma area.")
    chosen = random.Random(seed).sample(names, min(count, len(names)))
    rows = [
        {"asset_id": f"BUEIRO-{index:03d}", "image_file": name, "area_id": area_ids[index % len(area_ids)]}
        for index, name in enumerate(chosen, start=1)
    ]
    if inject_faults:
        rows += [
            {"asset_id": "BUEIRO-900", "image_file": "nao_existe.jpg", "area_id": area_ids[0]},
            {"asset_id": "BUEIRO-901", "image_file": chosen[0], "area_id": "AREA-INEXISTENTE"},
            {"asset_id": "BUEIRO-902", "image_file": "../fora_da_pasta.jpg", "area_id": area_ids[0]},
            dict(rows[0]),
        ]
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Gera inspecoes simuladas para a automacao.")
    parser.add_argument("--images-dir", type=Path, default=PROJECT_ROOT / "dataset" / "images" / "test")
    parser.add_argument(
        "--labels-dir",
        type=Path,
        default=PROJECT_ROOT / "dataset" / "labels" / "test",
        help="Rotulos usados so para escolher fotos com bueiro; use --any-image para ignorar.",
    )
    parser.add_argument("--any-image", action="store_true", help="Nao filtra por fotos com bueiro.")
    parser.add_argument("--areas", type=Path, default=PROJECT_ROOT / "data" / "areas_risco.csv")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data" / "inspecoes_exemplo.csv")
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--inject-faults", action="store_true", help="Inclui registros defeituosos para demonstrar o tratamento de falhas.")
    args = parser.parse_args(argv)

    with args.areas.open(encoding="utf-8-sig", newline="") as areas_file:
        area_ids = [row["area_id"] for row in csv.DictReader(areas_file) if row.get("area_id")]
    try:
        rows = build_manifest(
            args.images_dir,
            area_ids,
            args.count,
            args.seed,
            args.inject_faults,
            None if args.any_image else args.labels_dir,
        )
    except (OSError, ValueError) as error:
        print(f"Falha ao gerar manifesto: {error}", file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} registro(s) em {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
