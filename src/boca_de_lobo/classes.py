"""Taxonomia compartilhada pelas ferramentas e pelo dataset YOLO.

O modelo detecta o bueiro e os materiais que podem bloqueá-lo; o estado
(livre, parcial, entupido) é derivado em `decision.py`, não aprendido.
"""

CLASS_NAMES = (
    "bueiro",
    "lixo",
    "folha",
    "pedra",
    "sedimento",
)

DRAIN_NAME = CLASS_NAMES[0]
OBSTRUCTION_NAMES = frozenset(CLASS_NAMES[1:])
