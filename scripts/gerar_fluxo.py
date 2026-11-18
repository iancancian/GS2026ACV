"""Gera o fluxo BPMN (docs/fluxo_bpmn.bpmn) e sua imagem (assets/fluxo_bpmn.png).

Uso: python scripts/gerar_fluxo.py   (requer Pillow)
As coordenadas ficam num so lugar, entao o XML e o PNG nunca divergem.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from xml.sax.saxutils import quoteattr

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
TASK_W, TASK_H, GATE, EVENT = 150, 72, 48, 36
CANVAS = (1300, 680)
SCALE = 2  # supersampling para suavizar as bordas


@dataclass(frozen=True)
class Node:
    id: str
    kind: str  # start, end, end_error, service, user, gateway
    label: str
    cx: int
    cy: int

    @property
    def size(self) -> tuple[int, int]:
        if self.kind in ("service", "user"):
            return TASK_W, TASK_H
        return (GATE, GATE) if self.kind == "gateway" else (EVENT, EVENT)

    @property
    def box(self) -> tuple[int, int, int, int]:
        width, height = self.size
        return self.cx - width // 2, self.cy - height // 2, width, height

    def anchor(self, side: str) -> tuple[int, int]:
        x, y, width, height = self.box
        return {
            "left": (x, self.cy),
            "right": (x + width, self.cy),
            "top": (self.cx, y),
            "bottom": (self.cx, y + height),
        }[side]


NODES = [
    Node("inicio", "start", "Lote agendado antes do temporal", 64, 120),
    Node("ler", "service", "Ler áreas de risco e lista de inspeções (CSV)", 190, 120),
    Node("g_entradas", "gateway", "Entradas válidas?", 360, 120),
    Node("abortar", "end_error", "Abortar e registrar erro (código 2)", 360, 30),
    Node("previsao", "service", "Consultar previsão de chuva por área (API, 3 tentativas)", 520, 120),
    Node("g_previsao", "gateway", "Previsão disponível?", 700, 120),
    Node("indisponivel", "service", "Marcar previsão indisponível (nunca assumir sem chuva)", 860, 120),
    Node("validar", "service", "Validar registro, área e imagem", 190, 360),
    Node("g_valido", "gateway", "Registro válido?", 360, 360),
    Node("detectar", "service", "Detectar bueiro e obstruções (YOLO)", 520, 360),
    Node("cobertura", "service", "Calcular cobertura, estado e prioridade", 710, 360),
    Node("registrar", "service", "Registrar item e log de auditoria", 900, 360),
    Node("g_mais", "gateway", "Há mais inspeções?", 1070, 360),
    Node("rejeitar", "service", "Registrar rejeição no log e seguir", 360, 480),
    Node("gerar", "service", "Gerar fila de ações, tarefas de limpeza e avisos (JSON/CSV)", 1160, 600),
    Node("aprovar", "user", "Equipe revisa e aprova ações e avisos (humano)", 960, 600),
    Node("fim", "end", "MVP consome os arquivos", 800, 600),
]
BY_ID = {node.id: node for node in NODES}

# (id, origem, lado, destino, lado, pontos intermediarios, rotulo)
FLOWS = [
    ("f01", "inicio", "right", "ler", "left", [], ""),
    ("f02", "ler", "right", "g_entradas", "left", [], ""),
    ("f03", "g_entradas", "right", "previsao", "left", [], "Sim"),
    ("f04", "g_entradas", "top", "abortar", "bottom", [], "Não"),
    ("f05", "previsao", "right", "g_previsao", "left", [], ""),
    ("f06", "g_previsao", "bottom", "validar", "left", [(700, 240), (80, 240), (80, 360)], "Sim"),
    ("f07", "g_previsao", "right", "indisponivel", "left", [], "Não"),
    ("f08", "indisponivel", "bottom", "validar", "left", [(860, 240), (80, 240), (80, 360)], ""),
    ("f09", "validar", "right", "g_valido", "left", [], ""),
    ("f10", "g_valido", "right", "detectar", "left", [], "Sim"),
    ("f11", "g_valido", "bottom", "rejeitar", "top", [], "Não"),
    ("f12", "detectar", "right", "cobertura", "left", [], ""),
    ("f13", "cobertura", "right", "registrar", "left", [], ""),
    ("f14", "registrar", "right", "g_mais", "left", [], ""),
    ("f15", "rejeitar", "bottom", "g_mais", "bottom", [(360, 540), (1070, 540)], ""),
    ("f16", "g_mais", "top", "validar", "top", [(1070, 290), (190, 290)], "Sim"),
    ("f17", "g_mais", "right", "gerar", "top", [(1160, 360)], "Não"),
    ("f18", "gerar", "left", "aprovar", "right", [], ""),
    ("f19", "aprovar", "left", "fim", "right", [], ""),
]
TAGS = {
    "start": "startEvent",
    "end": "endEvent",
    "end_error": "endEvent",
    "service": "serviceTask",
    "user": "userTask",
    "gateway": "exclusiveGateway",
}


def waypoints(flow: tuple) -> list[tuple[int, int]]:
    _, source, source_side, target, target_side, via, _ = flow
    return [BY_ID[source].anchor(source_side), *via, BY_ID[target].anchor(target_side)]


def build_bpmn() -> str:
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<bpmn:definitions xmlns:bpmn="http://www.omg.org/spec/BPMN/20100524/MODEL" '
        'xmlns:bpmndi="http://www.omg.org/spec/BPMN/20100524/DI" '
        'xmlns:dc="http://www.omg.org/spec/DD/20100524/DC" '
        'xmlns:di="http://www.omg.org/spec/DD/20100524/DI" id="Definitions_BocaDeLobo" '
        'targetNamespace="http://bpmn.io/schema/bpmn">',
        '  <bpmn:process id="Process_BocaDeLobo" name="Inspecao preventiva de bocas de lobo" isExecutable="false">',
    ]
    for node in NODES:
        body = "".join(f"<bpmn:incoming>{flow[0]}</bpmn:incoming>" for flow in FLOWS if flow[3] == node.id)
        body += "".join(f"<bpmn:outgoing>{flow[0]}</bpmn:outgoing>" for flow in FLOWS if flow[1] == node.id)
        if node.kind == "end_error":
            body += "<bpmn:errorEventDefinition />"
        tag = TAGS[node.kind]
        lines.append(f"    <bpmn:{tag} id={quoteattr(node.id)} name={quoteattr(node.label)}>{body}</bpmn:{tag}>")
    for flow in FLOWS:
        name = f" name={quoteattr(flow[6])}" if flow[6] else ""
        lines.append(f'    <bpmn:sequenceFlow id="{flow[0]}"{name} sourceRef="{flow[1]}" targetRef="{flow[3]}" />')
    lines += [
        "  </bpmn:process>",
        '  <bpmndi:BPMNDiagram id="Diagram_1">',
        '    <bpmndi:BPMNPlane id="Plane_1" bpmnElement="Process_BocaDeLobo">',
    ]
    for node in NODES:
        x, y, width, height = node.box
        marker = ' isMarkerVisible="true"' if node.kind == "gateway" else ""
        lines += [
            f'      <bpmndi:BPMNShape id="{node.id}_di" bpmnElement="{node.id}"{marker}>',
            f'        <dc:Bounds x="{x}" y="{y}" width="{width}" height="{height}" />',
            "      </bpmndi:BPMNShape>",
        ]
    for flow in FLOWS:
        lines.append(f'      <bpmndi:BPMNEdge id="{flow[0]}_di" bpmnElement="{flow[0]}">')
        lines += [f'        <di:waypoint x="{x}" y="{y}" />' for x, y in waypoints(flow)]
        lines.append("      </bpmndi:BPMNEdge>")
    lines += ["    </bpmndi:BPMNPlane>", "  </bpmndi:BPMNDiagram>", "</bpmn:definitions>", ""]
    return "\n".join(lines)


class Canvas:
    """Desenha em coordenadas logicas; a escala fica encapsulada aqui."""

    def __init__(self) -> None:
        self.image = Image.new("RGB", (CANVAS[0] * SCALE, CANVAS[1] * SCALE), "white")
        self.draw = ImageDraw.Draw(self.image)
        self.body = self._font(14)
        self.small = self._font(13)
        self.bold = self._font(13, bold=True)

    @staticmethod
    def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
        for name in ("arialbd.ttf" if bold else "arial.ttf", "DejaVuSans.ttf"):
            try:
                return ImageFont.truetype(name, size * SCALE)
            except OSError:
                continue
        return ImageFont.load_default()

    @staticmethod
    def _scaled(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
        return [(x * SCALE, y * SCALE) for x, y in points]

    def wrap(self, text: str, font: ImageFont.ImageFont, width: int) -> list[str]:
        lines: list[str] = []
        current = ""
        for word in text.split():
            trial = f"{current} {word}".strip()
            if not current or self.draw.textlength(trial, font=font) <= width * SCALE:
                current = trial
            else:
                lines.append(current)
                current = word
        return [*lines, current]

    def text(
        self, x: float, y: float, content: str, font: ImageFont.ImageFont, fill: str, align: str = "left"
    ) -> None:
        width = self.draw.textlength(content, font=font)
        offset = {"left": 0, "center": width / 2, "right": width}[align]
        self.draw.text((x * SCALE - offset, y * SCALE), content, fill=fill, font=font)

    def arrow(self, points: list[tuple[int, int]]) -> None:
        color = "#37474f"
        self.draw.line(self._scaled(points), fill=color, width=2 * SCALE, joint="curve")
        (x1, y1), (x2, y2) = points[-2], points[-1]
        length = max(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5, 1)
        ux, uy = (x2 - x1) / length, (y2 - y1) / length
        head = [
            (x2, y2),
            (x2 - 12 * ux + 6 * uy, y2 - 12 * uy - 6 * ux),
            (x2 - 12 * ux - 6 * uy, y2 - 12 * uy + 6 * ux),
        ]
        self.draw.polygon(self._scaled(head), fill=color)

    def edge_label(self, points: list[tuple[int, int]], label: str) -> None:
        (x1, y1), (x2, y2) = points[0], points[1]
        if y1 == y2:
            position = (x1 + 8, y1 - 20)
        else:
            position = (x1 + 8, y1 + 4 if y2 > y1 else y1 - 22)
        self.text(*position, label, self.bold, "#37474f")

    def task(self, node: Node) -> None:
        x, y, width, height = node.box
        fill = "#fff8e1" if node.kind == "user" else "#e3f2fd"
        self.draw.rounded_rectangle(
            [x * SCALE, y * SCALE, (x + width) * SCALE, (y + height) * SCALE],
            radius=10 * SCALE, fill=fill, outline="#1565c0", width=2 * SCALE,
        )
        lines = self.wrap(node.label, self.body, width - 12)
        top = node.cy - len(lines) * 17 / 2
        for index, line in enumerate(lines):
            self.text(node.cx, top + index * 17, line, self.body, "#0d1b2a", align="center")

    def gateway(self, node: Node) -> None:
        x, y, width, height = node.box
        diamond = [(node.cx, y), (x + width, node.cy), (node.cx, y + height), (x, node.cy)]
        self.draw.polygon(self._scaled(diamond), fill="#fffde7", outline="#f9a825", width=2 * SCALE)
        for start, end in (((-9, -9), (9, 9)), ((-9, 9), (9, -9))):
            self.draw.line(
                self._scaled([(node.cx + start[0], node.cy + start[1]), (node.cx + end[0], node.cy + end[1])]),
                fill="#f9a825", width=3 * SCALE,
            )
        lines = self.wrap(node.label, self.small, 80)
        for index, line in enumerate(lines):
            self.text(node.cx - 6, y - 18 - 16 * (len(lines) - 1 - index), line, self.small, "#5d4037", align="right")

    def event(self, node: Node) -> None:
        x, y, width, height = node.box
        color = "#2e7d32" if node.kind == "start" else "#c62828"
        self.draw.ellipse(
            [x * SCALE, y * SCALE, (x + width) * SCALE, (y + height) * SCALE],
            fill="white", outline=color, width=(2 if node.kind == "start" else 5) * SCALE,
        )
        lines = self.wrap(node.label, self.small, 90 if node.kind == "start" else 120)
        for index, line in enumerate(lines):
            if node.kind == "end_error":
                self.text(x + width + 10, node.cy - 8 * len(lines) + 16 * index, line, self.small, "#263238")
            else:
                self.text(node.cx, y + height + 6 + 16 * index, line, self.small, "#263238", align="center")


def render(path: Path) -> None:
    canvas = Canvas()
    for flow in FLOWS:
        points = waypoints(flow)
        canvas.arrow(points)
        if flow[6]:
            canvas.edge_label(points, flow[6])
    for node in NODES:
        if node.kind in ("service", "user"):
            canvas.task(node)
        elif node.kind == "gateway":
            canvas.gateway(node)
        else:
            canvas.event(node)
    canvas.text(20, 650, "Fluxo BPMN: automação de inspeção preventiva de bocas de lobo (Boca de Lobo, CitizenTech)", canvas.small, "#455a64")
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.image.resize(CANVAS, Image.Resampling.LANCZOS).save(path)


def main() -> None:
    xml = build_bpmn()
    ET.fromstring(xml)  # falha cedo se o XML nao for bem formado
    bpmn_path = ROOT / "docs" / "fluxo_bpmn.bpmn"
    png_path = ROOT / "assets" / "fluxo_bpmn.png"
    bpmn_path.write_text(xml, encoding="utf-8")
    render(png_path)
    print(f"BPMN: {bpmn_path}\nPNG: {png_path}")


if __name__ == "__main__":
    main()
