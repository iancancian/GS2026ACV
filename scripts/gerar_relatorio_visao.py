"""Gera o relatorio de Applied Computer Vision. Requer reportlab e Pillow."""

from __future__ import annotations

import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]

RESUMO = (
    "Alagamentos urbanos afetam especialmente moradores de áreas baixas e periféricas, onde o acúmulo "
    "de materiais em bocas de lobo pode dificultar a drenagem. Este trabalho apresenta Boca de Lobo, "
    "um protótipo de visão computacional para apoiar a inspeção preventiva dessas estruturas. Um detector "
    "YOLO11n foi adaptado por treinamento supervisionado ao Drainage Dataset, conjunto público com "
    "anotações de bueiro, lixo, folha, pedra e sedimento. Após importação e validação, foram utilizadas "
    "674 imagens para treino, 193 para validação e 96 para teste. O treinamento empregou pesos "
    "pré-treinados, 80 épocas, imagens de 640 pixels e aumentos de dados do Ultralytics. Em imagens "
    "de teste, o modelo obteve precisão de 0,516, recall de 0,445 e mAP50 de 0,393. A classe bueiro "
    "alcançou mAP50 de 0,732; folhas e sedimento apresentaram desempenho inferior. Para cada bueiro "
    "detectado, uma regra calcula a união das caixas dos materiais sobrepostas à caixa da estrutura "
    "e produz uma triagem visual em três faixas. O sistema gera imagem anotada e relatório textual, "
    "além de dados estruturados que podem apoiar a automação do projeto. Os limiares da regra são "
    "experimentais e não foram calibrados com inspeções locais. Caixas delimitadoras não representam "
    "o volume de resíduos nem a condição interna da tubulação, e uma detecção ausente não garante "
    "que a estrutura esteja livre. O conjunto de teste é pequeno e pode conter imagens semelhantes "
    "às demais partições. Portanto, os resultados constituem uma linha de base para demonstração, "
    "exigindo revisão humana, coleta local e avaliação adicional cuidadosa e independente antes de qualquer uso operacional."
)

OBJETIVO = (
    "Desenvolver e avaliar um detector customizado de bueiros e materiais associados a possíveis "
    "obstruções, usando imagens e YOLO, para produzir uma triagem visual explicável que apoie "
    "inspeções preventivas. Demonstrar os resultados por meio de caixas, métricas e relatório, "
    "explicitando limitações e mantendo a decisão final sob responsabilidade humana em campo."
)


def build() -> None:
    metrics = json.loads((ROOT / "docs" / "resultados_teste.json").read_text(encoding="utf-8"))
    styles = getSampleStyleSheet()
    body = styles["BodyText"]
    body.leading = 14
    styles["Heading2"].keepWithNext = True
    story = [
        Paragraph("Boca de Lobo: inspeção preventiva com visão computacional", styles["Title"]),
        Paragraph("FIAP Global Solution 2026.2 · Applied Computer Vision", body),
        Spacer(1, 10),
        Paragraph("Integrantes", styles["Heading2"]),
        Paragraph(
            "Açussena Macedo Mautone — RM 552568<br/>Felipe Heilmann Marques — RM 551026<br/>"
            "Felipe Voidela Toledo — RM 98595<br/>Carlos Eduardo Caramante Ribeiro — RM 552159<br/>"
            "Ian Cancian Nachtergaele — RM 98387", body
        ),
        Paragraph(f"Resumo ({len(RESUMO.split())} palavras)", styles["Heading2"]),
        Paragraph(RESUMO, body),
        Paragraph("Introdução", styles["Heading2"]),
        Paragraph(
            "Resíduos acumulados em estruturas de drenagem podem contribuir para alagamentos. "
            "A proposta CitizenTech auxilia equipes a identificar pontos que merecem vistoria antes "
            "de chuvas intensas, sem substituir medições de campo ou a Defesa Civil. O componente "
            "processa fotografias; não há robô físico ou monitoramento contínuo implementado.", body
        ),
        Paragraph(f"Objetivo ({len(OBJETIVO.split())} palavras)", styles["Heading2"]),
        Paragraph(OBJETIVO, body),
        Paragraph("Ferramentas e fundamentação técnica", styles["Heading2"]),
        Paragraph(
            "Python e Ultralytics implementam o treinamento e a inferência do YOLO11n, detector "
            "que localiza objetos por caixas e atribui classes e confiança. Pillow trata imagens; "
            "PyYAML lê a configuração do conjunto. O Drainage Dataset, de wqwdas no Roboflow "
            "Universe, é disponibilizado sob CC BY 4.0. O modelo base foi pré-treinado no COCO. "
            "A licença do Ultralytics deve ser avaliada antes de uso comercial.", body
        ),
        Paragraph("Metodologia", styles["Heading2"]),
        Paragraph(
            "1. Importar o dataset no formato YOLO, mapear os nomes das cinco classes e reduzir "
            "imagens maiores que 1280 pixels no lado mais longo. O conjunto exportado não teve "
            "augmentation no Roboflow; o treinamento utilizou o aumento padrão do Ultralytics.<br/>"
            "2. Validar pares imagem/anotação, classes e duplicatas exatas; manter as partições "
            "de 674 imagens de treino, 193 de validação e 96 de teste.<br/>"
            "3. Treinar YOLO11n por 80 épocas, tamanho 640, lote 16, semente 42 e paciência 20, "
            "em uma GTX 1060. Selecionar os melhores pesos pela validação.<br/>"
            "4. Avaliar os pesos no conjunto de teste e registrar precisão, recall e mAP por classe.<br/>"
            "5. Processar uma foto com os pesos finais, gerar caixas e calcular, para cada bueiro, "
            "a área da união das caixas dos materiais dentro da caixa do bueiro, dividida pela "
            "área desta. Sobreposições entre materiais não são contadas duas vezes.<br/>"
            "6. Aplicar faixas experimentais: abaixo de 10% sem obstrução visível; de 10% "
            "a menos de 80% parcialmente obstruído; a partir de 80% entupido provável. Sem "
            "bueiro detectado, registrar estado indeterminado e revisão manual.", body
        ),
        Paragraph("Resultados", styles["Heading2"]),
        Paragraph(
            "No teste, o mAP50 geral foi 0,393; a precisão foi 0,516 e o recall, 0,445. "
            "A tabela apresenta métricas por classe; números não garantem desempenho em outro bairro.", body
        ),
    ]
    rows = [["Classe", "Precisão", "Recall", "mAP50", "mAP50-95"]]
    for name, result in metrics["per_class"].items():
        rows.append([name] + [f"{result[key]:.3f}" for key in ("precision", "recall", "map50", "map50_95")])
    table = Table(rows, colWidths=[3.5 * cm] * 5, repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dceaf5")),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    story.extend([Spacer(1, 8), table, Spacer(1, 10)])
    for index in (1, 2):
        path = ROOT / "assets" / f"exemplo_deteccao_{index}.jpg"
        image = Image(str(path))
        image.drawWidth = 14 * cm
        image.drawHeight = image.imageHeight * image.drawWidth / image.imageWidth
        story.append(KeepTogether([
            Paragraph(f"Exemplo {index}: imagem de teste com detecções do modelo.", body),
            image,
            Spacer(1, 8),
        ]))
    story.extend([
        Paragraph("Conclusão e limitações", styles["Heading2"]),
        Paragraph(
            "O protótipo demonstra treinamento e detecção customizados e produz triagem explicável, "
            "mas não diagnostica entupimento real. Os recalls de folha (0,244) e sedimento (0,125) "
            "são baixos; falhas podem subestimar a cobertura. Caixas não medem sujeira real nem "
            "obstrução interna. Os limiares são hipóteses não calibradas. A origem geográfica "
            "desconhecida e possíveis imagens semelhantes entre partições limitam a generalização. "
            "É necessário testar com fotos locais independentes, revisar erros, calibrar a regra "
            "e manter vistoria humana antes de qualquer decisão.", body
        ),
        Paragraph("Referências e reprodução", styles["Heading2"]),
        Paragraph(
            "Repositório GitHub: https://github.com/iancancian/GS2026ACV.<br/>"
            "Dataset: https://universe.roboflow.com/wqwdas/drainage-dataset (CC BY 4.0). "
            "Ultralytics: https://docs.ultralytics.com/. "
            "Comandos, parâmetros e limitações estão no README e em docs/RESULTADOS.md do projeto. "
            "As imagens ilustrativas são amostras do conjunto de teste.", body
        ),
    ])
    output = ROOT / "docs" / "Relatorio_Applied_Computer_Vision.pdf"
    SimpleDocTemplate(str(output), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm).build(story)
    print(f"PDF: {output}\nResumo: {len(RESUMO.split())} palavras\nObjetivo: {len(OBJETIVO.split())} palavras")


if __name__ == "__main__":
    build()
