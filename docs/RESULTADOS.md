# Resultados do modelo

Detector YOLO treinado com o *Drainage Dataset* (wqwdas, Roboflow Universe,
CC BY 4.0), 963 imagens divididas em 674 (train), 193 (val) e 96 (test).
Números abaixo vêm de `docs/resultados_teste.json`, gerado por
`python -m boca_de_lobo.evaluate --weights models/best.pt` no split **test**,
que não foi usado para escolher parâmetros.

## Configuração do treino

| Item | Valor |
|---|---|
| Modelo | YOLO11n (pesos iniciais `yolo11n.pt`, pré-treinado no COCO) |
| Parâmetros | 80 épocas, imgsz 640, batch 16, semente 42, paciência 20 |
| Hardware / software | GTX 1060 6 GB, Ultralytics 8.4.215, PyTorch 2.14.1+cu126 |
| Pré-processamento | Fotos reduzidas a no máximo 1280 px no maior lado (`import_roboflow`) |
| Augmentation | Somente o padrão do Ultralytics no treino; o export do Roboflow não tem |

## Desempenho no conjunto de teste (96 imagens)

| Classe | Instâncias | Precisão | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|---:|
| **Geral** | 179 | 0,516 | 0,445 | 0,393 | 0,155 |
| bueiro | 46 | 0,759 | 0,761 | 0,732 | 0,285 |
| lixo | 51 | 0,410 | 0,569 | 0,432 | 0,198 |
| folha | 45 | 0,214 | 0,244 | 0,107 | 0,042 |
| pedra | 21 | 0,846 | 0,524 | 0,536 | 0,199 |
| sedimento | 16 | 0,351 | 0,125 | 0,159 | 0,053 |

Na validação (193 imagens) o melhor ponto foi mAP50 0,40 e mAP50-95 0,20, em
linha com o teste.

## Como interpretar

- **Bueiro é a classe que sustenta o produto** e é a mais confiável (recall 0,76):
  é ela que decide se há bueiro para avaliar. Sem detecção do bueiro, a
  triagem vai para `revisao_manual`, nunca para "livre".
- **Folha e sedimento são fracos** (recall 0,24 e 0,13). O estado do bueiro,
  calculado pela cobertura, **subestima** obstruções desses materiais. Por isso
  toda prioridade exige aprovação humana e a ausência de obstrução detectada
  não significa bueiro livre.
- Os conjuntos são pequenos (16 a 51 instâncias por classe no teste): as
  métricas têm variação grande e servem como linha de base, não como garantia.
- O dataset não informa o local das fotos. A divisão do Roboflow pode misturar
  fotos parecidas do mesmo local entre os splits; o teste pode estar otimista.
  O validador só bloqueia imagens idênticas.
- Os limiares de cobertura (10% e 80%) são uma convenção inicial, ainda não
  calibrada com fotos do bairro-piloto.

## Próximos passos para melhorar

1. Coletar fotos locais (protocolo em `DATASET.md`) e usá-las como teste.
2. Aumentar exemplos de folha, sedimento e pedra e rever a anotação dessas classes.
3. Comparar com YOLO26n usando os mesmos dados e a mesma divisão
   (`--model yolo26n.pt --name yolo26n`); essa comparação não foi concluída.
4. Calibrar os limiares de cobertura e o limiar de confiança com dados locais.

## Execução de ponta a ponta (amostra)

`logs/amostra_execucao.log`: 16 registros de entrada, 12 processados e 4
rejeitados de propósito (imagem ausente, área desconhecida, caminho inválido e
ativo duplicado). Previsão real da Open-Meteo: máximo de 0,2 mm/h nas próximas
24 h, ou seja, sem chuva forte no dia. `logs/amostra_demo_chuva_forte.log` repete
o lote com `--rain-threshold 0.15` para exercitar o caminho de alerta (1 aviso
pendente de aprovação). Essa execução é demonstração do fluxo, não uma
previsão de chuva forte.
