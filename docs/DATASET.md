# Protocolo para banco de imagens YOLO

## 1. Objetivo e classes

Usar deteccao de objetos YOLO. Cada boca de lobo visivel recebe **uma caixa ao
redor da abertura** (`bueiro`) e cada agrupamento visivel de material que a
bloqueia ou esta proximo recebe sua propria caixa. O estado do bueiro (livre,
parcial, entupido) **nao e uma classe**: e calculado pela cobertura (ver
README), o que mantem as classes iguais as do dataset publico base.

| ID | Nome YOLO | Origem (Roboflow) | Regra de anotacao |
|---:|---|---|---|
| 0 | `bueiro` | `drainage` | Grade ou abertura de drenagem; caixa na abertura inteira. |
| 1 | `lixo` | `trash` | Residuos soltos (plastico, embalagens, papel); uma caixa por agrupamento. |
| 2 | `folha` | `leaf` | Folhas e material vegetal; uma caixa por agrupamento. |
| 3 | `pedra` | `rock` | Pedras e entulho. |
| 4 | `sedimento` | `silt` | Lama, areia e sedimento acumulado. |

Com o dataset base, o importador faz o mapeamento por nome e recusa classes
desconhecidas. Em fotos locais, use as mesmas regras e confira que as caixas de
obstrucao cobrem apenas o material visivel. Se a grade ou a obstrucao nao puder
ser julgada visualmente, nao adivinhe: descarte a imagem ate a decisao. Foto
sem deteccao de `bueiro` nao equivale a bueiro livre.

### Dataset base e atribuicao

Fonte: [wqwdas/drainage-dataset](https://universe.roboflow.com/wqwdas/drainage-dataset),
CC BY 4.0. Antes de treinar: prefira um fork sem augmentation (ver README), abra uma
amostra de cada classe, remova imagens com pessoas e confira se `bueiro` e
`sedimento` tem exemplos suficientes por split. O dataset nao informa o local
das fotos (`site_group` fica `desconhecido`); copias aumentadas do mesmo original
compartilham `asset_id` e o validador bloqueia imagens identicas em splits
diferentes. Mesmo assim, o conjunto de teste publico mede desempenho no dominio
do dataset, nao no seu bairro.

## 2. Coleta segura e consistente

- Obtenha autorizacao para a coleta e use somente locais publicos ou autorizados.
- Fotografe a estrutura e a sarjeta em enquadramento legivel; evite pessoas,
  placas de veiculos, interiores e fachadas residenciais.
- Se pessoa ou placa aparecer, descarte a foto ou desfaca a identificacao antes
  de qualquer compartilhamento. Nao use reconhecimento de pessoas.
- Registre um `asset_id` pseudonimo e um `site_group` (quadra/setor amplo), nunca
  nome de morador ou endereco residencial. Nao registre coordenada GPS precisa
  no dataset de treinamento.
- Capture variacao de modelo de grade, distancia, angulo, sombra, sol, horario,
  piso seco/molhado e folhas/lixo diferentes. Evite fotos em condicao perigosa.
- Para o mesmo ativo, capture uma foto de referencia e, quando houver mudanca,
  outra inspecao identificada por data. Nao conte rajadas de fotos quase
  identicas como exemplos independentes.
- Armazene os originais com acesso restrito; publique apenas amostras revisadas
  e licenciadas. Documente origem, autorizacao, retencao e eventual desfocagem.

## 3. Meta inicial e diversidade

O dataset base (963 imagens) e o ponto de partida. Para fotos locais, mire
**200–400 imagens utilizaveis** com exemplos claros de cada classe, distribuidas
entre locais, ativos, iluminacao e clima; elas servem para calibrar limiares e,
idealmente, formar um conjunto de teste do seu bairro. Isso e uma meta
exploratoria, nao uma garantia de desempenho. Se uma classe for rara, registre a
limitacao; nao duplique imagens para aparentar volume.

Mantenha `train`, `val` e `test` aproximadamente em 70/20/10 por grupos de
locais/ativos, nao por foto. Um `site_group` ou `asset_id` nao pode aparecer em
mais de uma particao. Guarde `test` sem ajustes iterativos e use-o somente para
a avaliacao final. Com poucos locais independentes, declare a limitacao e nao
apresente o resultado como generalizavel para outras cidades.

## 4. Pastas e anotacao

```text
dataset/
  data.yaml
  metadata.example.csv
  metadata.csv                 # local; nao versionado
  images/
    train/  val/  test/
  labels/
    train/  val/  test/
```

Cada imagem deve ter um `.txt` com o mesmo nome-base, inclusive se nao houver
objeto alvo (arquivo vazio). Formato de cada linha:

```text
class_id x_center y_center width height
```

Coordenadas sao normalizadas entre 0 e 1 em relacao a largura/altura da imagem.
Use uma ferramenta de anotacao que exporte YOLO Detection, confira classes e
caixas manualmente e faca uma segunda revisao dos exemplos ambiguos e de uma
amostra de cada anotador. Nao misture formato YOLO Segmentation com este projeto.

## 5. Manifesto de metadados

Copie `metadata.example.csv` para `metadata.csv` e acrescente uma linha por
imagem:

- `image_path`: caminho relativo, por exemplo `images/train/IMG_0001.jpg`;
- `site_group`: codigo amplo usado para separar os conjuntos (`desconhecido` e
  reservado a datasets publicos sem informacao de local);
- `asset_id`: identificador pseudonimo do ponto inspecionado;
- `captured_at`: data/hora ISO 8601, sem publicar coordenada;
- `lighting`, `weather`: categorias controladas e concisas;
- `annotator_code`: codigo nao identificavel do anotador;
- `source_dataset`, `source_license`, `source_url`, `source_author`: origem e
  licenca de imagens de terceiros (preenchidos pelo importador).

Nao inclua nome, telefone, endereco, rosto, placa ou coordenada precisa.
`metadata.csv` e fotos/anotacoes locais sao ignorados pelo Git. Antes de
compartilhar, revise licenca, privacidade e autorizacao do conjunto.

## 6. Validacao e treino

1. Importe o dataset base (README, passo 3) e retire imagens ilegiveis, ambiguas
   ou com pessoas.
2. Atribua cada grupo de local/ativo de fotos locais a exatamente um split.
3. Confira o manifesto e nomes de arquivos em pares imagem/label.
4. Rode `python -m boca_de_lobo.dataset`; corrija todos os erros.
5. Rode os testes `python -m unittest discover -s tests -v`.
6. Treine com `python -m boca_de_lobo.train --model yolo11n.pt`.
7. Use `python -m boca_de_lobo.evaluate --weights CAMINHO_PARA_best.pt` uma unica
   vez no conjunto `test`, sem ajustar parametros com base nesse resultado.
8. Registre versao de pesos iniciais, parametros, versao das dependencias,
   distribuicao de classes e resultados por classe. Revise falsos negativos e
   positivos em fotos nunca usadas no treino; documente limites de luz, chuva,
   tipos de grade e bairros antes de demonstrar.

Pesos iniciais podem ser baixados pelo Ultralytics na primeira execucao. Use
somente fontes e licencas adequadas ao trabalho e registre a origem. O treino
nao deve ser apresentado como concluido ate que esses passos sejam realizados
com o dataset real.
