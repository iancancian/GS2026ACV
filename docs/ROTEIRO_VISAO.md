# Demonstração — Applied Computer Vision

O vídeo `Demonstracao_Applied_Computer_Vision.mp4` é silencioso e legendado.
Apresenta duas inferências reais executadas com `models/best.pt` em fotos do
teste, as imagens anotadas e a triagem. Ele é gerado por
`python scripts/gerar_video_visao.py`, com o ambiente do projeto e o dataset
importado. As imagens são do Drainage Dataset, wqwdas, CC BY 4.0; revise
privacidade e licença antes da publicação. Publique o vídeo em local acessível
ao professor e acrescente o link ao README.

Para recriar o PDF acadêmico, execute `python scripts/gerar_relatorio_visao.py`
em um ambiente com `reportlab` e `Pillow` instalados. Os números são lidos de
`docs/resultados_teste.json`.

Para acrescentar narração, sugestão:

1. Apresente o problema de alagamentos e o papel preventivo da inspeção visual.
2. Explique as cinco classes e o treino customizado YOLO11n.
3. Mostre as duas fotos de entrada e seus resultados, apontando caixas, classe
   e confiança. Os arquivos JSON e relatórios textuais da execução ficam em
   `output/demonstracao_visao/`.
4. Explique que o estado é uma estimativa pela sobreposição de caixas, não
   uma classe treinada nem confirmação de entupimento. A ausência de detecção
   não demonstra que o bueiro esteja livre.
5. Informe o mAP50 geral de 0,393 e o baixo recall para folha e sedimento.
   Encerre ressaltando a revisão humana e a necessidade de avaliação local.

O vídeo é uma demonstração de fotos processadas, não transmissão ao vivo.
O PDF da disciplina não informa duração máxima para este componente.
