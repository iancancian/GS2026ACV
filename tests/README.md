# Testes e validação

```powershell
python -m unittest discover -s tests -v
```

A suíte roda sem GPU, sem internet e sem o dataset: o detector e a previsão do
tempo são substituídos por dublês (`fake_detector`, `fake_forecast`).

## Cenários exigidos (sucesso, decisão e falha)

| Tipo | Cenário | Teste |
|---|---|---|
| Sucesso | Lote válido gera fila, tarefas e avisos; arquivos gravados sem `.tmp` residual | `test_automation::test_success_and_decision_produce_integrated_outputs` |
| Sucesso | Cliente de previsão devolve o máximo horário e ignora valores nulos | `test_weather::test_returns_max_hourly_precipitation_ignoring_missing_values` |
| Decisão | Cobertura ≥ 80% → `entupido_provavel` e prioridade `alta` com chuva forte | `test_decision::test_blocked_drain_under_strong_rain_is_high_priority` |
| Decisão | Obstrução parcial só vira `alta` em área de risco | `test_decision::test_partial_obstruction_escalates_only_in_risk_area` |
| Decisão | Sobreposição de obstruções não é contada duas vezes; limiares 10% e 80% | `test_decision::CoverageTests` |
| Decisão | Limiar de chuva forte configurável | `test_automation::test_rain_threshold_is_configurable` |
| Falha | Registro incompleto, área desconhecida, duplicado, inconsistente, caminho fora da pasta, imagem inválida ou ausente, erro do detector: o lote continua e o motivo é registrado | `test_automation::test_item_failures_are_recorded_and_batch_continues` |
| Falha | API de previsão fora: 3 tentativas com espera crescente, depois `indisponivel` (nunca "sem chuva") | `test_weather`, `test_automation::test_forecast_outage_is_explicit_and_never_means_no_rain` |
| Falha | Entrada ilegível ou sem áreas válidas aborta o lote | `test_automation::test_fatal_input_errors_abort_the_batch` |
| Falha | Consumidor recusa fila com versão incompatível ou sem aprovação humana | `test_automation::test_consumer_rejects_incompatible_or_unapproved_queue` |
| Dados | Importação por nome de classe, polígonos, ZIP sem extração, redução de imagens, destino protegido | `test_import_roboflow` |
| Dados | Pares imagem/rótulo, coordenadas, imagens duplicadas entre splits, vazamento por ativo | `test_dataset` |

## Evidência de execução real

- Resultado da suíte: ver `python -m unittest discover -s tests` (todos passam).
- `logs/amostra_execucao.log`: lote real com sucessos, decisões e quatro falhas tratadas.
- `logs/amostra_demo_chuva_forte.log`: mesmo lote com o caminho de alerta ativo.
- `docs/resultados_teste.json`: métricas do detector no split de teste.

Os logs registram apenas identificadores (`BUEIRO-007`), nomes de arquivo e
motivos; não registram caminhos locais, mensagens internas de exceção nem dados pessoais.
