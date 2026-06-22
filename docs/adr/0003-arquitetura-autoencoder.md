# ADR-0003: Arquitetura do LSTM-Autoencoder

- **Status:** aceito (com itens PROVISÓRIOS)
- **Data:** 2026-06-22
- **Proveniência:** misto (ver por parâmetro)
- **Milestone:** M3 (Modelo & Treino)
- **Chaves em `config.yaml`:** `model.lstm_units`, `model.latent_dim`, `model.dropout`, `model.loss`

## Contexto

Encoder LSTM → bottleneck → Decoder LSTM, otimizando erro de reconstrução. Um modelo por
ativo (ver [ADR-0006](0006-avaliacao-injecao-e-setorial.md)). É preciso fixar largura das
camadas, dimensão do gargalo, regularização e função de perda.

## Decisão

| Parâmetro | Valor | Proveniência |
| --------- | ----- | ------------ |
| `lstm_units` | 64 | DEFAULT DE LITERATURA — Valkov: `units=64` nas duas camadas LSTM. |
| `dropout` | 0.2 | DEFAULT DE LITERATURA — Valkov: `rate=0.2`. |
| `latent_dim` | 16 | DECISÃO DE PROJETO — gargalo; compressão ~4:1 vs. `lstm_units`. |
| `loss` | `mse` | FUNDAMENTADO — README e Li (2020). |

Estrutura: `LSTM(64) → (gargalo) → RepeatVector(window_size) → LSTM(64, return_sequences)
→ TimeDistributed(Dense(n_features))`, com `Dropout(0.2)` entre camadas.

## Justificativa

- `lstm_units=64` e `dropout=0.2` reproduzem a implementação de referência de Valkov, que
  por sua vez segue o padrão da literatura de AE-LSTM em séries financeiras.
- `latent_dim=16`: a literatura **não padroniza** a dimensão do gargalo (Valkov nem usa um
  `latent_dim` explícito — comprime via `RepeatVector`). 16 é escolha de design para forçar
  compressão sem estrangular demais a reconstrução. **Deve ser validado em M3** (estudo de
  sensibilidade: 8/16/32).
- `loss=mse`: o README especifica MSE. Li (2020) também usa MSE. (Valkov usa MAE — ver alternativas.)

## Consequências

- **Tensão MinMax × reconstrução:** com `MinMaxScaler` ajustado no treino ([ADR-0001](0001-split-temporal-antes-do-scaler.md)),
  valores de teste podem sair de `[0,1]`. MSE penaliza esses extremos quadraticamente, o que
  **ajuda** a detecção (erro dispara em anomalias) mas torna o threshold sensível a outliers
  isolados — daí o uso de **percentil** e não do máximo em [ADR-0005](0005-thresholds-estatico-e-dinamico.md).
- `latent_dim` muito pequeno → underfitting (erro alto em tudo, baixa separabilidade);
  grande demais → o AE "copia" a entrada e deixa de flagrar anomalias. Monitorar curva de
  reconstrução treino vs. validação.

## Alternativas consideradas

- **`loss=mae` (Valkov):** mais robusto a outliers, mas o README fixou MSE; manter MSE por
  consistência com a proposta. MAE registrado como sensibilidade opcional.
- **VAE / camadas convolucionais:** fora de escopo da proposta.
