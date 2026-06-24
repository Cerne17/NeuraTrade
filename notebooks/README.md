# notebooks/ — estudos

Os notebooks são os **estudos** do projeto: exploram, treinam, medem e visualizam,
sempre **orquestrando a lógica de `src/`** (os notebooks não contêm regra de negócio).
Estão versionados **com saídas**, para registrar a evolução — não apenas o estado final.

> Para a **pipeline de produção** (reproduzível em um comando), use `python -m src`
> (ver [`src/pipeline.py`](../src/pipeline.py)), não os notebooks.

Execute na ordem numérica. Cada um consome `data/raw/` (versionado) e, a partir de M3,
modelos em `models/` (gerados por `03_train` ou `python -m src --train`).

## Comece aqui — guia de uso

| # | Notebook | Conteúdo |
| - | -------- | -------- |
| 00 | `00_quickstart` | **Guia de uso** (não é um estudo): exercita **todas as configurações** do sistema — pipeline de detecção, troca de agregação (`mean`/`max`/`percentile`), multivariado Close+Volume + atribuição por canal, contexto macro (regime idiossincrático/sistêmico), inferência em janela nova — com o script/CLI equivalente em cada passo. |

## Fase 1 — pipeline base (M1–M6)

| # | Notebook | Estudo |
| - | -------- | ------ |
| 01 | `01_eda` | Inspeção das séries, gaps, integridade dos dados (M1). |
| 02 | `02_preprocessing` | Log-retornos, split temporal, normalização, janelas (M2). |
| 03 | `03_train` | Treino do LSTM-Autoencoder, um modelo por ativo (M3). |
| 04 | `04_detection_thresholds` | Erro de reconstrução; limiar estático vs. dinâmico (M4). |
| 05 | `05_evaluation_synthetic` | Injeção sintética de anomalias; Precision/Recall/F1 (M5). |
| 06 | `06_events_correlation` | Sobreposição das anomalias com eventos econômicos/políticos (M6). |

## Fase 2 — evolução da modelagem (M8)

| # | Notebook | Estudo |
| - | -------- | ------ |
| 07 | `07_aggregation_recalibration` | Agregação `max`/`percentil` do erro + recalibração do limiar (ADR-0009). |
| 08 | `08_walkforward` | Seleção de `latent_dim` por validação walk-forward (ADR-0010). |
| 09 | `09_multivariate_ohlcv` | Entrada Close+Volume; atribuição de anomalia de volume (ADR-0011). |

## Consolidação (M9)

| # | Notebook | Estudo |
| - | -------- | ------ |
| 10 | `10_max_default_decision` | `max` no teste real 2020–2024; adoção como default (ADR-0009). |
| 11 | `11_ohlcv_full` | OHLCV `(30,5)` vs Close+Volume; `latent_dim` multivariado (ADR-0011). |

## Contexto macro (M11)

| # | Notebook | Estudo |
| - | -------- | ------ |
| 12 | `12_conditional_macro` | Conditional AE com macro (USDBRL/VIX/Selic/IPCA): distingue **idiossincrático** vs **sistêmico** — COVID=sistêmico, Americanas=idiossincrático (ADR-0012). |
| 13 | `13_conditional_tuning` | Tuning: dropar macro mensais inertes (Selic/IPCA) e revalidar `latent_dim` condicional por walk-forward (ADR-0012). |

As decisões de cada estudo estão registradas como ADRs em [`docs/adr/`](../docs/adr/).
