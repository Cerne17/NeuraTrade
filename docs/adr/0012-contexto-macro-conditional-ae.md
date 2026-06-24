# ADR-0012: Injeção de Contexto Macroeconômico via Conditional Autoencoder

- **Status:** proposto
- **Data:** 2026-06-24
- **Proveniência:** DECISÃO DE PROJETO
- **Milestone:** M11 (proposto)
- **Chaves em `config.yaml`:** `macro.features`, `macro.stationarize`
- **Código (protótipo):** [src/macro.py](../../src/macro.py),
  [src/conditional.py](../../src/conditional.py),
  `src/model.py::build_conditional_autoencoder`

## Contexto

A detecção é local ao ativo (Close+Volume, [ADR-0011](0011-tensor-multivariado-ohlcv.md)). Ela
**não distingue** um estresse de liquidez **idiossincrático** (problema do próprio ativo) de um
**choque macroeconômico sistêmico** (que move o mercado todo). O objetivo desta proposta é dar
**contexto sistêmico** à rede com indicadores macro (USDBRL, VIX, Selic, IPCA, ...) e permitir a
distinção.

A tentação natural — empilhar a macro no mesmo tensor reconstruído `(30, 2+N)` e pontuar tudo
junto — **não** atinge o objetivo, por três razões medidas/argumentadas:

1. **MSE ignora a macro.** A macro de baixa frequência, após `ffill`, é quase-constante na
   janela; sua variância escalada é mínima, a reconstrução é trivial e ela **não gera gradiente**.
   O erro total vira proxy de preço/volume — a macro fica inerte (mesmo efeito do MSE desbalanceado
   da [ADR-0011](0011-tensor-multivariado-ohlcv.md), agora mais severo).
2. **Max-pooling sobre canais macro cria falsos positivos.** Uma alta de Selic **agendada** ou um
   *print* de IPCA viram um pico no canal macro → o `max` ([ADR-0009](0009-agregacao-erro-janela.md))
   marcaria um evento programado como "anomalia".
3. **Colapsar tudo num escore perde a informação que separa os regimes.** Idiossincrático vs
   sistêmico só é distinguível olhando **qual bloco** disparou.

## Decisão

Adotar um **Conditional Autoencoder**: a macro entra como **contexto** (condiciona o encoder),
mas **não** é reconstruída — a perda recai **apenas** sobre o bloco de preço/volume.

### Topologia (`build_conditional_autoencoder`)

```
Input (window, n_pv + n_macro)          # encoder VÊ preço/volume + macro
  → LSTM(64) → Dropout(0.2)
  → Dense(latent, activation="tanh")     # gargalo (tanh: preserva sinal no espaço macro)
  → RepeatVector(window)
  → LSTM(64, return_sequences=True) → Dropout(0.2)
  → TimeDistributed(Dense(n_pv))         # reconstrói SÓ preço/volume
loss = MSE(saída, X[..., :n_pv])         # alvo = bloco pv; macro nunca no alvo
```

Treino: `model.fit(X_full, X_full[..., :n_pv])`. A macro condiciona a codificação sem poluir a loss.

### Score por bloco e decisão de regime (`conditional.py`)

- **`pv_error`** — erro de reconstrução do bloco preço/volume (anomalia do ativo), agregado por
  `max` no tempo ([ADR-0009](0009-agregacao-erro-janela.md)).
- **`macro_stress`** — **amplitude intra-janela** do bloco macro (max−min por feature, média entre
  features), medida **direto da entrada** (não por reconstrução). Estável → ≈0; choque macro → alto.
- Limiares = percentil 95 de cada score **no treino** (normalidade,
  [ADR-0005](0005-thresholds-estatico-e-dinamico.md)).

| `pv_error` | `macro_stress` | regime |
| ---------- | -------------- | ------ |
| > limiar   | ≤ limiar       | **idiossincrático** |
| > limiar   | > limiar       | **sistêmico** |
| ≤ limiar   | qualquer       | normal |

Janela não-anômala é `normal` mesmo com macro mexendo: macro sozinha, sem estresse no ativo, não é
anomalia **do ativo**.

## Justificativa

- **Macro como contexto, não alvo**, resolve o "MSE ignora a macro": ela influencia o *encoding*
  sem competir na loss.
- **Estacionarizar a macro** (`macro.stationarize`: USDBRL→log-retorno, Selic→Δ, etc.) evita
  misturar retorno estacionário com nível em tendência (escala instável entre folds, AE modelando
  tendência).
- **`macro_stress` por amplitude** (não reconstrução) impede que eventos macro agendados inflem o
  erro do ativo e dá o eixo ortogonal para a decisão de regime.
- Reaproveita o que já existe: scaler por coluna/`walk_forward_splits_multivariate`
  ([ADR-0010](0010-validacao-walk-forward.md)) e a lógica per-canal
  ([ADR-0011](0011-tensor-multivariado-ohlcv.md)).

## Consequências (riscos a monitorar)

- **Contrato de data (CRÍTICO):** `df_macro` deve ser indexado pela **data de publicação**, não de
  referência. Índice por referência (IPCA de março disponível em 1/mar) **vaza o futuro** no `ffill`.
  Idem COPOM (decisão vs vigência). Sem `bfill` (lookahead em D0); descarta-se o cabeçalho.
- **Frequência mista:** indicadores mensais (IPCA/Selic) ficam quase-inertes após `ffill`; priorizar
  os **diários** (USDBRL, VIX, e candidatos como DI/CDS Brasil). Avaliar se mensais entram.
- **Scaler:** macro com tendência satura o `MinMaxScaler`; estacionarizar mitiga, mas considerar
  `RobustScaler` para o bloco macro (VIX tem spikes — COVID ~80).
- **Capacidade vs dataset pequeno:** mais features num dataset de ~2450 janelas → risco de overfit.
  `latent_dim` precisa ser **revalidado** por walk-forward (foi insensível em [8,32] no caso 2-feature,
  [ADR-0010](0010-validacao-walk-forward.md)); não assumir que expandir ajuda.
- **Validação do rótulo de regime:** a separação idiossincrático/sistêmico precisa ser checada
  contra eventos reais (crash COVID = sistêmico; fraude Americanas 2023 = idiossincrático) — é a
  prova do conceito ([ADR-0008](0008-linha-do-tempo-eventos.md)).
- **Conector de dados** (BCB/SGS, FRED) fica fora do protótipo; é pré-requisito para validar.

## Alternativas consideradas

- **Macro empilhada no tensor reconstruído `(30, 2+N)` + `max` global:** rejeitada — MSE ignora a
  macro, eventos agendados viram falsos positivos, e o escalar único impede distinguir regimes.
- **Loss ponderada por canal:** insuficiente — forçar ajustar uma quase-constante rende pouco; não
  cria o eixo de decisão.
- **AE separado só de macro (erro de reconstrução macro):** mais pesado; a amplitude intra-janela
  entrega o sinal de estresse macro sem um segundo modelo (pode ser revisitado se necessário).
