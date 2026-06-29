# ADR-0018: Weight decay (AdamW) — avaliado e rejeitado como default

- **Status:** rejeitado como default (knob mantido, `weight_decay=0`)
- **Data:** 2026-06-28
- **Proveniência:** FUNDAMENTADO (medido por walk-forward)
- **Milestone:** M13 (experimentos de regularização)
- **Chaves em `config.yaml`:** `model.weight_decay` (default `0.0`)
- **Código:** `src/model.py::build_lstm_autoencoder` (AdamW quando `weight_decay>0`),
  `src/validation.py::cross_validate_weight_decay`, `scripts/experiment_weight_decay.py`

## Contexto

O modelo só regularizava por `Dropout(0.2)` ([ADR-0003](0003-arquitetura-autoencoder.md)) e
EarlyStopping ([ADR-0004](0004-configuracao-de-treino.md)) — **sem weight decay** (otimizador
`Adam` puro, não `AdamW`). Levantou-se a hipótese de que uma penalidade L2 desacoplada melhoraria a
generalização num dataset pequeno (~2450 janelas/ativo).

## Decisão

Tornar `weight_decay` um **knob** (`AdamW` quando `>0`, `Adam` quando `0`) e medir seu efeito por
**walk-forward** (ADR-0010) antes de adotar. Resultado: **manter `weight_decay=0` (Adam puro) como
default**; o knob fica disponível para experimentos futuros.

## Justificativa (evidência: `scripts/experiment_weight_decay.py`)

Walk-forward `n_splits=10`, candidatos `[0, 1e-5, 1e-4, 1e-3]`, `val_loss` médio (melhor época por
EarlyStopping) por ativo:

| Ativo | wd=0 | wd=1e-5 | wd=1e-4 | wd=1e-3 | desvio inter-fold |
| --- | --- | --- | --- | --- | --- |
| PETR4.SA | 0,015911 | 0,015839 | **0,015822** | 0,015835 | ±0,0113 |
| VALE3.SA | **0,017678** | 0,017684 | 0,017692 | 0,017751 | ±0,0154 |
| AMER3.SA | 0,015628 | 0,015747 | **0,015548** | 0,015788 | ±0,0166 |
| ITUB4.SA | 0,011056 | **0,011036** | 0,011044 | 0,011081 | ±0,0083 |

Os deltas vs baseline ficam em **1e-4 a 1e-5** — **100 a 1000× menores que o desvio inter-fold**
(~1e-2). Nenhum candidato, em nenhum ativo, sai do ruído: o sinal de melhora/piora é estatisticamente
**indistinguível de zero**. A direção do "melhor" wd nem é consistente entre ativos (1e-4 em
PETR4/AMER3, 1e-5 em ITUB4, 0 em VALE3), o que confirma ruído, não tendência.

## Consequências

- **Default permanece Adam puro** (`weight_decay=0`); a regularização efetiva segue sendo
  `Dropout(0.2)` + EarlyStopping.
- Mesma régua que rejeitou `latent_dim` (ADR-0010), atenção (ADR-0013) e Optuna (ADR-0014): a
  diferença precisa **superar o desvio inter-fold**. Reforça o tema central — *mais
  capacidade/regularização ≠ mais sinal*: o gargalo já é insensível, então mexer na penalidade de
  peso não tem o que corrigir.
- O knob fica registrado e testável; reabrir só se a arquitetura crescer (mais features/camadas) a
  ponto de o overfitting por magnitude de peso passar a existir de fato.

## Alternativas consideradas

- **Adotar `weight_decay=1e-4` como default** (melhor em 2/4 ativos): rejeitado — o ganho está dentro
  do ruído inter-fold e não se sustenta nos outros ativos.
- **L2 via `kernel_regularizer` nas camadas** (em vez de AdamW): não testado — AdamW é o decay
  desacoplado padrão e cobre a hipótese; mesma expectativa de inércia dado o gargalo insensível.
