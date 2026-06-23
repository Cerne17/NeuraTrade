# ADR-0011: Tensor multivariado OHLCV (30,1) → (30,5)

- **Status:** proposto
- **Data:** 2026-06-23
- **Proveniência:** DECISÃO DE PROJETO
- **Milestone:** M8 (Evolução da Modelagem · Fase 2)
- **Chaves em `config.yaml`:** `preprocessing.features`, `preprocessing.volume_log1p`
- **Issues:** #51 (ADR), #52 (preprocessing), #53 (model/train)

## Contexto

O modelo é **univariado**: a entrada é o log-retorno escalado, tensor `(window, 1)`
([ADR-0001](0001-split-temporal-antes-do-scaler.md), [src/preprocessing.py](../../src/preprocessing.py)).
Anomalias **precursoras baseadas em volume** (spikes de volume que antecedem o movimento de
preço) são, por construção, invisíveis a essa representação. Os dados crus já contêm OHLCV
([src/data.py](../../src/data.py), [ADR-0007](0007-coleta-e-tratamento-amer3.md)).

## Decisão

Expandir a entrada para múltiplas features, partindo do mínimo informativo e crescendo só se
houver ganho:

1. **Etapa 1 — Close + Volume `(30, 2)`:** captura o canal de volume (o prêmio real) com
   custo mínimo.
2. **Etapa 2 — OHLCV `(30, 5)`:** Open/High/Low/Close + Volume, **se** a Etapa 1 indicar ganho
   que justifique o custo.

Pré-processamento: **`log1p` no Volume** antes de escalar e **`MinMaxScaler` por coluna**,
ajustado **apenas no treino** (mantém [ADR-0001](0001-split-temporal-antes-do-scaler.md)).
O caminho univariado permanece o default.

## Justificativa

- **Volume é canal precursor:** spikes de volume antecedem movimento de preço — informação
  ausente do log-retorno.
- **Close+Volume antes de OHLCV:** Open/High/Low/Close são ~99% colineares; agregam pouca
  informação nova sobre o Close. O custo de 4 canais quase redundantes raramente compensa o
  ganho de 1 (Volume). Validar o incremento antes de pagar por ele.

## Consequências (riscos a monitorar)

- **Escalas heterogêneas:** Volume ~1e7 vs log-retorno ~1e-2. Scaler **global** faria o volume
  dominar a perda. `MinMaxScaler` **por coluna** é obrigatório.
- **Não-estacionariedade do Volume:** o Volume tem tendência de longo prazo; o range de treino
  (2010–2019) **satura** em 2020+ (volumes COVID estouram o `max` de treino → tudo vira ~1.0,
  anomalia trivial). `log1p` antes de escalar mitiga. Coerente com a observação de que valores
  de teste podem sair de [0,1] ([ADR-0001](0001-split-temporal-antes-do-scaler.md)).
- **MSE desbalanceado entre canais:** mesmo em [0,1], a variância por canal difere; a
  reconstrução otimiza o canal fácil e ignora o difícil, e o erro total vira proxy de 1–2
  canais. Considerar peso por canal e/ou erro **per-canal**.
- **Atribuição da anomalia:** o erro passa a ser um vetor de N canais. Guardar o erro
  **per-canal** para distinguir anomalia *volume-driven* vs *price-driven* — senão perde-se o
  objetivo declarado (precursor de volume). Impacta a detecção ([ADR-0005](0005-thresholds-estatico-e-dinamico.md))
  e a injeção sintética ([ADR-0006](0006-avaliacao-injecao-e-setorial.md): `volume_spike`,
  hoje NÃO implementado por falta de canal de volume).
- **Maior capacidade, mais overfitting:** 5× a entrada exige mais sinal de "normal"; validar com
  Walk-Forward ([ADR-0010](0010-validacao-walk-forward.md)).

## Alternativas consideradas

- **Manter univariado:** rejeitado — impossibilita anomalias de volume, uma das motivações da Fase 2.
- **OHLCV full direto:** evitado como primeiro passo — 4 canais colineares, maior superfície de
  bug, ganho incerto antes de validar Close+Volume.
- **Volume sem `log1p`:** rejeitado — satura no teste por não-estacionariedade.
