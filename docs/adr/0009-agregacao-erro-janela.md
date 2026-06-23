# ADR-0009: Agregação do erro de reconstrução por janela (max/percentil)

- **Status:** proposto
- **Data:** 2026-06-23
- **Proveniência:** DECISÃO DE PROJETO
- **Milestone:** M8 (Evolução da Modelagem · Fase 2)
- **Chaves em `config.yaml`:** `detection.aggregation`, `detection.aggregation_percentile`
- **Issues:** #45 (ADR), #46 (detect.py), #47 (recalibração)

## Contexto

O erro de reconstrução por janela é hoje a **média do erro absoluto** (MAE) sobre os
`window_size = 30` passos (ver [ADR-0002](0002-window-size-30.md), [ADR-0005](0005-thresholds-estatico-e-dinamico.md)):
`mean(|recon − X|)` sobre os eixos `(tempo, features)` em [src/detect.py](../../src/detect.py).

Um **choque sintético de um único dia** ([ADR-0006](0006-avaliacao-injecao-e-setorial.md))
entra na janela com peso `1/30`. A média **dilui** esse pico entre 29 dias normais, podendo
manter o erro da janela abaixo do limiar → o detector perde a anomalia (**Recall baixo**)
exatamente no tipo de evento que mais interessa.

## Decisão

Parametrizar a agregação temporal do erro por janela, com três modos:

1. **`mean`** (atual, default retrocompatível) — média sobre o tempo.
2. **`max`** — erro do pior passo da janela; preserva o choque pontual.
3. **`percentile`** — percentil alto (p.ex. p90–p95) sobre o tempo; meio-termo robusto
   entre `mean` e `max`.

A redução acontece em duas etapas, **nesta ordem**: primeiro média sobre o eixo de
*features* (`axis=-1`), depois agregação (`mean`/`max`/`percentile`) sobre o eixo *temporal*
(`axis=1`).

## Justificativa

- **`max` recupera o sinal pontual:** o evento de 1 dia deixa de competir com a média e
  sobrevive à agregação → Recall esperado ↑ em price shocks.
- **`percentile` como compromisso:** `max` puro é refém de **um** ponto ruidoso (um dia
  normal atípico vira falso positivo). O percentil alto da janela mantém a sensibilidade ao
  pico sem depender de um único passo — análogo ao racional que já rejeitou o *máximo do erro
  de treino* em favor do p95 ([ADR-0005](0005-thresholds-estatico-e-dinamico.md)).
- **Ordem dos eixos é crítica:** agregar antes de reduzir features, ou trocar os eixos,
  produz um vetor de forma errada **sem lançar exceção** — bug silencioso. Exige teste de shape.

## Consequências

- **Recalibração obrigatória do limiar (issue #47):** `max`/`percentile` mudam a distribuição
  inteira do erro. O `static_threshold`/`dynamic_threshold` (p95) foi calibrado sobre
  **Mean-MAE** e **não transfere**. Recalcular o limiar sobre o **mesmo** score agregado do
  treino; reusar o p95 antigo infla os falsos positivos.
- **Trade-off Precision↓:** maior sensibilidade ao pico aumenta a chance de marcar ruído
  normal. Reavaliar P/R/F1 (mean vs max vs percentil) nos quatro ativos antes de fixar o modo.
- **Comparabilidade:** ao reportar resultados, deixar explícito qual agregação gerou cada
  métrica; não comparar Recall de `max` com limiar de `mean`.

## Alternativas consideradas

- **Manter `mean` (status quo):** rejeitado — é a causa direta do Recall baixo em choques curtos.
- **`max` como único modo:** evitado como default — frágil a um ponto ruidoso; oferecido, mas
  o percentil é o compromisso recomendado.
- **top-k mean (média dos k maiores passos):** equivalente prático ao percentil; pode entrar
  como variante futura se o percentil não bastar.
