# ADR-0006: Avaliação por injeção sintética + comparação setorial

- **Status:** aceito (magnitudes PROVISÓRIAS)
- **Data:** 2026-06-22
- **Proveniência:** misto
- **Milestone:** M5 (Avaliação), M6 (Eventos & Comparação Setorial)
- **Chaves em `config.yaml`:** `evaluation.n_injections`, `evaluation.price_shock`, `evaluation.volume_spike`

## Contexto

O problema é não supervisionado: não há rótulos de anomalia reais. Para obter métricas
quantitativas (Precision/Recall/F1) é preciso conhecer a posição-verdade de algumas anomalias.

## Decisão

Avaliação em duas vias:

1. **Narrativa:** conferir se anomalias detectadas coincidem com eventos reais conhecidos
   (crash mar/2020, caso Americanas jan/2023, etc.) — via `src/events.py`.
2. **Quantitativa:** **injetar anomalias sintéticas** (price shocks e volume spikes) em
   posições conhecidas e medir Precision / Recall / F1.

Parâmetros: `n_injections=50`, `price_shock=0.15`, `volume_spike=5.0` — todos **PROVISÓRIOS**.

Um modelo por ativo, permitindo **comparação de generalização entre setores** (energia,
mineração, varejo, financeiro).

## Justificativa

- **Injeção sintética + métricas (FUNDAMENTADO conceitualmente):** Liu et al. (2025) propõem
  explicitamente um "artificial anomaly injection mechanism that simulates realistic market
  irregularities" e avaliam "across six representative stock categories" — exatamente o desenho
  adotado aqui (injeção + avaliação setorial).
- **Magnitudes PROVISÓRIAS:** Liu et al. não publicam os valores; `0.15` / `5.0` / `50` são
  placeholders. Devem ser **calibrados em M5 relativos à distribuição empírica dos retornos/volume
  de cada ativo** (p.ex. choque = k desvios-padrão do retorno), não fixados em absoluto — senão a
  dificuldade da detecção varia arbitrariamente entre ativos.

## Consequências

- Métricas dependem de como detecções contíguas viram "eventos" — uma anomalia injetada cai em até
  30 janelas ([ADR-0002](0002-window-size-30.md)). Definir regra de matching (tolerância temporal)
  antes de calcular P/R/F1 e documentá-la.
- A taxa-base de ~5% do p95 ([ADR-0005](0005-thresholds-estatico-e-dinamico.md)) limita a precisão
  máxima atingível; reportar junto.
- **Comparação setorial:** com um modelo por ativo, diferenças de métrica podem refletir o ativo,
  não o setor (n=1 por setor). Tratar como estudo de caso qualitativo, não inferência estatística.

## Alternativas consideradas

- **Rótulos manuais de anomalia real:** subjetivo e não reproduzível; injeção dá verdade controlada.
- **Apenas validação narrativa:** insuficiente para métrica quantitativa exigida pela proposta.

> **Atualização (M5/M6, 2026-06-23) — implementação e evidência.**
> - **Espaço de injeção:** os `price_shock` são somados ao log-retorno **já escalado**
>   (`apply_scaler`), não ao preço bruto — é o espaço em que o modelo foi treinado e evita
>   que a magnitude do choque varie com o nível de preço do ativo (`src/evaluate.py:inject_price_shocks`).
> - **Regra de rotulagem por janela:** uma janela é positiva se **qualquer** passo injetado
>   cair nela (`labels_to_window_labels`), pois um choque eleva o erro da janela inteira. O
>   vetor de rótulos segue a indexação de `make_windows`, mantendo alinhamento exato com `X_test`.
>   Métricas via `sklearn` com `zero_division=0` (`compute_metrics`).
> - **Tolerância de matching narrativo:** $\pm$`window_size` (30 dias) entre anomalia e evento
>   (ver [ADR-0008](0008-linha-do-tempo-eventos.md), que formaliza a regra e a curadoria dos eventos).
> - **Evidência empírica (teste 2020–2024):** com magnitude **absoluta** `0.15`, o *recall* é
>   uniformemente baixo nos ativos estáveis (PETR4/VALE3 recall <7%; o choque compete com o
>   ruído normal) e razoável onde o erro de base é alto (AMER3 estático: P≈0,69 R≈0,30 F1≈0,42).
>   Isto **confirma** a necessidade de calibrar o choque em $k\sigma$ por ativo: a magnitude
>   absoluta torna a dificuldade incomparável entre setores. A avaliação atual valida o
>   *pipeline* de métricas, não é resultado final.
> - **Escopo:** apenas *price shocks* implementados nesta etapa; `volume_spike` (chave no
>   `config.yaml`) segue previsto mas não usado.
> - **Pendente:** calibração relativa a $\sigma$ e estudo de sensibilidade de `dynamic_window`
>   ([ADR-0005](0005-thresholds-estatico-e-dinamico.md)).
