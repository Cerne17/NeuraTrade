# ADR-0017: Avaliação por evento (agrupar janelas contíguas)

- **Status:** aceito — implementado
- **Data:** 2026-06-24
- **Proveniência:** FUNDAMENTADO
- **Milestone:** M12
- **Código:** `src/evaluate.py::group_events`, `src/evaluate.py::event_metrics`

## Contexto

Consequência registrada em [ADR-0015](0015-metricas-classe-rara-pr-auc.md) (e antecipada em
[ADR-0005](0005-thresholds-estatico-e-dinamico.md)): o protocolo sintético com **janelas
sobrepostas** (`step < window_size`) faz cada injeção pontual contaminar `~window_size` janelas
consecutivas. Com a injeção *default* (`n_injections=50`), a prevalência **ao nível de janela** sobe
a ~70% — a anomalia **deixa de ser rara** e a métrica não reflete o cenário 99/1 que o projeto
pretende avaliar.

Avaliar cada janela contaminada como uma instância independente premia/pune o detector dezenas de
vezes por um único choque, distorcendo Precision/Recall e inflando a baseline da PR-AUC.

## Decisão

Adotar **avaliação por evento**:

1. `group_events(labels)` agrupa corridas contíguas de janelas positivas num único intervalo
   `(start, end)` — um **evento**.
2. `event_metrics(flags, labels)` calcula Precision/Recall/F1 ao nível de evento:
   - **Recall (evento):** fração dos eventos verdadeiros com **ao menos uma** janela sinalizada
     dentro (pegar qualquer parte do evento = detectá-lo; um alarme tardio ainda conta).
   - **Precision (evento):** fração dos eventos *previstos* (corridas contíguas de flags) que
     sobrepõem algum evento verdadeiro (alarmes isolados fora de qualquer anomalia = FP).

A leitura por janela (`compute_metrics`) permanece disponível como contraste, mas a **leitura
padrão do protocolo sintético passa a ser por evento**.

## Justificativa

- Métrica baseada em evento/intervalo é a prática consolidada em detecção de anomalias em séries
  temporais (range-based / point-adjust): a unidade de interesse é o *episódio*, não o ponto.
- Desfaz a inflação de prevalência sem depender de calibrar `n_injections` — robusto ao
  espaçamento aleatório das injeções e ao `step` das janelas.
- Complementa, não substitui, a redução de `n_injections`: as duas alavancas atacam o mesmo
  defeito (regime artificialmente abundante) por vias diferentes.

## Consequências

- Fecha a pendência de "corrigir o protocolo sintético" registrada em ADR-0015.
- Recall por evento é **mais permissivo** que por janela (basta 1 acerto no evento) — coerente com
  o objetivo de *detectar o episódio*, não cada janela dele; reportar as duas leituras evita
  leitura enganosa.
- No teste **real** (não supervisionado) segue sem rótulos: avaliação por evento só se aplica ao
  sintético; no real, fração marcada + correlação a eventos ([ADR-0008](0008-linha-do-tempo-eventos.md)).

## Alternativas consideradas

- **Só reduzir `n_injections`:** ajuda na prevalência, mas não resolve a dupla-contagem de uma
  injeção espalhada por janelas vizinhas; mantém-se como alavanca complementar, não como única.
- **Janelas não-sobrepostas (`step = window_size`):** reduziria a contaminação, mas desalinharia
  do pipeline de detecção real (que usa `step=1`) e descartaria janelas — rejeitado.
