# ADR-0015: Métricas para classe rara — PR-AUC e custo assimétrico

- **Status:** aceito — implementado (notebook `14_imbalanced_metrics`)
- **Data:** 2026-06-24
- **Proveniência:** FUNDAMENTADO
- **Milestone:** M12
- **Código:** `src/evaluate.py::score_curves`, `src/evaluate.py::expected_cost`

## Contexto

Proposta: sob forte desbalanceamento (anomalias raras), a **ROC-AUC engana** — o excesso de
verdadeiros-negativos infla a pontuação. Uma avaliação rigorosa usaria a **Curva Precision–Recall
(PR-AUC)** e o custo assimétrico de Falsos Positivos vs Falsos Negativos.

## Defesa a favor

- A crítica é **correta e clássica**: em classe rara a ROC superestima o desempenho.
- Custo baixo: o ground-truth sintético ([ADR-0006](0006-avaliacao-injecao-e-setorial.md)) torna
  PR-AUC computável; é uma adição de relato, não um novo modelo.
- Alinha o relatório à boa prática de detecção de anomalias.

## Defesa contra (parcial, e por isso valiosa)

- Já reportávamos **Precision/Recall/F1** (não acurácia/ROC), então parte do mérito já existia.
- **Surpresa empírica:** ao nível de **janela**, com a injeção default (50 choques), a anomalia
  **não é rara** — janelas sobrepostas fazem cada choque rotular ~`window_size` janelas, levando a
  prevalência a ~70%. No regime abundante ROC≈PR e o alerta não morde.

## Decisão

**Implementar `score_curves` (PR-AUC + ROC-AUC + baseline/lift) e `expected_cost` (FP×FN)**, e
adotar **PR-AUC + F1** como leitura padrão; a ROC-AUC fica só como contraste didático.

### Evidência (notebook `14`, PETR4, varrendo a raridade)

| `n_injections` | prevalência | **PR-AUC** | ROC-AUC |
| --- | --- | --- | --- |
| 2  | 0,049 | **0,151** | 0,840 |
| 5  | 0,105 | 0,314 | 0,847 |
| 10 | 0,188 | 0,537 | 0,921 |
| 25 | 0,447 | 0,707 | 0,878 |
| 50 | 0,715 | 0,865 | 0,885 |

No regime **raro** (`n=2`, prevalência 4,9%) a **ROC-AUC = 0,84 mascara** um desempenho fraco que a
**PR-AUC = 0,15** expõe (apenas ~3× o acaso). Conforme a classe se equilibra, as duas convergem.
**A ideia é confirmada — e expôs um defeito do nosso sintético.**

## Consequências

- **Reportar PR-AUC, não ROC-AUC**, ao avaliar o detector; incluir a prevalência como baseline.
- **Corrigir o protocolo sintético** para avaliar no regime realmente raro: reduzir `n_injections`
  e/ou contar eventos contíguos como um só (agrupar detecções), para que a métrica reflita o
  cenário 99/1. **Endereçado em [ADR-0017](0017-avaliacao-por-evento.md)** (`group_events` +
  `event_metrics`); reduzir `n_injections` segue como alavanca complementar.
- **Limite honesto:** no teste **real** (não supervisionado) não há rótulos → PR-AUC só é computável
  sobre o sintético; no real, segue-se com fração marcada + correlação a eventos
  ([ADR-0008](0008-linha-do-tempo-eventos.md)).
- O custo FP×FN (`expected_cost`) liga-se à validação financeira ([ADR-0016](0016-backtesting-financeiro.md)).

## Alternativas consideradas

- **Manter só ROC-AUC/acurácia:** rejeitado — engana em classe rara.
- **F1-macro:** útil, mas com 2 classes a PR-AUC (independente de limiar) é mais informativa que um
  F1 num único ponto; ambos são reportados.
