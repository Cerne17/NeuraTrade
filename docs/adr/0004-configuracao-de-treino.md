# ADR-0004: Configuração de treino

- **Status:** aceito
- **Data:** 2026-06-22
- **Proveniência:** misto
- **Milestone:** M3 (Modelo & Treino)
- **Chaves em `config.yaml`:** `train.optimizer`, `train.learning_rate`, `train.batch_size`, `train.validation_split`, `train.shuffle`, `train.epochs`, `train.early_stopping_patience`

## Contexto

Definir otimizador, parada e como separar a validação — esta última é o ponto
metodológico mais delicado em série temporal.

## Decisão

| Parâmetro | Valor | Proveniência |
| --------- | ----- | ------------ |
| `optimizer` | `adam` | DEFAULT — Li (2020), Valkov. |
| `learning_rate` | 0.001 | DEFAULT — padrão do Adam. |
| `batch_size` | 32 | DEFAULT — Valkov. |
| `validation_split` | 0.1 | DEFAULT — Valkov. |
| `shuffle` | **false** | FUNDAMENTADO (crítico) — Valkov: `shuffle=False`. |
| `epochs` | 100 | DECISÃO DE PROJETO — teto; EarlyStopping decide a parada. |
| `early_stopping_patience` | 10 | DECISÃO DE PROJETO. |

## Justificativa

- **`shuffle=False` é obrigatório.** O `validation_split` do Keras separa a **última fração**
  do array como validação **antes** de embaralhar. Se `shuffle=True`, cada época re-embaralha o
  treino e — combinado com janelas sobrepostas ([ADR-0002](0002-window-size-30.md)) — mistura
  temporalmente amostras, vazando padrão futuro para o treino. Com `shuffle=False`, treino e
  validação ficam em blocos cronológicos contíguos. Valkov fixa exatamente isso.
- `epochs=100` como teto alto deixa o `EarlyStopping` (monitor `val_loss`, `patience=10`,
  `restore_best_weights=True`) determinar a parada real — evita over/undertraining manual por ativo.

## Consequências

- A validação é o **bloco final** do treino (≈ 2018–2019). Janelas na fronteira treino/validação
  compartilham dias por causa da sobreposição — vazamento residual pequeno, aceito e documentado.
- A validação serve só para EarlyStopping/threshold, **não** é o conjunto de teste (2020+).
- `EarlyStopping` deve usar `restore_best_weights=True`; sem isso, salva os pesos da última época
  (pior). Verificar na implementação de `src/train.py` (M3).

## Alternativas consideradas

- **`TimeSeriesSplit` do sklearn:** mais rigoroso para CV, mas o fluxo Keras `fit` com
  `validation_split` + `shuffle=False` basta para o escopo e segue a referência.
- **Sem validação (treinar nº fixo de épocas):** rejeitado — impede EarlyStopping e calibração objetiva.
