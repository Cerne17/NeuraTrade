# ADR-0001: Split temporal antes da normalização; scaler ajustado só no treino

- **Status:** aceito
- **Data:** 2026-06-22
- **Proveniência:** FUNDAMENTADO
- **Milestone:** M2 (Pré-processamento)
- **Chaves em `config.yaml`:** `data.train_end`, `data.start`, `data.end`

## Contexto

Detecção não supervisionada: o modelo aprende "normalidade" no treino e mede desvio
no teste. Qualquer informação do futuro que vaze para o pré-processamento contamina a
avaliação e infla artificialmente o desempenho.

## Decisão

1. Split **temporal** (não aleatório): treino = 2010-01-01 a 2019-12-31; teste = 2020 a 2024.
2. O `MinMaxScaler` é **ajustado (`fit`) apenas no treino** e só então **aplicado
   (`transform`)** no teste.
3. A normalização ocorre **depois** do split, nunca sobre a série inteira.

## Justificativa

Explicitado no README ("split temporal antes da normalização… para evitar vazamento de
informação do futuro"). É a prática padrão para séries temporais: ajustar o scaler sobre
o conjunto completo vazaria estatísticas (min/max) do período de teste para o treino.

## Consequências

- Valores de teste podem cair **fora de `[0, 1]`** após o `transform` (extremos pós-2020
  além do range de treino). Isso é **esperado e desejável** — é justamente onde moram as
  anomalias. Não fazer clipping silencioso. Ver tensão com MinMax em [ADR-0003](0003-arquitetura-autoencoder.md).
- A definição de "normalidade" é relativa: o treino 2010–2019 contém volatilidade real
  (recessão 2014–2016, Lava Jato, impeachment 2016). Limitação assumida e discutida no relatório.

## Alternativas consideradas

- **Split aleatório / k-fold:** rejeitado — quebra a ordem temporal e vaza futuro.
- **StandardScaler / RobustScaler:** ver [ADR-0003](0003-arquitetura-autoencoder.md); README fixa MinMax.
