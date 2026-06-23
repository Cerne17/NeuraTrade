# ADR-0010: Validação Walk-Forward (TimeSeriesSplit)

- **Status:** aceito (executado em M8, notebook `08_walkforward`)
- **Data:** 2026-06-23
- **Proveniência:** FUNDAMENTADO (rigor metodológico de séries temporais)
- **Milestone:** M8 (Evolução da Modelagem · Fase 2)
- **Chaves em `config.yaml`:** `validation.n_splits` (nova seção)
- **Issues:** #48 (ADR), #49 (validation.py), #50 (notebook)

## Contexto

A seleção/validação de hiperparâmetros (em especial `latent_dim`,
[ADR-0003](0003-arquitetura-autoencoder.md)) apoia-se hoje num **holdout único** ao final do
treino (`validation_split = 0.1`, [ADR-0004](0004-configuracao-de-treino.md)). Uma única
partição de validação produz uma estimativa de erro com **alta variância**: a escolha do
hiperparâmetro pode refletir as particularidades de um só bloco temporal.

## Decisão

Adotar **validação Walk-Forward** com `sklearn.model_selection.TimeSeriesSplit`: treinar em
fatias **expansíveis** do passado e avaliar no bloco futuro adjacente, repetindo por `n_splits`
folds. Novo módulo `src/validation.py` (`walk_forward_splits`) e notebook `07_walkforward.ipynb`
para a seleção de `latent_dim`.

A validação Walk-Forward é ferramenta de **seleção/credibilidade**, não altera o detector em
produção; o modelo final é retreinado com o melhor hiperparâmetro sobre o treino completo.

## Justificativa

- **Rigor temporal:** `TimeSeriesSplit` respeita a ordem cronológica (nunca treina no futuro
  para prever o passado), coerente com toda a metodologia do projeto
  ([ADR-0001](0001-split-temporal-antes-do-scaler.md), `shuffle=False`).
- **Variância reduzida:** múltiplos folds dão média ± desvio do `val_loss` por candidato,
  base honesta para comparar `latent_dim` (ex.: 8/16/32).
- **`n_splits = 10` (FUNDAMENTADO, com ressalva):** Kohavi (1995) recomenda **10-fold** como o
  melhor compromisso viés/variância para estimação de acurácia e seleção de modelo, valor
  adotado como convenção. **Ressalva metodológica:** o estudo de Kohavi trata *k-fold
  estratificado sobre dados i.i.d.* (classificação); aqui usa-se *walk-forward sobre série
  temporal* (sem embaralhar, folds expansíveis), então o "10 ótimo" não transfere literalmente.
  Avaliou-se explicitamente se algum `k` diferente teria fundamentação **mais forte** para este
  cenário e a conclusão foi que **não**: a literatura de CV em séries temporais
  (Bergmeir & Benítez, 2012) valida o *esquema* forward-CV por preservar a causalidade, mas
  **não elege um `k` ótimo**; argumentos por `k` menor (custo) ou atado ao tamanho do fold
  (treino mínimo) são heurísticos, não teóricos; e Bengio & Grandvalet (2004) mostram que nem
  existe estimador não-viesado da variância do `k`-fold. Sem um resultado que supere 10, mantém-se
  **10 por convenção** (default consolidado), com esta ressalva registrada. O custo é gerenciável
  (folds menores treinam rápido; ver Consequências).

## Consequências

- **Anti-vazamento por fold (CRÍTICO):** o `MinMaxScaler` deve ser **refitado dentro de cada
  fold**, apenas sobre `train_idx`; `transform` no bloco de validação, **nunca** refit.
  Fitar o scaler antes do split contamina todos os folds (lookahead). Mantém a restrição da
  [ADR-0001](0001-split-temporal-antes-do-scaler.md) **por fold**.
- **Janelamento dentro do fold:** janelas deslizantes ([ADR-0002](0002-window-size-30.md)) que
  cruzam a fronteira treino/val misturam blocos. Gerar janelas **dentro** de cada partição ou
  descartar as que cruzam o corte.
- **Custo computacional:** retreina o LSTM `K` vezes. Mitigar com folds menores / menos epochs
  no CV; treino final completo.
- **Semântica dos folds:** o treino é "normalidade" 2010–2019 ([ADR-0001](0001-split-temporal-antes-do-scaler.md));
  documentar o que cada fold representa para que o `val_loss` seja interpretável.
- Resultado pode atualizar [ADR-0003](0003-arquitetura-autoencoder.md) (`latent_dim`).

> **Atualização (M8, 2026-06-23) — execução e evidência (issue #50).**
> Implementado em [src/validation.py](../../src/validation.py) e orquestrado no notebook
> `08_walkforward`: `TimeSeriesSplit` com `n_splits=5`, scaler refitado por fold, janelas dentro
> de cada recorte. Seleção de `latent_dim ∈ {8, 16, 32}` sobre os quatro ativos (período de
> treino), `val_loss` médio entre ativos:
>
> | `latent_dim` | val\_loss médio (4 ativos) | desvio entre ativos |
> | --- | --- | --- |
> | 8  | 0,015537 | 0,004453 |
> | **16** | **0,015406** | 0,004398 |
> | 32 | 0,015462 | 0,004317 |
>
> - **`latent_dim = 16` confirmado** (menor `val_loss` médio), validando a escolha de projeto
>   do [ADR-0003](0003-arquitetura-autoencoder.md).
> - **Achado principal — insensibilidade:** as três dimensões empatam (diferenças na 4ª casa
>   decimal, $\ll$ desvio inter-fold $\approx 0{,}004$). No intervalo [8, 32] o tamanho do
>   gargalo **não** é hiperparâmetro sensível: a reconstrução é robusta à compressão. Isso é uma
>   evidência positiva de robustez, não um empate inconclusivo.
> - O valor de `config.yaml` (`latent_dim=16`) permanece, agora **fundamentado por experimento**
>   em vez de só por design. A malha walk-forward fica disponível para validar futuros
>   hiperparâmetros e o modelo multivariado ([ADR-0011](0011-tensor-multivariado-ohlcv.md)).

> **Atualização (2026-06-23) — `n_splits` 5 → 10 (Kohavi 1995).**
> Adotou-se `n_splits=10` por convenção (Kohavi, 1995), com a ressalva i.i.d. registrada na
> Justificativa. Reexecutado o `08_walkforward` com 10 folds (`val_loss` médio entre os 4 ativos):
>
> | `latent_dim` | val\_loss médio (4 ativos) | desvio entre ativos |
> | --- | --- | --- |
> | 8  | 0,015426 | 0,002902 |
> | **16** | **0,015399** | 0,002888 |
> | 32 | 0,015450 | 0,002976 |
>
> - **Mesma conclusão:** `latent_dim=16` segue o melhor (e o gargalo segue **insensível** em
>   [8, 32]). A mudança de `k` **não** alterou o veredito --- robustez confirmada por um segundo
>   desenho de validação.
> - O desvio entre ativos caiu (≈0,0029 vs 0,0044 com 5 folds): mais folds → média um pouco mais
>   estável, exatamente o efeito de redução de variância que motiva o 10-fold.

## Alternativas consideradas

- **Holdout único (status quo):** rejeitado para seleção — estimativa de alta variância.
- **K-Fold embaralhado:** **proibido** em série temporal — vaza futuro no treino, invalida a
  estimativa.
- **Blocked/purged CV (gap entre train e val):** mais conservador; fica como evolução futura
  se a contiguidade das janelas mostrar contaminação residual na fronteira.
