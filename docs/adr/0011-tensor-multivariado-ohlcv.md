# ADR-0011: Tensor multivariado OHLCV (30,1) → (30,5)

- **Status:** aceito — Etapa 1 (Close+Volume) validada em M8, notebook `09_multivariate_ohlcv`
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

> **Atualização (M8, 2026-06-23) — execução e evidência (issues #52/#53).**
> Implementado em [src/preprocessing.py](../../src/preprocessing.py)
> (`preprocess_ticker_multivariate`, `build_features`, `make_windows` ND) e
> [src/train.py](../../src/train.py) (`n_features` inferido, sufixo `_multi`). Notebook
> `09_multivariate_ohlcv` treinou a Etapa 1 (Close+Volume, `(30,2)`) nos quatro ativos.
>
> - **Treino saudável:** `val_loss` ≈ 0,0021–0,0025 (abaixo do univariado ≈ 0,004; dois canais,
>   escala por coluna efetiva). Nenhum canal colapsou.
> - **Atribuição correta (teste decisivo):** injetando um pico de volume (`k·σ`) só no canal de
>   volume, a variação média do erro per-canal (`max`) nas janelas injetadas foi:
>
> | Ticker | Δ erro canal **preço** | Δ erro canal **volume** |
> | --- | --- | --- |
> | PETR4.SA | +0,00002 | **+0,05084** |
> | VALE3.SA | −0,00011 | **+0,07575** |
> | AMER3.SA | +0,00011 | **+0,35980** |
> | ITUB4.SA | −0,00002 | **+0,05799** |
>
> - **O canal de volume dispara, o de preço não se move** (Δpreço ≈ 0). A anomalia é atribuída
>   ao volume — exatamente a capacidade que o univariado não tem (a injeção de `volume_spike`,
>   [ADR-0006](0006-avaliacao-injecao-e-setorial.md), antes impossível, agora é avaliável).
> - **`log1p` + scaler por coluna confirmados:** sem eles, o volume (~1e7) dominaria a perda e
>   saturaria pós-2020. Os dois canais conviveram sem que um anulasse o outro.
> - **Pendente:** Etapa 2 (OHLCV completo `(30,5)`) só se acrescentar sinal sobre Close+Volume;
>   e validar `latent_dim` do multivariado com a malha walk-forward ([ADR-0010](0010-validacao-walk-forward.md)).
>   O univariado permanece o default em produção.

> **Atualização (M9, 2026-06-23) — Etapa 2 decidida: ficar em Close+Volume (issues #59/#60).**
> Notebook `11_ohlcv_full` treinou OHLCV `(30,5)` vs Close+Volume `(30,2)` nos quatro ativos.
>
> - **OHLCV não compensa:** o `val_loss` da Etapa 2 **piora** em 2 de 4 ativos (VALE3
>   $0{,}00232 \to 0{,}00273$; AMER3 $0{,}00254 \to 0{,}00355$) e melhora marginalmente nos
>   outros — reconstruir 5 canais é mais difícil sem ganho de detecção. A **atribuição do pico de
>   volume é idêntica** em `(30,2)` e `(30,5)`. **Decisão: permanecer em Close+Volume `(30,2)`;
>   Etapa 2 (OHLCV completo) rejeitada.**
> - **Ressalva à premissa de colinearidade:** os *log-retornos* de O/H/L/C **não** são ~99%
>   correlacionados (corr. observada $0{,}39$–$0{,}75$); o ~99% referia-se aos *níveis* de preço.
>   Ainda assim, a informação extra não ajudou a detecção — o argumento de parcimônia se sustenta
>   pelo resultado, não pela colinearidade.
> - **`latent_dim` multivariado (#60):** walk-forward (`walk_forward_splits_multivariate`) sobre
>   Close+Volume deu `val_loss` médio $0{,}01415$ (8), $0{,}01427$ (16), $0{,}01433$ (32) — de
>   novo **insensível**, com leve vantagem para 8. Mantém-se `latent_dim=16` por consistência com
>   o univariado ([ADR-0003](0003-arquitetura-autoencoder.md)); a diferença é desprezível.
> - **`config.yaml`** segue `features: [Close, Volume]`. O univariado permanece o default em
>   produção; o multivariado é a via para anomalias de volume quando requisitado.

## Alternativas consideradas

- **Manter univariado:** rejeitado — impossibilita anomalias de volume, uma das motivações da Fase 2.
- **OHLCV full direto:** evitado como primeiro passo — 4 canais colineares, maior superfície de
  bug, ganho incerto antes de validar Close+Volume.
- **Volume sem `log1p`:** rejeitado — satura no teste por não-estacionariedade.
