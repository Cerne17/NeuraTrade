# Architecture Decision Records (ADRs)

Registro das decisões de projeto, com foco em **metodologia** e **hiperparâmetros**.
Cada decisão de configuração relevante tem um ADR; o `config.yaml` referencia o ADR
correspondente em comentário.

## Princípio de rigor

Toda escolha de hiperparâmetro é classificada por **proveniência**:

| Status | Significado |
| ------ | ----------- |
| **FUNDAMENTADO** | Ancorado no README do projeto e/ou na literatura referenciada. |
| **DEFAULT DE LITERATURA** | Valor convencional na implementação de referência (Valkov / Li), adotado por compatibilidade. |
| **DECISÃO DE PROJETO** | Escolha de design sem fonte direta; rationale explícito. |
| **PROVISÓRIO** | Placeholder a ser calibrado por experimento; **não** usar em resultado final sem validação. |

Valores `PROVISÓRIO` aparecem marcados também no `config.yaml`.

## Fontes

- **Li, S. (2020).** *Time Series of Price Anomaly Detection with LSTM.* Towards Data Science.
  https://medium.com/data-science/time-series-of-price-anomaly-detection-with-lstm-11a12ba4f6d9
- **Valkov, V.** *Time Series Anomaly Detection with LSTM Autoencoders using Keras.* curiousily.com.
  https://curiousily.com/posts/anomaly-detection-in-time-series-with-lstms-using-keras-in-python/
- **Petrovic, D.** *Anomaly Detection in Stock Price with LSTM Autoencoder.* GitHub.
- **Liu et al. (2025).** *Robust Anomaly Detection in Financial Markets Using LSTM Autoencoders and GANs.* OPAST.
- **(2021).** *Anomaly Detection on Bitcoin Values.* IEEE.
- **Kohavi, R. (1995).** *A Study of Cross-Validation and Bootstrap for Accuracy Estimation and Model
  Selection.* IJCAI. — recomenda **10-fold** como melhor compromisso viés/variância (ver ADR-0010).
- **Bergmeir, C. & Benítez, J. M. (2012).** *On the use of cross-validation for time series predictor
  evaluation.* Information Sciences. — valida forward-CV em séries temporais (não elege um `k`).
- **Bengio, Y. & Grandvalet, Y. (2004).** *No Unbiased Estimator of the Variance of K-Fold
  Cross-Validation.* JMLR.

## Índice

| ADR | Decisão | Status dominante |
| --- | ------- | ---------------- |
| [0001](0001-split-temporal-antes-do-scaler.md) | Split temporal antes da normalização; scaler só no treino | FUNDAMENTADO |
| [0002](0002-window-size-30.md) | `window_size = 30` | FUNDAMENTADO |
| [0003](0003-arquitetura-autoencoder.md) | Arquitetura: `lstm_units=64`, `latent_dim=16`, `dropout=0.2`, `loss=mse` | misto |
| [0004](0004-configuracao-de-treino.md) | Treino: adam, lr=1e-3, batch=32, val_split=0.1, **shuffle=False**, EarlyStopping | misto |
| [0005](0005-thresholds-estatico-e-dinamico.md) | Threshold estático p95 + dinâmico causal | misto |
| [0006](0006-avaliacao-injecao-e-setorial.md) | Avaliação: injeção sintética + P/R/F1 + comparação setorial | misto |
| [0007](0007-coleta-e-tratamento-amer3.md) | Coleta/cache, `auto_adjust`, tratamento do caso AMER3 | misto |
| [0008](0008-linha-do-tempo-eventos.md) | Linha do tempo de eventos + tolerância de matching ($\pm$`window_size`) | DECISÃO DE PROJETO |
| [0009](0009-agregacao-erro-janela.md) | Agregação do erro por janela: `mean`/`max`/`percentil` (Recall) | DECISÃO DE PROJETO · **aceito** (M8) |
| [0010](0010-validacao-walk-forward.md) | Validação Walk-Forward (`TimeSeriesSplit`) para seleção de hiperparâmetros | FUNDAMENTADO · **aceito** (M8) |
| [0011](0011-tensor-multivariado-ohlcv.md) | Tensor multivariado OHLCV `(30,1)→(30,5)`, scaler por coluna, `log1p` volume | DECISÃO DE PROJETO · **aceito** (M8, Etapa 1) |
| [0012](0012-contexto-macro-conditional-ae.md) | Contexto macro via Conditional AE (encoder vê macro, loss só em preço/volume) + decisão idiossincrático/sistêmico | DECISÃO DE PROJETO · **proposto** |

> **Fase 2 (M8) — concluída e validada por experimento:**
> - **ADR-0009** (`07_aggregation_recalibration`): agregação `max` dobrou o Recall (0,16→0,35) e
>   ainda elevou a Precision (0,55→0,84).
> - **ADR-0010** (`08_walkforward`): `latent_dim=16` confirmado; modelo insensível ao gargalo em
>   [8, 32] (diferenças $\ll$ desvio inter-fold).
> - **ADR-0011** (`09_multivariate_ohlcv`): Close+Volume `(30,2)` atribui o pico de volume ao
>   canal de volume (Δpreço ≈ 0) — capacidade ausente no univariado.
>
> **Consolidação (M9):**
> - **`max` adotado como default** (`10_max_default_decision`): no teste real 2020–2024 marca a
>   mesma fração de janelas que `mean` (~0,10), sem inflar falsos positivos. `config.yaml` →
>   `aggregation: max`; atualiza ADR-0005/0009.
> - **OHLCV completo rejeitado** (`11_ohlcv_full`): piora `val_loss` em 2/4 ativos sem ganho de
>   atribuição → permanece **Close+Volume**. `latent_dim` multivariado também insensível ([8,32]).

Template para novos ADRs: [TEMPLATE.md](TEMPLATE.md).
