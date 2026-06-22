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

## Índice

| ADR | Decisão | Status dominante |
| --- | ------- | ---------------- |
| [0001](0001-split-temporal-antes-do-scaler.md) | Split temporal antes da normalização; scaler só no treino | FUNDAMENTADO |
| [0002](0002-window-size-30.md) | `window_size = 30` | FUNDAMENTADO |
| [0003](0003-arquitetura-autoencoder.md) | Arquitetura: `lstm_units=64`, `latent_dim=16`, `dropout=0.2`, `loss=mse` | misto |
| [0004](0004-configuracao-de-treino.md) | Treino: adam, lr=1e-3, batch=32, val_split=0.1, **shuffle=False**, EarlyStopping | misto |
| [0005](0005-thresholds-estatico-e-dinamico.md) | Threshold estático p95 + dinâmico causal | misto |
| [0006](0006-avaliacao-injecao-e-setorial.md) | Avaliação: injeção sintética + P/R/F1 + comparação setorial | misto |

Template para novos ADRs: [TEMPLATE.md](TEMPLATE.md).
