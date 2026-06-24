# ADR-0013: Mecanismos de Atenção / Transformers no autoencoder

- **Status:** rejeitado (adiado para trabalho futuro condicionado)
- **Data:** 2026-06-24
- **Proveniência:** DECISÃO DE PROJETO
- **Milestone:** M12 (avaliação de ideias)
- **Chaves em `config.yaml`:** — (nenhuma; não implementado)

## Contexto

Proposta: incorporar uma camada de **Atenção** (ou blocos **Transformer**) ao
encoder/decoder, sob o argumento de que LSTMs "esquecem" em janelas longas e a atenção permitiria
focar em eventos passados específicos (p.ex. um choque macro de 6 meses atrás) na reconstrução.

## Defesa a favor

- Atenção captura dependências de longo alcance melhor que o estado oculto recorrente.
- Transformers são o estado da arte em sequências e dariam um "verniz" arquitetural moderno.
- Pesos de atenção são **interpretáveis** — mostrariam *quais* passos a rede usou.

## Defesa contra (decisiva neste projeto)

1. **A premissa não se aplica à nossa janela.** `window_size = 30` pregões
   ([ADR-0002](0002-window-size-30.md)) ≈ 6 semanas. Não há "esquecimento de longo prazo" a
   resolver — a LSTM lida com 30 passos sem dificuldade. Um evento de "6 meses atrás" **nem está na
   janela** (precisaria de `window_size ≈ 126`, que não escolhemos).
2. **O contexto de longo alcance já é injetado de outra forma.** A [ADR-0012](0012-contexto-macro-conditional-ae.md)
   dá o estado **sistêmico** (USD/BRL, VIX) como contexto — é o resumo macro de longo prazo, sem
   precisar atender a passos distantes.
3. **Dataset minúsculo.** ~2450 janelas/ativo (2010–2019). Transformers são *data-hungry*;
   atenção adiciona parâmetros. Em dados escassos, o resultado esperado é **overfitting**, não ganho.
4. **Evidência empírica de que não falta capacidade.** O `latent_dim` mostrou-se **insensível** em
   [8, 32] em todas as representações testadas (uni, multivariado, condicional —
   [ADR-0010](0010-validacao-walk-forward.md)/[ADR-0011](0011-tensor-multivariado-ohlcv.md)). Um modelo
   que não satura o gargalo **não está limitado por capacidade**; atenção não atacaria o gargalo real.
5. **Tema recorrente do projeto:** "mais dimensão/capacidade ≠ mais sinal" (OHLCV rejeitado, macro
   mensal inerte). Atenção é mais um passo na direção que os dados já rejeitaram.

## Decisão

**Não implementar.** A complexidade não se justifica para `window_size=30` num dataset pequeno cujo
gargalo já é insensível. Fica como **trabalho futuro condicionado**: só faria sentido se o projeto
adotasse janelas longas (p.ex. 126/252) *e* mais dados (mais ativos/maior histórico), cenário em que
o esquecimento de longo prazo passaria a existir de fato.

## Consequências

- Mantém a arquitetura enxuta e treinável em CPU em minutos.
- Registra explicitamente que a ideia foi avaliada e rejeitada **com razão**, não por omissão.
