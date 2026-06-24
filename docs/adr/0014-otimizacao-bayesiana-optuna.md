# ADR-0014: Otimização Bayesiana de hiperparâmetros (Optuna)

- **Status:** rejeitado (não justificado pela evidência atual)
- **Data:** 2026-06-24
- **Proveniência:** DECISÃO DE PROJETO
- **Milestone:** M12 (avaliação de ideias)
- **Chaves em `config.yaml`:** — (nenhuma; não implementado)

## Contexto

Proposta: integrar **Optuna** para **otimização bayesiana** dos hiperparâmetros (`latent_dim`,
`learning_rate`, ...), substituindo a escolha "manual / grid" por uma busca inteligente guiada pelo
histórico, dando mais rigor científico à topologia.

## Defesa a favor

- Otimização bayesiana explora o espaço de forma mais eficiente que grid/manual.
- Dá um processo de seleção sistemático e reportável (estudos, importâncias de hiperparâmetro).
- Padrão moderno em ML aplicado.

## Defesa contra (decisiva neste projeto)

1. **A seleção não foi "manual/grid casual".** Usou-se **validação Walk-Forward** com
   `TimeSeriesSplit` ([ADR-0010](0010-validacao-walk-forward.md)) — a abordagem **correta para série
   temporal** (Optuna sem CV temporal seria *menos* rigoroso, não mais).
2. **A paisagem é plana — e isso foi medido.** O `latent_dim` é **insensível** em [8, 32]: as
   diferenças de `val_loss` ficam na 4ª casa decimal, muito abaixo do desvio entre folds
   (≈0,003), em todas as representações (uni, multivariado, condicional). Rodar otimização bayesiana
   sobre uma superfície plana **não acha nada melhor** — só produz uma falsa sensação de precisão
   ("o ótimo é 17,3!") sobre ruído.
3. **Os hiperparâmetros sensíveis têm proveniência, não busca.** `window_size=30`, `lstm_units=64`,
   `dropout=0.2`, `lr=1e-3` vêm da literatura de referência (Valkov/Li,
   [ADR-0003](0003-arquitetura-autoencoder.md)/[ADR-0004](0004-configuracao-de-treino.md)) — escolhas
   defensáveis e reproduzíveis. Otimizá-las num dataset pequeno arrisca **overfitar o
   hiperparâmetro** ao período de validação.
4. **Custo vs ganho.** Optuna adiciona dependência + dezenas a centenas de treinos. Já temos a
   resposta (insensibilidade) por um custo muito menor (3 candidatos × walk-forward).

## Decisão

**Não implementar.** A evidência diz que a superfície de `latent_dim` é plana; busca bayesiana
adicionaria "verniz" sem substância. Fica como **trabalho futuro condicionado**: justificável **se**
a arquitetura crescer (atenção, muito mais features) e a paisagem deixar de ser plana, ou se entrarem
hiperparâmetros genuinamente sensíveis e acoplados.

## Consequências

- Mantém o pipeline sem dependências extras e reproduzível.
- A honestidade fica registrada: preferimos relatar a **insensibilidade medida** a esconder essa
  realidade atrás de uma busca cara.
