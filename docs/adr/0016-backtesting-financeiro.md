# ADR-0016: Backtesting financeiro e impacto real

- **Status:** parcial — custo assimétrico aceito; backtest de portfólio rejeitado (trabalho futuro)
- **Data:** 2026-06-24
- **Proveniência:** DECISÃO DE PROJETO
- **Milestone:** M12
- **Código (parcial):** `src/evaluate.py::expected_cost`

## Contexto

Proposta: validar o modelo não só por erro (MSE/PR-AUC), mas por um **backtest de risco simulado** —
traduzir o alarme em **dinheiro** (custo de um falso positivo, impacto de um alarme atrasado num
portfólio), fechando a ponte entre a matemática e a utilidade prática.

## Defesa a favor

- Liga o trabalho ao **domínio do problema** (finanças), não só à estatística.
- "Quanto custa um erro?" é a pergunta que um praticante faria.
- Narrativa forte para a banca.

## Defesa contra (decisiva para o **backtest de portfólio**)

1. **Anomalia ≠ sinal de trade.** O sistema **detecta** anomalias; não diz *o que fazer* (vender?
   proteger? comprar?). Um backtest de P&L exige uma **estratégia** — uma hipótese que o projeto
   não tem e não se propôs a ter. Inventá-la é mudar o escopo (de detecção para *trading*).
2. **Risco de overclaim.** Um backtest ingênuo no período de teste fabrica facilmente uma história
   de "o modelo dá lucro" frágil a *look-ahead* na estratégia e a *overfit* do período. Seria
   estatisticamente irresponsável apresentá-lo como validação.
3. **Escopo.** Simulação de portfólio (custos de transação, *slippage*, sizing, latência do alarme) é
   praticamente **um segundo projeto**. Diluiria o foco do que está validado.

## Defesa a favor da versão **mínima** (custo assimétrico)

- O núcleo correto da ideia — "o custo de um FP ≠ o de um FN" — **é** traduzível sem inventar
  estratégia: pesos relativos `cost_fp`/`cost_fn` sobre a matriz de confusão.
- Conecta-se diretamente à escolha de limiar ([ADR-0005](0005-thresholds-estatico-e-dinamico.md)) e à
  PR-AUC ([ADR-0015](0015-metricas-classe-rara-pr-auc.md)).

## Decisão

- **Aceitar a versão mínima:** `expected_cost(flags, labels, cost_fp, cost_fn)` (implementado),
  reportando o custo total sob diferentes razões FN:FP. Isso já mostra que **a escolha de limiar é
  uma decisão de risco**, não só estatística (notebook `14`, Seção 3).
- **Rejeitar o backtest de portfólio** como entrega: sem uma estratégia justificada, seria
  especulação. Fica como **trabalho futuro** explícito, com pré-requisitos claros (definir a ação
  associada à anomalia, custos de transação, protocolo anti-*look-ahead*).

## Consequências

- A "ponte para o domínio" é feita pelo **custo assimétrico**, honesto e dentro do escopo.
- O relatório declara o backtest financeiro como direção futura — sem prometer um número de lucro
  que não pode ser defendido com rigor.

## Alternativas consideradas

- **Backtest completo de portfólio agora:** rejeitado — exige estratégia inexistente e arrisca
  overclaim.
- **Não tratar custo algum:** rejeitado — perderia o ponto válido (assimetria FP×FN), que é barato e
  informativo.
