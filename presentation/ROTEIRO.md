# NeuraTrade — Roteiro de Apresentação (10 min)

Fala sugerida por slide. Tom direto, primeira pessoa do plural. Tempos são teto —
se atrasar, cortar primeiro os **[opcional]**. Divisão de voz: **[A]** Ana Beatriz,
**[M]** Miguel (ajuste como preferirem). Ensaiar a demo **uma vez antes**.

Orçamento: Ideia 1:30 · Metodologia 1:45 · Desafios 3:00 · Resultados+Demo 2:15 · Conclusão 0:45.

---

## Bloco 1 — A ideia (0:00–1:30)

**Slide 1 (Capa) — [A] 20s**
> "NeuraTrade: detecção **não supervisionada** de anomalias em ações da B3. A
> pergunta é simples — dá para uma rede aprender o que é um dia 'normal' na bolsa
> e apontar sozinha quando algo foge disso? Sem ninguém rotular o que é anomalia."

**Slide 2 (Problema) — [A] 40s**
> "Anomalia aqui é uma janela de tempo que destoa do padrão: um crash, uma fraude,
> um choque macro. O problema é que **não temos rótulos** — ninguém marcou 'aqui é
> anomalia'. E tem uma sutileza: nosso período 'normal', 2010 a 2019, já inclui
> recessão, Lava Jato, impeachment. 'Normal' é relativo, e o modelo tem que
> conviver com isso."

**Slide 3 (Ideia em uma frase) — [M] 30s**
> "A ideia central: treinar um **autoencoder** só no período normal. Ele aprende a
> **reconstruir** o padrão típico. Quando aparece algo que ele nunca viu, ele
> reconstrói mal — e esse **erro de reconstrução** alto é o nosso alarme. Um modelo
> por ativo, quatro setores, e cruzamos as anomalias com eventos brasileiros reais."

## Bloco 2 — Metodologia (1:30–3:15)

**Slide 4 (Arquitetura) — [M] 45s**
> "A entrada é uma janela de 30 pregões de log-retornos. Um **encoder LSTM**
> comprime isso num vetor de 16 dimensões — o gargalo — e um **decoder LSTM**
> tenta reconstruir a janela de volta. Se a reconstrução erra feio, é porque o
> padrão é estranho. A rede é pequena de propósito: são só ~2450 janelas por ativo,
> então capacidade demais só traria overfitting."

**Slide 5 (Sem vazamento) — [A] 60s**
> "Aqui está o ponto metodológico mais importante. A gente **separa treino e teste
> antes de normalizar**. Parece detalhe, mas se você normaliza com a série inteira,
> o scaler já 'viu' o futuro — é vazamento, e infla o resultado. Então o scaler é
> ajustado **só no treino**. O limiar de anomalia também vem do **erro de treino**,
> nunca do período que estamos avaliando. E para escolher hiperparâmetros usamos
> **walk-forward** com 10 folds, com uma régua honesta: uma melhora só vale se for
> **maior que a variação entre os folds**."

**Slide 6 (Abertura desafios) — [A] 15s**
> "Montar o autoencoder foi a parte fácil. A parte interessante foi o que **quebrou**
> no caminho. Quatro desafios, quatro decisões — todas medidas, todas documentadas."

## Bloco 3 — Desafios e soluções (3:15–6:30) · núcleo da apresentação

**Slide 7 (Vazamento + choque diluído) — [M] 60s**
> "Dois primeiros. Um: o vazamento que já mencionei — resolvido com o split antes do
> scaler, aplicado fold a fold. Dois, mais sutil: um choque de **um único dia** era
> **diluído** pela média sobre os 30 passos da janela. O pico sumia. Trocamos a
> agregação do erro de **média para máximo** — pega o pior passo da janela. O
> resultado foi medido: o Recall **dobrou**, de 0,16 para 0,35, e a Precisão ainda
> **subiu**, de 0,55 para 0,84. Uma troca de uma linha, ganho grande."

**Slide 8 (Idiossincrático vs sistêmico) — [A] 60s**
> "Terceiro desafio, o mais conceitual. Uma queda pode ser **do ativo** — uma fraude
> — ou **do mercado inteiro** — uma crise. O erro sozinho não distingue os dois.
> A solução foi um **Conditional Autoencoder**: o encoder enxerga o contexto macro,
> dólar e VIX, mas a perda continua só no preço e volume. Aí conseguimos separar. E
> funcionou: a **COVID em 2020 saiu como sistêmica**; a fraude da **Americanas em
> 2023 saiu como idiossincrática** — exatamente a distinção que a gente queria."

**Slide 9 (Métrica que engana) — [M] 60s**
> "Quarto: a métrica. Quando a classe é rara, a ROC-AUC **engana** — ela infla por
> causa do mar de casos normais. No regime raro, a ROC dava 0,84, parecia ótimo, mas
> a **PR-AUC expunha um 0,15** — o modelo ia mal e a ROC escondia. Trocamos para
> PR-AUC. E de brinde descobrimos um defeito nosso: as janelas sobrepostas inflavam
> a prevalência para 70%, então corrigimos para **avaliar por evento**, agrupando
> janelas contíguas num alarme só."

**Slide 10 (Tema central) — [A] 30s**
> "Se tem uma lição que atravessa o projeto é essa: **mais capacidade não é mais
> sinal**. Testamos aumentar o gargalo, atenção, Transformers, Optuna, e por último
> **weight decay** — e todos ficaram **dentro do ruído** entre os folds. O modelo
> já é robusto. O que move o ponteiro é *o que* você dá pra ele, não *quanto*."

## Bloco 4 — Resultados e demonstração (6:30–9:15)

**Slide 11 (Resultados) — [M] 45s**
> "Nos números: no teste real o detector marca cerca de 10% das janelas, sem
> explodir em falsos positivos. A separação idiossincrático/sistêmico bate com os
> eventos conhecidos. E tudo é reprodutível: roda offline, walk-forward, 32 testes
> automatizados, 18 decisões registradas como ADR. Agora deixa eu **mostrar ao vivo**."

**Slide 12 (Demo) — [M] 90s** — ver **Roteiro da Demo** abaixo.

## Bloco 5 — Conclusões (9:15–10:00)

**Slide 13 (Conclusões) — [A] 45s**
> "Fechando: um autoencoder aprende a normalidade e sinaliza desvios **sem rótulo
> nenhum**. Mas o coração do trabalho não é a rede — é o **rigor**: anti-vazamento,
> walk-forward, PR-AUC, avaliação por evento. E honestidade: a gente rejeitou várias
> ideias 'modernas' **com prova na mão**, não por preguiça. Como trabalho futuro
> fica o backtest financeiro, que exige uma estratégia de trade que foge do escopo.
> Obrigado — perguntas?"

---

## Roteiro da Demo (Slide 12) — 90 segundos, 3 gestos

Antes de começar: `streamlit run demo/sandbox.py`, ativo **PETR4**, período **teste**,
janela do navegador já aberta. Se a internet cair, tudo bem — **roda offline**.

1. **(20s) `mean` → `max`.** "Aqui está o erro por janela da PETR4 em 2020–2024.
   Com agregação **média**, olhem o pico da COVID — meio apagado. Troco para
   **máximo**…" *(muda o seletor)* "…e o pico **salta** acima do limiar. Foi
   exatamente o ganho de Recall do slide anterior, agora ao vivo."

2. **(30s) subir o percentil do limiar.** "O limiar é o percentil do erro de
   **treino** — nunca deste período, pra não vazar. Se eu subo de p95 para p99…"
   *(arrasta)* "…fico mais conservador, menos alarmes. É o trade-off Precision×Recall
   na mão de quem opera."

3. **(30s) injetar choque + métrica.** "Agora ligo a **injeção sintética**: coloco
   choques artificiais em posições conhecidas e o painel calcula **Recall, Precisão
   e PR-AUC ao vivo**." *(aumenta `n_injections` / `k·σ`)* "Choque mais forte, o
   detector pega mais — e a PR-AUC responde na hora."

4. **(10s) fechamento.** "Repare no que **não** tem slider aqui: `latent_dim`,
   weight decay, arquitetura. Esses a gente **decidiu por experimento** — mudar exige
   retreino. O sandbox mostra que o detector é robusto a *como* se lê o erro."

**Plano B (se o Streamlit falhar):** mostrar o CSV `figures/experiment_weight_decay.csv`
e as figuras em `report/figures/` (`m5_distribuicao_erro.png`, `m6_covid_contagio.png`),
narrando os mesmos 3 pontos. Ensaiar este plano B também.

---

## Checklist pré-apresentação
- [ ] `streamlit run demo/sandbox.py` abre e responde (testar os 3 gestos).
- [ ] Modelos presentes em `models/` (senão `python -m src --train`).
- [ ] Figuras de reserva abertas em abas.
- [ ] Cronômetro no palco — o núcleo (Desafios) é onde se ganha ou perde tempo.
- [ ] Definir quem fala cada slide e o **handoff** entre A e M.
