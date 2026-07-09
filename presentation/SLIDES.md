# NeuraTrade — Guia de Slides (apresentação de 10 min)

Detecção não supervisionada de anomalias em ações da B3 com LSTM-Autoencoder.
**13 slides**, ritmo direto. Cada slide: título curto, 3–5 bullets, 1 visual.
Tempo-alvo por bloco em `ROTEIRO.md`. Figuras já prontas em `report/figures/`.

> Regra de ouro: **1 ideia por slide**. Não ler o slide — o slide é apoio, a fala
> é o conteúdo (ver roteiro).

---

## Slide 1 — Capa (0:00–0:20)
- **Título:** NeuraTrade — Detecção Não Supervisionada de Anomalias em Ações da B3
- Subtítulo: LSTM-Autoencoder · PETR4 · VALE3 · AMER3 · ITUB4
- Autores: Ana Beatriz · Miguel Cerne — Redes Neurais 2026.1
- Visual: 1 gráfico de série com um pico destacado (teaser).

## Slide 2 — O problema (0:20–1:00)
- Anomalia financeira = janela que **destoa do padrão normal** (crash, fraude, choque).
- Rótulos não existem a priori → problema **não supervisionado**.
- Ideia central: aprender a "normalidade" e **medir o quanto algo desvia dela**.
- Pergunta que guia tudo: *o que é "normal" numa série que já teve recessão e Lava Jato?*
- Visual: linha do tempo 2010–2024 com eventos marcados.

## Slide 3 — A ideia em uma frase (1:00–1:30)
- Treinar um **autoencoder** só em período "normal" (2010–2019).
- Ele aprende a **reconstruir** padrões típicos; erro de reconstrução alto = anomalia.
- Um modelo **por ativo** → comparação setorial (energia, mineração, varejo, banco).
- Anomalias detectadas são **cruzadas com eventos brasileiros** documentados.
- Visual: diagrama Encoder → gargalo → Decoder, com seta "erro de reconstrução".

## Slide 4 — Arquitetura (1:30–2:15)
- Entrada: janela de **30 pregões** de log-retornos (`Close`, `+Volume`).
- **Encoder LSTM** → gargalo denso (`latent_dim=16`) → **Decoder LSTM** → reconstrução.
- Perda MAE; um choque vira **erro alto** porque o modelo nunca viu aquilo.
- Rede pequena **de propósito** (~2450 janelas/ativo — dataset pequeno).
- Visual: esquema das camadas com shapes `(30,2)→16→(30,2)`.

## Slide 5 — Metodologia sem vazamento (2:15–3:15)
- **Split temporal ANTES de normalizar** — o scaler vê só o treino (nunca o futuro).
- Treino = "normalidade" 2010–2019; teste = 2020–2024.
- Limiar = **percentil do erro de treino** (p95), não do período avaliado.
- **Walk-forward** (`TimeSeriesSplit`, k=10) para escolher hiperparâmetros — sem holdout ingênuo.
- Régua honesta: uma melhora só conta se **supera o desvio entre folds**.
- Visual: barra temporal treino|teste + folds expansíveis empilhados.

## Slide 6 — Desafios (abertura) (3:15–3:30)
- "A parte interessante não foi montar o autoencoder — foi o que **quebrou** no caminho."
- 4 desafios, 4 decisões medidas (todas viram ADR).
- Visual: 4 ícones (vazamento · choque diluído · idiossincrático vs sistêmico · métrica que engana).

## Slide 7 — Desafio 1+2: erro que some e vazamento (3:30–4:30)
- **Vazamento:** normalizar antes de separar treino/teste puxa o futuro → corrigido com split-antes-do-scaler (por fold no walk-forward).
- **Choque de 1 dia diluído:** média sobre 30 passos apagava o pico → trocamos por **agregação `max`**.
- Resultado medido: `max` **dobrou o Recall** (0,16 → 0,35) e **subiu a Precision** (0,55 → 0,84).
- Visual: antes/depois — série do erro com `mean` (pico sumido) vs `max` (pico nítido).

## Slide 8 — Desafio 3: idiossincrático vs sistêmico (4:30–5:30)
- Uma queda pode ser **do ativo** (fraude) ou **do mercado** (crise global). O erro sozinho não separa.
- Solução: **Conditional Autoencoder** — encoder vê a macro (USD/BRL, VIX), mas a perda recai só em preço/volume.
- Prova de conceito: **COVID/2020 → sistêmico**; **Americanas/2023 → idiossincrático**.
- Números: PETR4 na COVID = 30 janelas sistêmicas; AMER3 = 49 idiossincráticas.
- Visual: tabela COVID×Americanas por ativo (report/figures/m6_covid_contagio.png).

## Slide 9 — Desafio 4: a métrica que engana (5:30–6:30)
- Classe rara: a **ROC-AUC infla** (mar de verdadeiros-negativos). Trocamos por **PR-AUC**.
- No regime raro (prevalência 4,9%): **ROC = 0,84 mascara** o que **PR-AUC = 0,15 expõe**.
- Descoberta de brinde: nosso sintético inflava a prevalência a ~70% (janelas sobrepostas) → **avaliação por evento** (agrupa janelas contíguas).
- Custo **assimétrico** FP×FN: escolher limiar é decisão de risco, não só estatística.
- Visual: tabela PR-AUC vs ROC-AUC ao variar a raridade.

## Slide 10 — Tema central (6:30–7:00)
- **Mais capacidade ≠ mais sinal.** Testamos e rejeitamos, com evidência:
  - `latent_dim` insensível em [8, 32]; Atenção/Transformers; Optuna; **weight decay** (ADR-0018).
- Todos ficaram **dentro do ruído inter-fold**. O modelo é robusto e parcimonioso.
- O que importa é *o que* se dá (volume, macro diária, agregação `max`), não *quanto*.
- Visual: gráfico "delta vs desvio inter-fold" — barras minúsculas contra a régua.

## Slide 11 — Resultados (7:00–7:45)
- Detector marca ~10% das janelas no teste real, **sem explosão de falsos positivos**.
- Separação idiossincrático/sistêmico **bate com os eventos conhecidos**.
- Pipeline reprodutível, offline, walk-forward, 32 testes verdes, decisões em 18 ADRs.
- Visual: série de PETR4 2020 com a janela da COVID acesa (report/figures/m5_distribuicao_erro.png).

## Slide 12 — Demonstração ao vivo (7:45–9:15)
- **Sandbox interativo** (Streamlit): mexer em agregação, limiar e injeção **em tempo real**.
- Roteiro da demo em `ROTEIRO.md` (3 gestos: `mean`→`max`, subir percentil, injetar choque).
- Mensagem: os parâmetros de *leitura do erro* mudam ao vivo; os de *capacidade* são decididos por experimento.
- Visual: a própria tela do sandbox (`streamlit run demo/sandbox.py`).

## Slide 13 — Conclusões (9:15–10:00)
- Autoencoder aprende "normalidade" e sinaliza desvios **sem rótulos**.
- Rigor > aparência: anti-vazamento, walk-forward, PR-AUC, avaliação por evento.
- Honestidade metodológica: rejeitamos ideias "modernas" **com prova**, não por omissão.
- Trabalho futuro: backtest financeiro (exige estratégia), mais eventos macro, calibrar sintético.
- Visual: 3 ícones-resumo + QR/link do repositório.

---

### Apêndice (slides de reserva para perguntas)
- A1 — Por que MAE e não MSE? (robustez a outliers)
- A2 — Detalhe anti-vazamento no walk-forward (scaler por fold)
- A3 — Limiar estático vs dinâmico causal (janela de 252 pregões)
- A4 — Tabela completa do experimento de weight decay (ADR-0018)
