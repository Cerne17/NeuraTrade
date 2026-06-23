"""Inferência interativa: roda os modelos treinados em janelas de tempo à escolha.

Pergunta um intervalo (ex.: 2025-01-01 a 2025-03-31) e um ticker (ou todos), baixa
os dados, aplica os modelos e lista as janelas marcadas como anômalas. Repete até
você sair. Os modelos e o limiar vêm do treino (2010–2019) — só a janela avaliada
muda (ADR-0001).

Uso:
    .venv/bin/python scripts/run_inference.py
"""

from __future__ import annotations

from datetime import date

from src.config import CONFIG
from src.inference import fetch_window, infer_window


def _ask(prompt: str, default: str) -> str:
    resp = input(f"{prompt} [{default}]: ").strip()
    return resp or default


def _run_one(ticker: str, start: str, end: str) -> None:
    try:
        dfs = fetch_window([ticker], start, end)
        res = infer_window(ticker, dfs[ticker])
    except Exception as exc:  # rede, janela curta, modelo ausente
        print(f"  {ticker}: erro — {exc}")
        return

    anomalias = res[res["anomalia"]]
    n, total = len(anomalias), len(res)
    pct = 100 * n / total if total else 0
    print(f"\n  {ticker}: {n}/{total} janelas anômalas ({pct:.1f}%)")
    if n:
        print("  datas anômalas (fim da janela | erro | limiar):")
        for d, row in anomalias.iterrows():
            print(f"    {d.date()}  erro={row['erro']:.4f}  limiar={row['limiar']:.4f}")


def main() -> int:
    tickers = CONFIG["tickers"]
    hoje = date.today().isoformat()
    print("=" * 60)
    print("NeuraTrade · inferência interativa")
    print(f"tickers: {', '.join(tickers)}")
    print(f"agregação do erro: {CONFIG['detection'].get('aggregation')}")
    print("=" * 60)

    while True:
        start = _ask("Início (YYYY-MM-DD)", "2025-01-01")
        end = _ask("Fim (YYYY-MM-DD)", hoje)
        alvo = _ask(f"Ticker ({'/'.join(tickers)}/all)", "all")

        escolhidos = tickers if alvo.lower() == "all" else [alvo.upper()]
        escolhidos = [t for t in escolhidos if t in tickers] or tickers

        print(f"\nBaixando e inferindo {start} → {end} ...")
        for t in escolhidos:
            _run_one(t, start, end)

        if _ask("\nOutra janela? (s/n)", "n").lower() not in ("s", "sim", "y"):
            print("Até logo.")
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
