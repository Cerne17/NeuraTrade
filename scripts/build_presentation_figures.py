"""Gera os diagramas da apresentação que NÃO saem dos notebooks.

As figuras de resultado vêm dos notebooks (`report/figures/*.png`). Estes três
são esquemáticos/derivados e são desenhados aqui, de forma reprodutível e offline:

1. ``04_arquitetura.png``   — esquema Encoder → gargalo → Decoder.
2. ``05_walkforward.png``   — barra temporal treino|teste + folds expansíveis.
3. ``10_weight_decay.png``  — delta de val_loss vs ruído inter-fold (ADR-0018),
   lido de ``figures/experiment_weight_decay.csv``.

Uso:
    python scripts/build_presentation_figures.py
Saída: ``presentation/figures/``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "presentation" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

PETROL = "#0f6e6e"
GREY = "#8a8a8a"
ACCENT = "#d94f2a"
BLUE = "#2a6fd9"
DPI = 200


# ---------------------------------------------------------------------------
# 1. Arquitetura Encoder -> gargalo -> Decoder
# ---------------------------------------------------------------------------
def fig_arquitetura() -> None:
    fig, ax = plt.subplots(figsize=(11.4, 3.2))
    ax.set_xlim(-0.2, 12.6)
    ax.set_ylim(0, 4)
    ax.axis("off")

    blocks = [
        (0.3, "Entrada\n(30, 2)\nClose+Volume", GREY),
        (2.4, "Encoder\nLSTM (64)", PETROL),
        (4.5, "Gargalo\nDense (16)", ACCENT),
        (6.6, "RepeatVector\n(30)", PETROL),
        (8.7, "Decoder\nLSTM (64)", PETROL),
        (10.6, "Reconstrução\n(30, 2)", GREY),
    ]
    w, h, y = 1.7, 1.6, 1.4
    centers = []
    for x, label, color in blocks:
        box = FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.12",
            linewidth=1.5, edgecolor=color, facecolor=color + "22",
        )
        ax.add_patch(box)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=9)
        centers.append(x + w / 2)

    for i in range(len(blocks) - 1):
        x0 = blocks[i][0] + w
        x1 = blocks[i + 1][0]
        ax.add_patch(FancyArrowPatch(
            (x0, y + h / 2), (x1, y + h / 2),
            arrowstyle="-|>", mutation_scale=14, color="#555", linewidth=1.4,
        ))

    # Seta de erro: entrada vs reconstrução
    ax.add_patch(FancyArrowPatch(
        (centers[0], y), (centers[0], 0.55), arrowstyle="-", color=ACCENT, linewidth=1.2))
    ax.add_patch(FancyArrowPatch(
        (centers[-1], y), (centers[-1], 0.55), arrowstyle="-", color=ACCENT, linewidth=1.2))
    ax.add_patch(FancyArrowPatch(
        (centers[0], 0.55), (centers[-1], 0.55),
        arrowstyle="<|-|>", mutation_scale=13, color=ACCENT, linewidth=1.6))
    ax.text((centers[0] + centers[-1]) / 2, 0.2,
            "erro de reconstrução (MAE)  =  alarme de anomalia",
            ha="center", va="center", fontsize=10, color=ACCENT, fontweight="bold")

    ax.set_title("LSTM-Autoencoder — comprime a normalidade e mede o desvio",
                 fontsize=12, fontweight="bold", color=PETROL)
    fig.tight_layout()
    fig.savefig(OUT / "04_arquitetura.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2. Split temporal + walk-forward
# ---------------------------------------------------------------------------
def fig_walkforward() -> None:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 4.2),
                                   gridspec_kw={"height_ratios": [1, 2.4]})

    # (a) barra treino|teste
    ax1.barh(0, 10, color=PETROL + "cc", edgecolor="none")
    ax1.barh(0, 5, left=10, color=ACCENT + "cc", edgecolor="none")
    ax1.text(5, 0, "TREINO — normalidade\n2010–2019", ha="center", va="center",
             color="white", fontsize=10, fontweight="bold")
    ax1.text(12.5, 0, "TESTE\n2020–2024", ha="center", va="center",
             color="white", fontsize=10, fontweight="bold")
    ax1.set_xlim(0, 15)
    ax1.axis("off")
    ax1.set_title("Split temporal ANTES de normalizar (scaler só no treino → sem vazamento)",
                  fontsize=11, fontweight="bold", color=PETROL)

    # (b) folds walk-forward expansíveis (só sobre o treino)
    ax2.set_xlim(0, 10)
    ax2.set_ylim(-1.0, 5.5)
    ax2.axis("off")
    ax2.text(5, 5.2, "Walk-forward (TimeSeriesSplit, k=10) — seleção de hiperparâmetros",
             ha="center", fontsize=11, fontweight="bold", color=PETROL)
    n = 5
    for i in range(n):
        tr = 2 + i * 1.4
        va = 1.4
        y = n - 1 - i
        ax2.barh(y, tr, color=BLUE + "cc", edgecolor="none", height=0.6)
        ax2.barh(y, va, left=tr, color="#f0a500", edgecolor="none", height=0.6)
        if i == 0:
            ax2.text(tr / 2, y, "treino", ha="center", va="center", color="white", fontsize=8)
            ax2.text(tr + va / 2, y, "val", ha="center", va="center", color="white", fontsize=8)
        ax2.text(-0.1, y, f"fold {i+1}", ha="right", va="center", fontsize=8, color="#555")
    ax2.text(5, -0.8, "régua: uma melhora só conta se for maior que o desvio entre folds",
             ha="center", fontsize=9, style="italic", color=GREY)

    fig.tight_layout()
    fig.savefig(OUT / "05_walkforward.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3. Weight decay: delta vs ruído inter-fold (ADR-0018)
# ---------------------------------------------------------------------------
def fig_weight_decay() -> None:
    csv = ROOT / "figures" / "experiment_weight_decay.csv"
    if not csv.exists():
        print(f"[skip] {csv} ausente — rode scripts/experiment_weight_decay.py antes.")
        return
    df = pd.read_csv(csv)
    tickers = list(df["ticker"].unique())
    cands = sorted(c for c in df["weight_decay"].unique() if c > 0)

    fig, axes = plt.subplots(1, len(tickers), figsize=(12, 3.4), sharey=False)
    for ax, tk in zip(axes, tickers):
        g = df[df["ticker"] == tk]
        base = g.loc[g["weight_decay"] == 0.0, "val_loss_mean"].iloc[0]
        std = g.loc[g["weight_decay"] == 0.0, "val_loss_std"].iloc[0]
        deltas = [g.loc[g["weight_decay"] == c, "val_loss_mean"].iloc[0] - base for c in cands]

        ax.axhspan(-std, std, color=GREY, alpha=0.22, label="ruído inter-fold (±σ)")
        ax.axhline(0, color="#333", linewidth=0.8)
        colors = [ACCENT if d > 0 else PETROL for d in deltas]
        ax.bar([f"{c:g}" for c in cands], deltas, color=colors, width=0.55)
        worst = max(abs(d) for d in deltas)
        ax.text(0.5, 0.93, f"|Δ| máx ≈ {worst:.0e}\nσ ≈ {std:.0e}  →  {worst/std:.0%} do ruído",
                transform=ax.transAxes, ha="center", va="top", fontsize=7, color="#333")
        ax.set_title(tk.replace(".SA", ""), fontsize=10, fontweight="bold")
        ax.set_ylim(-std * 1.15, std * 1.15)
        ax.tick_params(axis="x", labelsize=7)
        ax.tick_params(axis="y", labelsize=6)
        ax.set_xlabel("weight_decay", fontsize=8)
    axes[0].set_ylabel("Δ val_loss vs baseline", fontsize=9)
    fig.suptitle(
        "Weight decay (AdamW): todo ganho cabe dentro do ruído inter-fold → indiferente (ADR-0018)",
        fontsize=12, fontweight="bold", color=PETROL, y=1.04,
    )
    fig.tight_layout()
    fig.savefig(OUT / "10_weight_decay.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    fig_arquitetura()
    fig_walkforward()
    fig_weight_decay()
    print(f"Figuras geradas em {OUT}")
    for p in sorted(OUT.glob("*.png")):
        print(" -", p.relative_to(ROOT))


if __name__ == "__main__":
    main()
