"""Figuras 3 a 5 de la tesis (BETO y XLM-RoBERTa) a partir de results/*/04_*.

Lee unicamente artefactos versionados (no requiere GPU ni pesos) y escribe PNG a
300 ppp en results/figuras_tesis/. Paleta apta para daltonismo: azul (BETO),
naranja (XLM-R) y escala secuencial de un solo tono para el mapa de calor.

Uso (desde la raiz del repositorio):
    python scripts/05_figuras_tesis.py
"""
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

RESULTS = Path("results")
OUT = RESULTS / "figuras_tesis"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
COLOR = {"BETO": "#2a78d6", "XLM-R": "#eb6834"}
MODELS = ["BETO", "XLM-R"]
SCEN = ["Baseline (0 %)", "Mixto (50 %)", "Full (100 %)"]
RUNS = {
    ("BETO", SCEN[0]): "beto_baseline__augmented_baseline",
    ("BETO", SCEN[1]): "beto_mixed__augmented_50",
    ("BETO", SCEN[2]): "beto_full__augmented_100",
    ("XLM-R", SCEN[0]): "xlmr_baseline__augmented_baseline",
    ("XLM-R", SCEN[1]): "xlmr_mixed__augmented_50",
    ("XLM-R", SCEN[2]): "xlmr_full__augmented_100",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "figure.facecolor": SURF, "axes.facecolor": SURF,
})


def load(run):
    d = RESULTS / run
    return (json.loads((d / "04_eval_summary.json").read_text(encoding="utf-8")),
            json.loads((d / "04_bootstrap.json").read_text(encoding="utf-8")))


def ci_plot(key, fname, ylabel):
    fig, ax = plt.subplots(figsize=(7.2, 3.8), dpi=300)
    lows = []
    for i, model in enumerate(MODELS):
        for j, scen in enumerate(SCEN):
            summ, boot = load(RUNS[(model, scen)])
            x = j + (-0.12 if i == 0 else 0.12)
            v, lo, hi = summ[key]["f1"], boot[key]["ci_low"], boot[key]["ci_high"]
            lows.append(lo)
            ax.vlines(x, lo, hi, color=COLOR[model], lw=2)
            ax.plot(x, v, "o", ms=7, color=COLOR[model], mew=0,
                    label=model if j == 0 else None)
            ax.annotate(f"{v:.4f}", (x, hi), textcoords="offset points",
                        xytext=(0, 6), ha="center", fontsize=8, color=INK2)
    ax.set_xticks(range(3))
    ax.set_xticklabels(SCEN, color=INK)
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(min(lows) - 0.03, 1.0)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, loc="lower left", ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / fname, facecolor=SURF)
    plt.close(fig)


def heatmap(fname):
    cols, mat, order, support = [], [], None, None
    for model in MODELS:
        for scen in SCEN:
            path = RESULTS / RUNS[(model, scen)] / "04_per_entity.csv"
            with open(path, encoding="utf-8-sig") as f:
                rows = {r["label"]: r for r in csv.DictReader(f)}
            if order is None:
                order = sorted(rows, key=lambda lab: -int(rows[lab]["support"]))
                support = {lab: int(rows[lab]["support"]) for lab in order}
            mat.append([float(rows[lab]["f1"]) for lab in order])
            cols.append((model, scen))
    data = [[mat[c][r] for c in range(len(cols))] for r in range(len(order))]
    cmap = LinearSegmentedColormap.from_list(
        "azul", ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#104281", "#0d366b"])
    fig, ax = plt.subplots(figsize=(7.4, 8.6), dpi=300)
    im = ax.imshow(data, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for r, row in enumerate(data):
        for c, v in enumerate(row):
            ax.text(c, r, f"{v:.2f}", ha="center", va="center", fontsize=7.5,
                    color=SURF if v > 0.6 else INK)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([f"{lab.replace('_', ' ').lower()} (n={support[lab]})" for lab in order],
                       fontsize=7.5, color=INK)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([scen.split(" ")[0] for _, scen in cols], fontsize=8, color=INK)
    ax.xaxis.tick_top()
    for i, model in enumerate(MODELS):
        ax.text(1 + 3 * i, -1.6, model, ha="center", va="bottom", fontsize=10,
                fontweight="bold", color=COLOR[model])
    ax.axvline(2.5, color=SURF, lw=4)
    ax.set_xticks([x - 0.5 for x in range(1, len(cols))], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, len(order))], minor=True)
    ax.grid(which="minor", color=SURF, lw=1.5)
    ax.tick_params(which="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("F1 por entidad")
    cb.outline.set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / fname, facecolor=SURF)
    plt.close(fig)


def main():
    missing = [run for run in RUNS.values() if not (RESULTS / run / "04_eval_summary.json").exists()]
    if missing:
        raise SystemExit(f"Faltan corridas en results/: {missing}")
    OUT.mkdir(parents=True, exist_ok=True)
    ci_plot("micro", "figura3_micro_f1.png", "Micro-F1 (IC 95 %)")
    ci_plot("macro", "figura4_macro_f1.png", "Macro-F1 (IC 95 %)")
    heatmap("figura5_f1_por_entidad.png")
    print(f"Figuras escritas en {OUT}/")


if __name__ == "__main__":
    main()
