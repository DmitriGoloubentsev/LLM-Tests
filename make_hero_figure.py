#!/usr/bin/env python3
"""make_hero_figure.py — the four-panel summary figure used in the README.

One panel per configuration, same axes throughout: correct significant digits against the grid
point. Read left-to-right it is the whole study — a model that recalls, the same model when recall
is unavailable, a model that derives, and what happens when you throttle its thinking.

Usage: .venv-plot/bin/python make_hero_figure.py [--dark]
Writes docs/hero_recall_ablation[_dark].png
"""
import argparse, csv, json, os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))

# Palette validated with the dataviz validator (light surface, categorical, 2 slots):
# lightness band, chroma floor, CVD separation dE 23.1 (protan), normal-vision dE 30.1, contrast.
LIGHT = {"surface": "#FCFCFB", "ink": "#22252A", "ink2": "#4A505A", "muted": "#6E7581",
         "grid": "#E3E4E6", "data": "#2F6FD0", "bad": "#C43B3B"}
DARK = {"surface": "#14161A", "ink": "#ECEEF1", "ink2": "#B6BCC5", "muted": "#8A929C",
        "grid": "#2A2E35", "data": "#5B94E8", "bad": "#E2685F"}

PANELS = [
    ("test1/results/gemma4-31b_ollamacloud_temp0",
     "gemma4:31b — round arguments", "x = -1.00, -0.98, … it can recall these"),
    ("test1v2/results/exp_gemma4-31b_v2",
     "gemma4:31b — recall-proof arguments", "x = -0.9513299347, … it cannot"),
    ("test1v2/results/exp_ds-v4-flash_v2",
     "DeepSeek-V4-Flash — recall-proof", "derives every value; unbounded thinking"),
    ("test1v2/results/exp_ds-v4-flash_v2_efflow",
     "DeepSeek-V4-Flash — recall-proof", "same, with reasoning_effort = low"),
]


def load(d):
    rows = []
    with open(os.path.join(HERE, d, "results.csv")) as f:
        for r in csv.DictReader(f):
            rows.append({"x": float(r["x"]),
                         "digits": float(r["correct_digits"]) if r["correct_digits"] else None})
    s = json.load(open(os.path.join(HERE, d, "summary.json")))
    return rows, s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dark", action="store_true")
    a = ap.parse_args()
    c = DARK if a.dark else LIGHT
    plt.rcParams.update({
        "figure.facecolor": c["surface"], "axes.facecolor": c["surface"],
        "savefig.facecolor": c["surface"], "axes.edgecolor": c["grid"],
        "axes.labelcolor": c["ink2"], "text.color": c["ink"],
        "xtick.color": c["ink2"], "ytick.color": c["ink2"],
        "grid.color": c["grid"], "grid.linewidth": 0.8, "axes.grid": True,
        "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 9.5, "figure.dpi": 150,
    })

    fig, axes = plt.subplots(2, 2, figsize=(13, 7.6), sharex=True, sharey=True)
    fig.subplots_adjust(hspace=0.52, wspace=0.09, top=0.82)

    for ax, (d, title, sub) in zip(axes.ravel(), PANELS):
        rows, s = load(d)
        ok = [r for r in rows if r["digits"] is not None and r["digits"] >= 0]
        gross = [r for r in rows if r["digits"] is not None and r["digits"] < 0]
        bad = [r for r in rows if r["digits"] is None]

        ax.axhline(15, color=c["muted"], lw=1.1, ls=(0, (5, 4)), zorder=1)
        ax.scatter([r["x"] for r in ok], [r["digits"] for r in ok], s=26, color=c["data"],
                   edgecolor=c["surface"], linewidth=1.0, zorder=3)
        for r in gross + bad:
            ax.scatter([r["x"]], [1.2], s=60, marker="X", color=c["bad"], zorder=4)

        med = s.get("median_correct_digits")
        if med is None:                                   # test1 summaries predate the field
            v = sorted(r["digits"] for r in rows if r["digits"] is not None)  # incl. off-scale
            med = v[len(v) // 2] if v else float("nan")
        note = f"median {med:.1f} digits · {s['answered']}/{s['points']} answered"
        if bad:
            note += f" · {len(bad)} no answer"
        if gross:
            note += f" · {len(gross)} off-scale"
        ax.set_title(title, loc="left", color=c["ink"], fontsize=11, fontweight="bold", pad=38)
        ax.text(0, 1.135, sub, transform=ax.transAxes, fontsize=9, color=c["ink2"])
        ax.text(0, 1.035, note, transform=ax.transAxes, fontsize=9, color=c["ink2"])
        ax.set_ylim(0, 16.6)
        ax.set_xlim(-1.12, 1.12)

    for ax in axes[1]:
        ax.set_xlabel("x")
    for ax in axes[:, 0]:
        ax.set_ylabel("correct significant digits")

    fig.text(0.5, 0.955, "Same function, same 101 points, same prompt — exp(x) to 15 digits, "
                         "one stateless API call each",
             ha="center", fontsize=13, fontweight="bold", color=c["ink"])
    fig.text(0.5, 0.918, "dashed line = agreement with an IEEE-754 double  ·  "
                         "red ✕ = no usable answer returned",
             ha="center", fontsize=9.5, color=c["ink2"])

    out = os.path.join(HERE, "docs",
                       f"hero_recall_ablation{'_dark' if a.dark else ''}.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(out)


if __name__ == "__main__":
    main()
