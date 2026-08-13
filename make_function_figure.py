#!/usr/bin/env python3
"""make_function_figure.py — the five-function result, in two panels.

Left: every shortcutter scores the same *ordering* across five functions
(sqrt > exp > cos ~ log > sin), on a flat ~18 tokens per point. The band is not a
fixed property of small models — it tracks how well each function yields to a short
local expansion.

Right: the round-vs-jittered recall gap, which shows the ordering is not just
memorisation. If sqrt scored well because it was better *memorised*, jittering the
arguments would cost it what it costs exp. It doesn't.

Usage: .venv-plot/bin/python make_function_figure.py [--dark]
Writes docs/function_ordering[_dark].png
"""
import argparse, json, os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LIGHT = {"surface": "#FCFCFB", "ink": "#22252A", "ink2": "#4A505A", "muted": "#6E7581",
         "grid": "#E3E4E6", "data": "#2F6FD0", "bad": "#C43B3B"}
DARK = {"surface": "#14161A", "ink": "#ECEEF1", "ink2": "#B6BCC5", "muted": "#8A929C",
        "grid": "#2A2E35", "data": "#5B94E8", "bad": "#E2685F"}

# One fixed hue order, assigned per model and never cycled (5 models, 5 slots).
SERIES = ["#2F6FD0", "#C43B3B", "#3F8F5B", "#8A5FC0", "#C1863B"]

FUNCS = ["sqrt", "exp", "cos", "log", "sin"]

MODELS = [                       # label, tag suffix used under test1v2/results/
    ("mistral-large-3 675B", "mistral-large-3-675b_ollamacloud"),
    ("gemma4:31b",           "gemma4-31b"),
    ("qwen3-coder ~30B",     "qwen3-coder-latest_local"),
    ("qwen2.5-coder 14B",    "qwen2.5-coder-14b_local"),
    ("devstral 24B",         "devstral-24b_local"),
]

# Models with a full round-argument ablation (right panel).
ABLATED = [("qwen3-coder ~30B", "qwen3-coder-latest_local"),
           ("devstral 24B",     "devstral-24b_local")]


def med(tag):
    for cand in (tag, tag + "_v2", tag + "_ollamacloud"):
        p = os.path.join("test1v2/results", cand, "summary.json")
        if os.path.exists(p):
            try:
                return json.load(open(p))["median_correct_digits"]
            except Exception:
                pass
    return None


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
        "font.size": 10, "axes.titlesize": 11,
    })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.4, 5.4), width_ratios=[1.25, 1])

    # ---- left: the ordering, one line per model ---------------------------------
    xs = range(len(FUNCS))
    for i, (label, base) in enumerate(MODELS):
        ys = [med(f"{f}_{base}") for f in FUNCS]
        if any(y is None for y in ys):
            continue
        ax1.plot(xs, ys, marker="o", ms=7, lw=2, color=SERIES[i],
                 markeredgecolor=c["surface"], markeredgewidth=1.5, label=label, zorder=3)
        ax1.annotate(f"{ys[0]:.1f}", (0, ys[0]), textcoords="offset points",
                     xytext=(-30, -4), fontsize=9, color=c["ink2"])
    ax1.set_xticks(list(xs))
    ax1.set_xticklabels([f"{f}(x)" for f in FUNCS])
    ax1.set_ylabel("median correct significant digits")
    ax1.set_ylim(1.6, 8.2)
    ax1.set_title("sqrt first in all five models — at a flat ~18 tokens/point",
                  loc="left", color=c["ink"], pad=22)
    ax1.text(0, 1.015, "five models · 14B → 675B · three vendors · 101 points each · "
             "the cos/log/sin tail reshuffles; sqrt does not",
             transform=ax1.transAxes, fontsize=9, color=c["ink2"])
    ax1.legend(frameon=False, fontsize=9, loc="upper right")

    # ---- right: recall gap, round minus jittered --------------------------------
    w = 0.36
    for k, (label, base) in enumerate(ABLATED):
        gaps = []
        for f in FUNCS:
            r, j = med(f"{f}_{base}_round"), med(f"{f}_{base}")
            gaps.append(None if (r is None or j is None) else r - j)
        pos = [i + (k - 0.5) * w for i in xs]
        vals = [0 if g is None else g for g in gaps]
        ax2.bar(pos, vals, width=w, color=SERIES[k * 2], edgecolor=c["surface"],
                linewidth=1.2, label=label, zorder=3)
    ax2.axhline(0, color=c["muted"], lw=1.0, zorder=2)
    ax2.set_xticks(list(xs))
    ax2.set_xticklabels([f"{f}(x)" for f in FUNCS])
    ax2.set_ylabel("digits gained from round arguments")
    ax2.set_title("Memorisation helps on exp — only if the model memorised",
                  loc="left", color=c["ink"], pad=22)
    ax2.text(0, 1.015, "round-argument score minus recall-proof score · "
             "gemma4 (not shown, heaviest recaller) gains +5.83 on exp",
             transform=ax2.transAxes, fontsize=9, color=c["ink2"])
    ax2.legend(frameon=False, fontsize=9, loc="upper right")

    fig.tight_layout()
    out = os.path.join("docs", f"function_ordering{'_dark' if a.dark else ''}.png")
    os.makedirs("docs", exist_ok=True)
    fig.savefig(out, dpi=170, bbox_inches="tight")
    print(out)


if __name__ == "__main__":
    main()
