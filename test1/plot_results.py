#!/usr/bin/env python3
"""plot_results.py — figures for a test1 run.

Usage: plot_results.py results/<tag> [--dark]
Writes PNGs into <results dir>/plots/.

Palette (validated with the dataviz validator, light surface, categorical, 2 slots — all checks
pass: lightness band, chroma floor, CVD separation dE 23.1 protan, normal-vision dE 30.1, contrast):
  data   #2F6FD0     outlier/failure  #C43B3B
Gray #6E7581 is annotation ink only (grid, reference lines), never a series color.
"""
import argparse, csv, json, math, os, sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator

LIGHT = {"surface": "#FCFCFB", "ink": "#22252A", "ink2": "#4A505A", "muted": "#6E7581",
         "grid": "#E3E4E6", "data": "#2F6FD0", "bad": "#C43B3B"}
DARK = {"surface": "#14161A", "ink": "#ECEEF1", "ink2": "#B6BCC5", "muted": "#8A929C",
        "grid": "#2A2E35", "data": "#5B94E8", "bad": "#E2685F"}

CEILING = 15.0          # cap used by correct_digits(): IEEE-754 double agreement
TOKEN_CAP = 65536       # model's max completion tokens — the x=0.98 failure mode


def style(c):
    plt.rcParams.update({
        "figure.facecolor": c["surface"], "axes.facecolor": c["surface"],
        "savefig.facecolor": c["surface"],
        "axes.edgecolor": c["grid"], "axes.labelcolor": c["ink2"],
        "text.color": c["ink"], "xtick.color": c["ink2"], "ytick.color": c["ink2"],
        "grid.color": c["grid"], "grid.linewidth": 0.8,
        "axes.grid": True, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
        "figure.dpi": 140,
    })


def load(d):
    rows = []
    with open(os.path.join(d, "results.csv")) as f:
        for r in csv.DictReader(f):
            rows.append({
                "x": float(r["x"]),
                "digits": float(r["correct_digits"]) if r["correct_digits"] else None,
                "abs_err": float(r["abs_err"]) if r["abs_err"] else None,
                "rtok": int(r["reasoning_tokens"] or 0),
                "ctok": int(r["completion_tokens"] or 0),
                "lat": float(r["latency_s"]) if r["latency_s"] else None,
                "got": r["got"],
            })
    summary = json.load(open(os.path.join(d, "summary.json")))
    return sorted(rows, key=lambda r: r["x"]), summary


GROSS = 0.0   # below 0 correct digits = the answer is off by more than the value itself


def fig_accuracy(rows, s, c, out):
    """Panel A: accuracy per point. Panel B: effort per point. Shared x — two panels, never
    two y-scales on one axes."""
    ok = [r for r in rows if r["digits"] is not None and r["digits"] >= GROSS]
    gross = [r for r in rows if r["digits"] is not None and r["digits"] < GROSS]
    bad = [r for r in rows if r["digits"] is None]
    worst = sorted(ok, key=lambda r: r["digits"])[:3]
    worst_x = {r["x"] for r in worst}
    floor_y = 5.6 if not (ok and min(r["digits"] for r in ok) < 6) else \
        max(0.5, min(r["digits"] for r in ok) - 1.6)

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 7.2), sharex=True,
                                 gridspec_kw={"height_ratios": [1.35, 1], "hspace": 0.13})

    a1.axhline(CEILING, color=c["muted"], lw=1.2, ls=(0, (5, 4)), zorder=1)
    a1.text(1.02, CEILING, "double-precision\nagreement (15)", color=c["muted"],
            fontsize=8.5, va="center", ha="left", transform=a1.get_yaxis_transform())
    normal = [r for r in ok if r["x"] not in worst_x]
    a1.scatter([r["x"] for r in normal], [r["digits"] for r in normal], s=42,
               color=c["data"], edgecolor=c["surface"], linewidth=1.6, zorder=3)
    a1.scatter([r["x"] for r in worst], [r["digits"] for r in worst], s=70,
               color=c["bad"], edgecolor=c["surface"], linewidth=1.6, zorder=4)
    for k, r in enumerate(worst):
        # label below the point when it sits high enough to collide with the cloud above,
        # and stagger successive labels so neighbouring worst points don't overprint
        below = r["digits"] > 10
        dy = (-20 - 13 * (k % 2)) if below else (14 + 13 * (k % 2))
        a1.annotate(f"x={r['x']:g} · {r['digits']:.1f} digits", (r["x"], r["digits"]),
                    textcoords="offset points", xytext=(0, dy),
                    ha="center", va="top" if below else "bottom",
                    fontsize=8.5, color=c["ink2"])
    if bad:
        a1.scatter([r["x"] for r in bad], [floor_y] * len(bad), s=90, marker="X",
                   color=c["bad"], zorder=4)
        if len(bad) <= 3:                      # few enough to name individually
            for k, r in enumerate(sorted(bad, key=lambda r: r["x"])):
                # alternate side + height so neighbouring failures don't overprint
                right = k % 2 == 0
                a1.annotate(f"x={r['x']:g} · no answer", (r["x"], floor_y),
                            textcoords="offset points",
                            xytext=(12 if right else -12, 4 + 12 * (k % 2)),
                            ha="left" if right else "right",
                            va="bottom", fontsize=8.5, color=c["bad"])
        else:                                  # one aggregated label instead of 47 collisions
            a1.annotate(f"{len(bad)} points returned no answer "
                        f"(ran out of output budget mid-arithmetic)",
                        (sum(r["x"] for r in bad) / len(bad), floor_y),
                        textcoords="offset points", xytext=(0, 10), ha="center", va="bottom",
                        fontsize=9, color=c["bad"])
    for r in gross:
        a1.scatter([r["x"]], [floor_y], s=90, marker="X", color=c["bad"], zorder=4)
        a1.annotate(f"x={r['x']:g} · off scale ({r['digits']:.0f} digits)\ngot {r['got']}",
                    (r["x"], floor_y), textcoords="offset points", xytext=(0, -8), ha="center",
                    va="top", fontsize=8.5, color=c["bad"])
    a1.set_ylim(floor_y - (2.2 if gross else 1.0), 16.6)
    a1.set_ylabel("correct significant digits")
    fn = s["meta"].get("func", "exp")
    a1.set_title(f"{fn}(x) accuracy per point — {s['meta']['model']}, "
                 f"{len(rows)} independent API calls",
                 loc="left", color=c["ink"], pad=26)
    # A run where nothing parsed leaves these None — report the shape of the failure instead.
    acc = (f"worst {s['min_correct_digits']:.2f} digits · max abs err {s['max_abs_err']:.1e}"
           if s["min_correct_digits"] is not None else "no point produced a parseable answer")
    a1.text(0, 1.015, f"{s['answered']}/{s['points']} answered · {s['exact_float_matches']} exact doubles · " + acc,
            transform=a1.transAxes, fontsize=9, color=c["ink2"])

    # A non-reasoning model reports 0 reasoning tokens for every point — plot what it did
    # produce (completion tokens) instead of a flat line of zeros.
    thinks = any(r["rtok"] > 0 for r in rows)
    tok = (lambda r: r["rtok"]) if thinks else (lambda r: r["ctok"])
    base = 3 if not thinks else 30
    if thinks:
        a2.axhline(TOKEN_CAP, color=c["muted"], lw=1.2, ls=(0, (5, 4)), zorder=1)
        a2.text(1.02, TOKEN_CAP, "output cap\n(65,536)", color=c["muted"], fontsize=8.5,
                va="center", ha="left", transform=a2.get_yaxis_transform())
    gross_x = {r["x"] for r in gross}
    for r in rows:
        col = c["bad"] if (r["digits"] is None or r["x"] in worst_x or r["x"] in gross_x) else c["data"]
        a2.plot([r["x"], r["x"]], [base, max(tok(r), base)], color=col, lw=1.6, alpha=0.55, zorder=2)
        a2.scatter([r["x"]], [max(tok(r), base)], s=30, color=col,
                   edgecolor=c["surface"], linewidth=1.2, zorder=3)
    a2.set_yscale("log")
    a2.yaxis.set_major_locator(LogLocator(base=10))
    a2.set_ylim(base * 0.8, 120000 if thinks else max(tok(r) for r in rows) * 3)
    a2.set_ylabel("reasoning tokens" if thinks else "completion tokens")
    a2.set_xlabel("x")
    a2.set_title("thinking spent on the same point" if thinks else
                 "output tokens per point (this model emits no reasoning tokens)",
                 loc="left", color=c["ink2"], fontsize=10.5)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_histogram(rows, s, c, out):
    ok = [r for r in rows if r["digits"] is not None]
    lo_edge = 6
    counts = [0] * (16 - lo_edge)          # buckets: "<6", 6..14, "15 (exact)"
    under = 0
    for r in ok:
        b = min(int(r["digits"]), 15)
        if b < lo_edge:
            under += 1
        else:
            counts[b - lo_edge] += 1
    labels = [f"{b}" for b in range(lo_edge, 16)]
    labels[-1] = "15 (exact)"
    if under:
        counts = [under] + counts
        labels = [f"<{lo_edge}"] + labels
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    def bucket_val(l):
        return -1 if l.startswith("<") else int(l.split()[0])
    colors = [c["bad"] if bucket_val(l) < 13 else c["data"] for l in labels]
    bars = ax.bar(labels, counts, width=0.68, color=colors, zorder=3)
    for b, n in zip(bars, counts):
        if n:
            ax.annotate(str(n), (b.get_x() + b.get_width() / 2, n), textcoords="offset points",
                        xytext=(0, 4), ha="center", fontsize=9, color=c["ink2"])
    ax.set_xlabel("correct significant digits (floor)")
    ax.set_ylabel("points")
    ax.set_title("Most answers are exact; the tail is what you can rely on", loc="left",
                 color=c["ink"], pad=24)
    ax.text(0, 1.015, f"red = below 13 digits ({sum(n for l, n in zip(labels, counts) if bucket_val(l) < 13)} points)",
            transform=ax.transAxes, fontsize=9, color=c["ink2"])
    ax.grid(axis="x", visible=False)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_effort(rows, s, c, out):
    ok = [r for r in rows if r["digits"] is not None and r["digits"] >= GROSS and r["ctok"] > 0]
    worst = sorted(ok, key=lambda r: r["digits"])[:3]
    worst_x = {r["x"] for r in worst}
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    normal = [r for r in ok if r["x"] not in worst_x]
    tok = lambda r: max(r["rtok"] or r["ctok"], 1)
    ax.scatter([tok(r) for r in normal], [r["digits"] for r in normal], s=44,
               color=c["data"], edgecolor=c["surface"], linewidth=1.5, alpha=0.9, zorder=3)
    ax.scatter([tok(r) for r in worst], [r["digits"] for r in worst], s=80,
               color=c["bad"], edgecolor=c["surface"], linewidth=1.5, zorder=4)
    for r in worst:
        ax.annotate(f"x={r['x']:g}", (tok(r), r["digits"]), textcoords="offset points",
                    xytext=(9, -3), fontsize=9, color=c["ink2"])
    ax.axhline(CEILING, color=c["muted"], lw=1.2, ls=(0, (5, 4)), zorder=1)
    ax.set_xscale("log")
    ax.set_xlabel("reasoning tokens (log)")
    ax.set_ylabel("correct significant digits")
    ax.set_ylim(min(5.5, min(r["digits"] for r in ok) - 0.8), 16.4)
    ax.set_title("More thinking is not a guarantee", loc="left", color=c["ink"], pad=24)
    lo = [r for r in ok if tok(r) < 1000]
    hi = [r for r in ok if tok(r) >= 1000]
    def frac(rs): return 100 * sum(1 for r in rs if r["digits"] >= CEILING) / len(rs) if rs else 0.0
    ax.text(0, 1.015, f"<1k tokens: {frac(lo):.0f}% exact (n={len(lo)})   ·   "
                     f"≥1k tokens: {frac(hi):.0f}% exact (n={len(hi)})",
            transform=ax.transAxes, fontsize=9, color=c["ink2"])
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir")
    ap.add_argument("--dark", action="store_true")
    a = ap.parse_args()
    c = DARK if a.dark else LIGHT
    style(c)
    rows, s = load(a.results_dir)
    outdir = os.path.join(a.results_dir, "plots")
    os.makedirs(outdir, exist_ok=True)
    sfx = "_dark" if a.dark else ""
    made = [
        fig_accuracy(rows, s, c, os.path.join(outdir, f"1_accuracy_vs_x{sfx}.png")),
        fig_histogram(rows, s, c, os.path.join(outdir, f"2_digit_histogram{sfx}.png")),
    ]
    # Figures 1 and 2 still say something when every point failed (the failure markers and an
    # empty histogram are the result). Figure 3 plots digits against tokens, so with no answered
    # point there is nothing to place — skip it rather than emit an empty axes.
    if any(r["digits"] is not None for r in rows):
        made.append(fig_effort(rows, s, c, os.path.join(outdir, f"3_effort_vs_accuracy{sfx}.png")))
    else:
        made.append(f"(skipped 3_effort_vs_accuracy{sfx}.png: no point produced an answer)")
    for m in made:
        print(m)


if __name__ == "__main__":
    main()
