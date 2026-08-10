#!/usr/bin/env python3
"""analyze_thinking.py — what does the model actually DO inside the reasoning block?

Reads a test1 `raw.jsonl` and profiles each point's reasoning trace: length, which numeric
method it used, how often it re-checked itself, and whether any of that correlates with the
accuracy of the final answer.

Usage: analysis/analyze_thinking.py results/<tag>/raw.jsonl [--dump-worst N] [--out <md>]
"""
import argparse, json, math, os, re, statistics, sys

METHODS = {
    # split x into parts whose exponentials the model claims to know by heart
    "recalled_constants": r"e\^-?0?\.\d+\s*=|e\^\(-?\d+/\d+\)",
    "taylor_series":      r"\bseries\b|\bTaylor\b|/\s*n!|\bfactorial\b|\bt\d+\s*=",
    "squaring":           r"\bsquar(e|ing)\b|\^2\s*=|half of|\bsqrt\b",
    "log_inversion":      r"\bln\(|\blog\b",
    "long_multiplication": r"Break \d|Compute .*\*|\*\s*0\.0+\d",
}
SELF_CHECK = r"\bverify\b|\bcheck\b|\bcross-?check\b|let'?s confirm"
DOUBT      = r"\bwait\b|\bhmm+\b|\bactually\b|\brecompute\b|\bmistake\b|\blet me redo\b|\boff by\b"
CANDIDATE  = r"1\.\d{10,}|0\.\d{10,}|\d\.\d{10,}"


def digits(expected, got):
    if got is None or expected == 0:
        return None
    rel = abs(got - expected) / abs(expected)
    return 15.0 if rel == 0 else min(15.0, -math.log10(rel))


def profile(rec):
    r = rec.get("reasoning") or ""
    p = {"x": rec["x"], "chars": len(r),
         "reasoning_tokens": rec.get("reasoning_tokens", 0),
         "completion_tokens": rec.get("completion_tokens", 0),
         "latency_s": rec.get("latency_s"),
         "digits": digits(rec["expected"], rec.get("got")),
         "abs_err": None if rec.get("got") is None else abs(rec["got"] - rec["expected"]),
         "self_checks": len(re.findall(SELF_CHECK, r, re.I)),
         "doubt_markers": len(re.findall(DOUBT, r, re.I)),
         "distinct_candidates": len(set(re.findall(CANDIDATE, r))),
         "mult_ops": r.count("*")}
    for name, pat in METHODS.items():
        p[name] = len(re.findall(pat, r, re.I))
    return p


def corr(a, b):
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    if len(pairs) < 3:
        return None
    xs, ys = zip(*pairs)
    if len(set(xs)) < 2 or len(set(ys)) < 2:
        return None
    return statistics.correlation(xs, ys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw")
    ap.add_argument("--dump-worst", type=int, default=3, help="save full traces of the N least accurate points")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    recs = [json.loads(l) for l in open(a.raw) if l.strip()]
    profs = [profile(r) for r in recs]
    outdir = os.path.dirname(os.path.abspath(a.out or a.raw))
    md = a.out or os.path.join(os.path.dirname(os.path.abspath(a.raw)), "THINKING_ANALYSIS.md")

    def stat(key):
        v = [p[key] for p in profs if p[key] is not None]
        if not v:
            return "-"
        return f"{min(v):,.0f} / {statistics.median(v):,.0f} / {max(v):,.0f}"

    lines = []
    add = lines.append
    add(f"# Reasoning-trace analysis — `{a.raw}`\n")
    add(f"{len(recs)} points.\n")
    add("## Volume (min / median / max)\n")
    add("| metric | min / median / max |")
    add("|---|---|")
    for k in ("reasoning_tokens", "completion_tokens", "chars", "mult_ops", "latency_s"):
        add(f"| {k} | {stat(k)} |")
    add("")
    add("## Method markers (points where the pattern appears at all)\n")
    add("| method | points | median hits |")
    add("|---|---|---|")
    for name in METHODS:
        used = [p[name] for p in profs if p[name] > 0]
        add(f"| {name} | {len(used)}/{len(profs)} | {statistics.median(used) if used else 0:.0f} |")
    add("")
    add("## Self-checking\n")
    for k in ("self_checks", "doubt_markers", "distinct_candidates"):
        v = [p[k] for p in profs]
        add(f"- **{k}**: median {statistics.median(v):.0f}, max {max(v)}, "
            f"points with 0: {sum(1 for x in v if x == 0)}")
    add("")
    add("## Accuracy vs effort\n")
    dg = [p["digits"] for p in profs]
    add(f"- correct digits: min {min(x for x in dg if x is not None):.2f}, "
        f"median {statistics.median([x for x in dg if x is not None]):.2f}")
    for k in ("reasoning_tokens", "self_checks", "doubt_markers", "mult_ops"):
        c = corr([p[k] for p in profs], dg)
        add(f"- corr({k}, correct_digits) = {'n/a' if c is None else f'{c:+.2f}'}")
    add("")
    worst = sorted([p for p in profs if p["digits"] is not None], key=lambda p: p["digits"])[:a.dump_worst]
    add("## Least accurate points\n")
    add("| x | correct digits | abs err | reasoning tokens | self-checks | doubt |")
    add("|---|---|---|---|---|---|")
    for p in worst:
        add(f"| {p['x']} | {p['digits']:.2f} | {p['abs_err']:.2e} | {p['reasoning_tokens']:,} | "
            f"{p['self_checks']} | {p['doubt_markers']} |")
    add("")
    # per-point CSV for spreadsheets
    csv = os.path.join(os.path.dirname(md), "thinking_profile.csv")
    cols = ["x", "digits", "abs_err", "reasoning_tokens", "completion_tokens", "chars", "latency_s",
            "self_checks", "doubt_markers", "distinct_candidates", "mult_ops"] + list(METHODS)
    with open(csv, "w") as f:
        f.write(",".join(cols) + "\n")
        for p in profs:
            f.write(",".join("" if p[c] is None else str(p[c]) for c in cols) + "\n")
    add(f"Per-point profile: `{os.path.basename(csv)}`\n")

    # dump the worst traces verbatim for human review
    by_x = {r["x"]: r for r in recs}
    for p in worst:
        r = by_x[p["x"]]
        fn = os.path.join(os.path.dirname(md), f"trace_x{p['x']}.md")
        with open(fn, "w") as f:
            f.write(f"# Trace x={p['x']} — {p['digits']:.2f} correct digits\n\n"
                    f"expected `{r['expected']!r}` · got `{r['got']!r}` · "
                    f"{r.get('reasoning_tokens',0):,} reasoning tokens\n\n"
                    f"## Answer\n\n```\n{r.get('raw')}\n```\n\n## Reasoning\n\n```text\n"
                    f"{r.get('reasoning') or '(none returned)'}\n```\n")
        add(f"Full trace: `{os.path.basename(fn)}`")

    open(md, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n-> {md}")


if __name__ == "__main__":
    main()
