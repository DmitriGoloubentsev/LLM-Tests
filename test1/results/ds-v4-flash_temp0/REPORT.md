# test1 — `exp(x)` accuracy, DeepSeek-V4-Flash (API, no tools)

`deepseek-v4-flash` · `api.deepseek.com/v1` · temperature 0 · 101 points (x = −1.00…1.00 step 0.02)
· one stateless request per point · concurrency 8 · 2026-08-06.

## Headline

| metric | value |
|---|---|
| answered | **100 / 101** (1 produced no answer) |
| exact IEEE-754 double matches | **30 / 100** |
| ≥15 correct digits | **72 / 100** |
| **min correct digits** | **6.18** |
| mean correct digits | 14.48 |
| max abs err | 7.32e-07 (at x = 0.10) |
| mean abs err | 7.33e-09 |
| RMS abs err | 7.32e-08 |
| max rel err | 6.62e-07 |
| wall / cost | 916s · 979,136 completion tokens (978,311 reasoning) · **$0.28** |
| latency per point | mean 55s, min 0.9s, max 393s |

Accuracy histogram (correct digits): 15+ → 72 · 14 → 11 · 13 → 7 · 12 → 5 · 11 → 2 · 10 → 1 ·
8 → 1 · 6 → 1.

**Read it as: usually double-precision-exact, but the guaranteed accuracy across the range is only
~6 digits.** For a routine that must be trusted pointwise, that is the number that matters.

## The three interesting points

### x = 0.10 — worst answer, and it was the *cheapest* (128 reasoning tokens, 1.8s)
```
exp(0.1)=1.105170185988091... Actually e^0.1 = 1.105170185988091? Let's verify.
e^0.1 = 1 +0.1+0.01/2+0.001/6+... = 1.105170... Known value: 1.105170185988091.
```
It recalled a memorized constant, did a one-line sanity check that only confirmed the leading
digits, and shipped it. True value `1.1051709180756477` — **the recalled string is corrupt from the
7th digit** (`...170185988091` vs `...170918075648`). Fast, confident, wrong.

### x = 0.98 — the only non-answer: ran out of output budget mid-arithmetic
65,536 completion tokens, **all** reasoning, 358s, empty `content`. 156,239 chars of trace that
ends mid-sum:
```
...1.020201340026755600 + 0.000000000000000254 = 1.020201340026755854. Then
+0.000000000000000000635 = 1.020201340026755855? So e^0.02 =
```
It hit the model's output cap while chasing digits past the 18th place and never emitted an answer.
A `max_tokens` guard (`--max-tokens`) converts this from a silent empty reply into a clean failure.

### x = −1.00 — correct but self-truncated (37 reasoning tokens, 1.2s)
Its reasoning states `exp(-1) = 0.36787944117144233` (exact), but the answer field is
`0.367879441171` — it truncated to exactly 12 digits because the prompt said "at least 12
significant digits". Scored 11.92 digits. **Prompt artifact, not an arithmetic error** — worth
fixing in a v2 prompt ("give all digits you are confident in, do not round to 12").

## Effort vs accuracy

| bucket | n | exact (≥15 digits) | mean digits | min digits |
|---|---|---|---|---|
| shortcut (<1k reasoning tokens) | 24 | 14 (58%) | 14.01 | 6.18 |
| derived (≥1k reasoning tokens) | 76 | 58 (76%) | 14.63 | 8.92 |

Correlation between reasoning tokens and correct digits is only **+0.14** — because 72% of points
sit at the 15-digit ceiling, so the coefficient is dominated by ties. The bucket split is the
honest statement: **long derivations are more reliable, but they are not a guarantee** (x = −0.42
spent 7,041 tokens and still landed at 8.92 digits), and short recalls are usually right (14 of 24
were exact).

## Method profile across all 101 traces

| method marker | points using it | median hits |
|---|---|---|
| recalled constants (`e^0.3 = …`) | 93/101 | 9 |
| Taylor series | 81/101 | 12 |
| long multiplication | 75/101 | 64 |
| squaring / halving | 66/101 | 2 |
| log inversion | 22/101 | 2 |

Volume per point: reasoning tokens 33 / 5,722 / 65,536 (min/median/max), trace chars
138 / 13,234 / 156,239, latency 1 / 35 / 393s. Self-check phrases: median 7 per trace (max 89);
doubt markers ("wait", "actually", "recompute"): median 11 (max 250).

The dominant pattern (detailed in [`../../analysis/THINKING_x0.34.md`](../../analysis/THINKING_x0.34.md))
is **solve twice, then reconcile**: split the argument and multiply recalled constants by hand, then
re-derive the same value from the Taylor series, and only answer when both agree.

## Plots

`plots/` (light + `_dark` variants of each, regenerate with
`../.venv-plot/bin/python plot_results.py results/<tag> [--dark]`):

| file | what it shows |
|---|---|
| `1_accuracy_vs_x.png` | correct digits per grid point vs the 15-digit double-precision line, with the 3 worst points and the x=0.98 non-answer called out; a second panel below shares the x axis and shows the reasoning tokens spent on each point against the 65,536 output cap |
| `2_digit_histogram.png` | distribution of correct digits — 72 exact, a 10-point tail below 13 |
| `3_effort_vs_accuracy.png` | reasoning tokens (log) vs correct digits, with the <1k / ≥1k exactness split |

Panel B of figure 1 is a separate axes, not a second y-scale: it lines up effort with accuracy
without a dual-axis chart. Palette (`#2F6FD0` data, `#C43B3B` outlier/failure) passes the
dataviz validator on all six checks (CVD ΔE 23.1 protan, normal-vision ΔE 30.1); gray is
annotation ink only, and every red mark also carries a text label so identity is never colour-alone.

## Files
`raw.jsonl` (per-point reply + full reasoning + usage) · `results.csv` · `summary.json` ·
`../../analysis/THINKING_ANALYSIS.md` + `thinking_profile.csv` + `trace_x*.md`.

## Suggested next runs
- `--max-tokens 4096` — bound the x = 0.98 failure mode, see what it costs in digits.
- Prompt v2 that forbids self-truncation (fixes x = −1.00's 11.92).
- Same grid against a local server: `--base-url http://127.0.0.1:8090/v1 --model qwen3.6 --api-key dummy`.
