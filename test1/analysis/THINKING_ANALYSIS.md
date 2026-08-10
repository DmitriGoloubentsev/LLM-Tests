# Reasoning-trace analysis — `results/ds-v4-flash_temp0/raw.jsonl`

101 points.

## Volume (min / median / max)

| metric | min / median / max |
|---|---|
| reasoning_tokens | 33 / 5,722 / 65,536 |
| completion_tokens | 40 / 5,731 / 65,536 |
| chars | 138 / 13,234 / 156,239 |
| mult_ops | 0 / 121 / 1,363 |
| latency_s | 1 / 35 / 393 |

## Method markers (points where the pattern appears at all)

| method | points | median hits |
|---|---|---|
| recalled_constants | 93/101 | 9 |
| taylor_series | 81/101 | 12 |
| squaring | 66/101 | 2 |
| log_inversion | 22/101 | 2 |
| long_multiplication | 75/101 | 64 |

## Self-checking

- **self_checks**: median 7, max 89, points with 0: 8
- **doubt_markers**: median 11, max 250, points with 0: 15
- **distinct_candidates**: median 134, max 1290, points with 0: 1

## Accuracy vs effort

- correct digits: min 6.18, median 15.00
- corr(reasoning_tokens, correct_digits) = +0.14
- corr(self_checks, correct_digits) = +0.15
- corr(doubt_markers, correct_digits) = +0.15
- corr(mult_ops, correct_digits) = +0.14

## Least accurate points

| x | correct digits | abs err | reasoning tokens | self-checks | doubt |
|---|---|---|---|---|---|
| 0.1 | 6.18 | 7.32e-07 | 128 | 1 | 1 |
| -0.42 | 8.92 | 7.86e-10 | 7,041 | 7 | 7 |
| 0.44 | 10.93 | 1.80e-11 | 952 | 1 | 0 |

Per-point profile: `thinking_profile.csv`

Full trace: `trace_x0.1.md`
Full trace: `trace_x-0.42.md`
Full trace: `trace_x0.44.md`
