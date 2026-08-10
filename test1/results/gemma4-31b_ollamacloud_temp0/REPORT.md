# test1 — `exp(x)` accuracy, gemma4:31b (Ollama Cloud, no tools)

`gemma4:31b` · `https://ollama.com/v1` (Ollama Cloud, key from
`agentCodingTest/config/api_glm_5_2_ollama/config.env`) · temperature 0 · 101 points
(x = −1.00…1.00 step 0.02) · one stateless request per point · concurrency 8 · 2026-08-06.

Ran on Ollama Cloud rather than the local daemon because **the GPU is currently unavailable**
(`nvidia-smi`: "couldn't communicate with the NVIDIA driver"; no `/dev/nvidia*`), so a local
31B-dense would have been CPU-only. Ollama Cloud has no `qwen3.6` — its catalogue is
kimi-k2.6/k2.7/k3, mistral-large-3, deepseek-v4-flash:0731, gpt-oss 20b/120b, glm-5.2,
minimax-m2.7, qwen3.5:397b, nemotron-3-ultra, gemma4:31b.

## Headline

| metric | value |
|---|---|
| answered | **101 / 101** |
| exact IEEE-754 double matches | **1 / 101** |
| median correct digits | **10.40** |
| mean correct digits | 10.19 |
| **min correct digits** | **−12.00** (one gross failure; 5.11 excluding it) |
| points ≥12 digits | 31 / 101 |
| points ≥9 digits | 67 / 101 |
| max abs err | 1.79e+12 — the gross failure |
| max abs err excluding it | 2.0e-05 (x = 0.94) |
| wall / tokens | **21.9s** · 1,491 completion tokens total (**0 reasoning**) |
| latency per point | mean 1.67s |

## What it actually does

It emits ~14 tokens per call — a bare 12-digit decimal, no reasoning, no working. It is
**recalling and interpolating a digit string, not computing**. The signature is visible in the
answers: a correct 6–8 digit prefix followed by invented tail digits, often with a repeating
`…115` / `…1115` motif:

```
x=0.94  got 2.56000151115   expected 2.55998141833   (7.9e-06)
x=0.46  got 1.58406716828   expected 1.58407398499   (4.3e-06)
x=0.92  got 2.50928911115   expected 2.50929038994   (5.1e-07)
x=-0.94 got 0.390627581115  expected 0.390627835359  (6.5e-07)
```
34 of 101 points are wrong by more than 1e-9 relative.

### The gross failure: x = 0.58
```
raw reply: 1786038444035        expected: 1.78603843075007
```
**The decimal point is missing.** The digits are roughly right (1.786038444035 — itself only
correct to 8 digits), but as a number the answer is off by a factor of 1e12. A downstream consumer
parsing this reply gets a value 10¹² too large with no error signal. This is the single most
dangerous failure mode in the whole test — worse than a wrong digit, because it is silent and huge.

## Versus DeepSeek-V4-Flash on the identical grid

| | gemma4:31b | DeepSeek-V4-Flash |
|---|---|---|
| answered | 101/101 | 100/101 |
| exact doubles | 1 | 30 |
| median correct digits | 10.40 | **15.00** |
| min correct digits | −12.00 (5.11 excl. gross) | **6.18** |
| points ≥12 digits | 31 | **95** |
| max abs err | 1.79e+12 | **7.32e-07** |
| completion tokens | **1,491** | 979,136 |
| wall (concurrency 8) | **22s** | 916s |
| mean latency/point | **1.7s** | 55.2s |
| reasoning tokens | 0 | 978,311 |

**657× fewer tokens, 42× faster, and ~5 fewer correct digits — plus one catastrophic formatting
failure.** The two models are doing different things: DeepSeek derives the value (argument
splitting + Taylor series, cross-checked), gemma4 recalls it. Recall is free and gets you ~10
digits; derivation costs ~6k tokens per point and gets you the double.

Which is "better" depends entirely on the tolerance you need. For 6-digit engineering work gemma4
is 42× cheaper in wall time and effectively free in tokens. For anything that feeds a numerical
pipeline, neither is safe without validation — but gemma4's failure mode (silent 1e12 offset) is
far harder to catch than DeepSeek's (a missing answer, which is loud).

## Plots
`plots/1_accuracy_vs_x.png` (panel B automatically switches to completion tokens for a
non-reasoning model), `2_digit_histogram.png`, `3_effort_vs_accuracy.png`; `_dark` variants of each.

## Repro
```bash
OLLAMA_API_KEY=<key> ./run_test1.py --base-url https://ollama.com/v1 --model gemma4:31b \
  --api-key-env OLLAMA_API_KEY --concurrency 8 --temperature 0 --tag gemma4-31b_ollamacloud_temp0
../.venv-plot/bin/python plot_results.py results/gemma4-31b_ollamacloud_temp0
```
