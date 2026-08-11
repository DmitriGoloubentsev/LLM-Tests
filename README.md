# LLM-Tests — can a model *follow a derivation*, or does it just reach for a tool?

Ask any assistant for `exp(-0.9513299347)` to 15 digits and you will get a perfect answer. It will
write three lines of Python and run them. That measures the sandbox, not the model.

**This is that exam with the calculator taken away.** No tools are available — not restricted, not
discouraged, *absent* — so the model has to do what a student does on a closed-book paper: reduce
the argument onto a constant it remembers, expand a Taylor series, and work through a dozen
16-digit multiplications by hand. Thousands of tokens of arithmetic in which one slipped carry
destroys the result. There is no way to bluff it. The answer is either right to fifteen places or
it isn't, and the grader can tell to a fraction of a digit.

That is the whole point: **we are not testing whether a model can get the right number — we are
testing whether it can follow a long derivation without losing the thread.** With a tool, every
model scores 100%. Without one, the spread is enormous.

So this harness removes every escape hatch: **one stateless API call per point — no tools, no code
execution, no conversation, no retries** — and scores the reply by how many significant digits
survive against IEEE-754 double precision.

The metric is `correct digits = -log10(relative error)`, capped at 15. Pass/fail hides everything
interesting here; "4.8 digits" and "15 digits" are both "wrong" to a boolean grader.

Why this task and not a benchmark of word problems:

- **Unfakeable.** A confident paragraph earns nothing; only correct digits score.
- **Gradable to a fraction of a digit,** against a reference that is correct by construction.
- **Recall-proof on demand.** The arguments are jittered to ten decimal places by a seeded hash, so
  they cannot be in any training corpus — the model must derive or guess (§ *the two tests*).
- **It separates two things every other eval conflates:** knowing the method and executing it
  without error over thousands of tokens.

**How "no calculator" is enforced.** Over a plain API this is free — no `tools` array is sent, so
there is nothing to call. The risky path is the Claude Code CLI backend, where the model normally
*has* an interpreter: there it is launched with `--tools ""` (the entire built-in tool set
disabled) plus a throwaway `$HOME` and empty working directory, so no `CLAUDE.md` reaches the
context either. Every CLI reply is verified to have `num_turns == 1` — a tool call would require a
second turn, so a single turn is proof the answer came out of the model, not out of `python`.

What comes out is a clean split. Some models spend ~24,000 tokens per point and land exactly on the
double-precision line. Others answer the identical question in **16 tokens** and land 3 digits in —
they recognise `exp(x)`, emit a remembered-looking number, and stop. Across 70 runs and 30+ models,
almost nothing sits in between, and **which side a model falls on is not predicted by its size, its
vendor, or its price.**

Stdlib Python only. Works against any OpenAI-compatible endpoint — hosted or a local
llama.cpp/SGLang/vLLM/Ollama server.

![Four configurations of the same 101-point exp(x) grid](docs/hero_recall_ablation.png)

Four runs of the same function, the same 101 points and the same prompt. Top row: one model on
arguments it can recall, then on arguments it cannot — the same model loses six digits. Bottom row:
a reasoning model that derives every value instead, and what happens when you throttle its
thinking. Regenerate with `.venv-plot/bin/python make_hero_figure.py [--dark]`.

## The two tests

| | grid | question it answers |
|---|---|---|
| **test1** | `x = -1.00 … 1.00` step `0.02` (101 round arguments) | how accurate is the model on values it may have memorised? |
| **test1v2** | the same 101 points, each nudged by a deterministic ~10-decimal offset (`-0.9513299347`, …) | how accurate is it when the argument **cannot** be in any training corpus? |

test1v2 also supports `--func exp|sin|cos|log|sqrt`.

## Headline results (2026-08, temperature 0, 101 points per cell)

**Round arguments (test1, `exp`):**

| | gemma4:31b | DeepSeek-V4-Flash | GLM-5.2 |
|---|---|---|---|
| median correct digits | 10.40 | 15.00 | 15.00 |
| worst point | **−12.0** | 6.18 | 8.44 |
| exact doubles | 1 | 30 | 34 |
| unanswered | 0 | 1 | 2 |
| completion tokens | 1,491 | 979,136 | 1,926,343 |
| wall @ concurrency 8 | 22s | 916s | 4,288s |

gemma4's worst point is not a rounding error: asked for `exp(0.58)` it replied `1786038444035` —
**the decimal point is missing**, so the answer is off by a factor of 10¹² while looking perfectly
well-formed.

![gemma4:31b on round arguments](test1/results/gemma4-31b_ollamacloud_temp0/plots/1_accuracy_vs_x.png)

The lower panel of every per-run figure shows the tokens spent on each point. gemma4 emits ~14 per
answer and no reasoning at all — it is recalling, not computing, and the accuracy scatter reflects
that.

**Recall-proof arguments (test1v2)** — 70 runs, 30+ models, 7 endpoints. Sorted by median accuracy,
the models fall into four groups that behave qualitatively differently. The `tok/pt` column is the
one to read alongside the median: it says whether the model *derived* the answer or *recalled* it.

**1 — Derivers.** Thousands of tokens per point, and they land on the double-precision line.

| model | fn | answered | median | worst | tok/pt | cost |
|---|---|---|---|---|---|---|
| openai/gpt-5.6-sol | exp | 5/5 | **15.00** | 14.17 | 4,254 | $0.64 |
| anthropic/claude-opus-5 (OpenRouter) | exp | 10/10 | **15.00** | 13.38 | 16,688 | $4.17 |
| deepseek-v4-flash | exp | 54/101 | **15.00** | 13.17 | 51,695 | $1.46 |
| deepseek-v4-flash `effort=low` | exp | **98/101** | **15.00** | 5.04 | 26,520 | **$0.75** |
| Claude Opus 5 (CLI) `effort=max` | exp | 20/20 | **15.00** | 5.56 | 15,690 | sub |
| z-ai/glm-5.2 | exp | 18/101 | **15.00** | 8.37 | 30,181 | free |
| **kimi-k2.6** | exp | 13/20 | **13.68** | 8.31 | 57,209 | free |
| Claude Fable 5 (CLI) | exp | 62/101 | **14.68** | 10.49 | 5,503 | sub |
| Claude Opus 5 (CLI) | sin | 101/101 | **14.38** | 8.32 | 7,924 | sub |

**2 — The sparse middle.** They genuinely spend tokens deriving, and still fall short.

| model | fn | answered | median | worst | tok/pt |
|---|---|---|---|---|---|
| anthropic/claude-sonnet-5 | exp | 10/10 | 12.10 | 9.30 | 22,630 |
| anthropic/claude-opus-4.8 `effort=high` | exp | 10/10 | 11.13 | 9.20 | 4,141 |
| minimax-m2.7 | exp | 1/3 | 9.86 | 9.86 | 59,263 |
| gpt-oss:120b | exp | 98/101 | 9.55 | 3.18 | 9,105 |
| poolside/laguna-xs-2.1 | exp | 2/3 | 6.57 | 6.14 | 24,345 |
| gpt-oss:20b | exp | 93/101 | 6.37 | 2.44 | 26,420 |
| qwen/qwen3.7-flash | exp | 80/101 | 5.63 | 2.84 | 27,579 |
| **qwen3.5:397b** | exp | **101/101** | 5.19 | 3.95 | 6,015 |

**3 — Shortcutters.** ~18 tokens per point, 3–5 digits, essentially never fail and never improve.

| model | size | fn | answered | median | tok/pt |
|---|---|---|---|---|---|
| google/gemma-4-26b-a4b (MoE) | 26B | exp | 3/3 | 4.91 | 19 |
| gemma4:31b (dense) | 31B | exp | 101/101 | 4.76 | 18 |
| anthropic/claude-opus-4.8 (default) | — | exp | 10/10 | 4.66 | **9** |
| upstage/solar-pro4 | — | exp | 101/101 | 4.41 | 18 |
| **mistral-large-3:675b** | **675B** | exp | 101/101 | 4.29 | 18 |
| qwen3-coder | ~30B | exp | 101/101 | 3.72 | 18 |
| **devstral:24b** | **24B** | exp | 101/101 | 3.46 | 16 |
| qwen2.5-coder:14b | 14B | exp | 101/101 | 2.93 | 18 |

**4 — Never finish.** They reason until the output cap and return **empty content**.

| model | answered | tok/pt | endpoint |
|---|---|---|---|
| minimax-m3 | 0/101 | 32,000 (cap) | Ollama Cloud |
| inclusionai/ling-3.0-tiny | 0/101 | 27,564 | OpenRouter |
| cohere/north-mini-code | 0/3 | 26,176 | OpenRouter |
| deepseek-v4-pro | 0/3 | 65,536 (cap) | Ollama Cloud |
| glm-5.1 | 0/3 | 32,768 (cap) | Ollama Cloud |
| claude-opus-4.6 `effort=high`, uncapped | 0/10 | 65,536 (cap) | OpenRouter |

Plus one category of its own: **the Nemotron family is broken on this task at every size.**
`nemotron-3-nano-30b` scored a median of **0.00** correct digits over 101 points (one reply was
`229144000.0` where the answer is `0.89`); `nemotron-3-ultra-550b` managed 3.69; and the
*reasoning-tuned* `nemotron-3-nano-omni-30b-reasoning` scored **−0.00** after burning 32,768 tokens
per point. Three models, three sizes, two endpoints, one result.

<details>
<summary>Original 36-run table (sorted purely by median)</summary>


| model | fn | answered | median | worst | tokens | cost | notes |
|---|---|---|---|---|---|---|---|
| anthropic/claude-opus-4.7 | exp | 2/10 | **15.00** | 15.00 | 279,493 | $6.99 | effort=high, cap 32k |
| openai/gpt-5.6-sol | sin | 5/5 | **15.00** | 14.31 | 22,205 | $0.67 |  |
| openai/gpt-5.6-sol | exp | 5/5 | **15.00** | 14.17 | 21,271 | $0.64 |  |
| anthropic/claude-opus-5 | exp | 10/10 | **15.00** | 13.38 | 166,881 | $4.17 |  |
| deepseek-v4-flash | exp | 54/101 | **15.00** | 13.17 | 5,221,157 | $1.46 |  |
| deepseek-v4-flash | sin | 39/101 | **15.00** | 11.05 | 5,904,512 | $1.65 |  |
| z-ai/glm-5.2 | exp | 3/21 | **15.00** | 9.41 | 1,301,405 | $3.15 |  |
| glm-5.2 | exp | 18/101 | **15.00** | 8.37 | 3,048,291 | free |  |
| openai/gpt-5.6-sol-pro | exp | 5/5 | **15.00** | 6.18 | 69,237 | $2.08 |  |
| Claude Opus 5 (CLI) | exp | 20/20 | **15.00** | 5.56 | 313,793 | sub | effort=max |
| deepseek-v4-flash | exp | 98/101 | **15.00** | 5.04 | 2,678,556 | $0.75 | effort=low |
| Claude Opus 5 (CLI) | sin | 20/20 | **15.00** | 3.43 | 327,260 | sub | effort=max |
| Claude Opus 5 (CLI) | exp | 21/21 | **14.87** | 4.33 | 80,758 | sub |  |
| Claude Fable 5 (CLI) | exp | 62/101 | **14.68** | 10.49 | 555,843 | sub |  |
| Claude Fable 5 (CLI) | exp | 20/20 | **14.43** | 11.78 | 181,633 | sub |  |
| deepseek-v4-flash | sin | 98/101 | **14.38** | -0.30 | 3,003,926 | $0.84 | effort=low |
| Claude Opus 5 (CLI) | sin | 101/101 | **14.38** | 8.32 | 800,352 | sub |  |
| Claude Opus 5 (CLI) | exp | 101/101 | **14.37** | 3.88 | 566,897 | sub |  |
| Claude Fable 5 (CLI) | sin | 20/20 | **14.23** | 3.77 | 211,897 | sub |  |
| nvidia/nemotron-3-super-120b-a12b | exp | 8/101 | **12.71** | 2.98 | 79,109 | free |  |
| anthropic/claude-sonnet-5 | exp | 10/10 | **11.98** | 9.30 | 226,295 | $2.26 |  |
| anthropic/claude-opus-4.8 | exp | 10/10 | **11.04** | 9.20 | 41,409 | $1.04 | effort=high, cap 32k |
| gpt-oss:120b | exp | 98/101 | **9.44** | 3.18 | 919,632 | free |  |
| anthropic/claude-opus-4.6 | exp | 10/10 | **6.93** | 3.70 | 10,549 | $0.26 |  |
| gpt-oss:20b | exp | 93/101 | **6.37** | 2.44 | 2,668,420 | free |  |
| qwen/qwen3.7-flash | sin | 69/101 | **6.06** | 3.15 | 1,931,809 | $0.25 |  |
| qwen/qwen3.7-flash | exp | 80/101 | **5.61** | 2.84 | 2,785,476 | $0.36 |  |
| qwen3.5:397b | exp | 101/101 | **5.37** | 3.49 | 653,567 | free |  |
| Claude Haiku 4.5 (CLI) | exp | 20/20 | **4.91** | 2.47 | 215,964 | sub |  |
| gpt-oss:120b | exp | 101/101 | **4.83** | 3.18 | 47,723 | free | effort=low |
| gemma4:31b | exp | 101/101 | **4.76** | 2.97 | 1,804 | free |  |
| anthropic/claude-opus-4.8 | exp | 10/10 | **4.61** | 4.02 | 90 | $0.00 |  |
| anthropic/claude-opus-4.7 | exp | 10/10 | **4.46** | 3.57 | 553 | $0.01 |  |
| Claude Opus 4.8 (CLI) | exp | 101/101 | **4.46** | 2.50 | 13,109 | sub |  |
| Claude Opus 4.8 (CLI) | exp | 101/101 | **4.44** | 2.92 | 3,370 | sub | --effort high (no-op on 4.8) |
| gemma4:31b | sin | 101/101 | **4.09** | 2.43 | 1,880 | free |  |

</details>

`sub` = Claude Code subscription usage (not a per-token bill). `free` = Ollama Cloud free tier.
Sampled runs (`n/10`, `n/20`, `n/5`) use `--sample`, a deterministic subset of the same grid, so
every model is scored on identical arguments.

### Cost per correct answer

Median accuracy alone hides an order-of-magnitude spread in what a correct digit costs:

| model | $ per 101-point grid | answered | ≥12 digits | **$ per 12-digit point** |
|---|---|---|---|---|
| **DeepSeek-V4-Flash, `effort=low`** | **$0.75** | 98/101 | 82 | **$0.009** |
| DeepSeek-V4-Flash, default | $1.46 | 54/101 | 54 | $0.027 |
| gpt-5.6-sol | ~$12.93 | 5/5 | 5 | $0.128 |
| Claude Opus 5 (OpenRouter) | ~$42.12 | 10/10 | 10 | $0.417 |
| Claude Sonnet 5 | ~$22.83 | 10/10 | 5 | $0.452 |
| Claude Opus 4.8, `effort=high` | ~$10.50 | 10/10 | 2 | $0.520 |

**DeepSeek-V4-Flash at `effort=low` is the standout: a 12-significant-digit answer for about one
cent, ~14x cheaper than the next best and ~46x cheaper than Opus 5 for the same median.** Two
caveats keep it from being a blanket recommendation: at default effort it exhausts its output
budget on 46% of points, and at low effort it produced the study's most dangerous artifact — a
`sin` value correct to eleven digits **with the wrong sign**.

Three findings the round-argument grid could not have shown:

1. **gemma4 loses ~6 digits when it cannot recall.** Median 10.40 → 4.76; points reaching 9+ digits
   fall from 67/101 to 1/101. Same model, same prompt, same ~18 tokens per answer. Its test1 score
   was a lookup table almost in full.
2. **DeepSeek gets *more* accurate and stops finishing.** Worst case improves 6.18 → 13.17 digits,
   because with nothing to recall it derives every value (argument splitting + Taylor series,
   cross-checked — one trace is 46,000 characters of long multiplication for a single point). But
   47 of 101 calls exhaust the 65,536-token output limit mid-arithmetic and return **empty**.
3. **Throttling the thinking trades silence for silent errors.** `reasoning_effort=low` lifts the
   answered rate 54 → 98 at half the cost, median still 15 digits — but the tail collapses. The
   worst `sin` point came back as `+0.5423228246946903` where the answer is `-0.54232282470165383`:
   **eleven correct digits with the sign flipped.**

![DeepSeek on recall-proof arguments, unbounded thinking](test1v2/results/exp_ds-v4-flash_v2/plots/1_accuracy_vs_x.png)

*Unbounded thinking.* Every answered point sits on the double-precision line; the red row along the
bottom is the 47 calls that never produced one, and the lower panel shows why — they are pinned
against the 65,536-token output cap.

![DeepSeek with reasoning_effort=low](test1v2/results/exp_ds-v4-flash_v2_efflow/plots/1_accuracy_vs_x.png)

*Throttled thinking.* Same model, same grid: the token panel drops off the cap and nearly every
point now returns — at the cost of a spray of answers falling to 5–12 digits.


### Two findings from the wider model sweep

**Thinking volume does not buy digits — method does.** `qwen3.5:397b` spends ~6.5k tokens per
point reasoning and lands at **5.37** digits; `gpt-oss:120b` at `effort=low` spends ~470 and lands
at **4.83**. Fourteen times the thinking buys half a digit. What separates the 15-digit models is
*what* they do (argument splitting + series, cross-checked) rather than how long they do it.

**The same model differs by host.** Claude Opus 5 through the Claude Code CLI averages ~3.8k
tokens/point and bottoms out at 4.33 digits; through OpenRouter it averages ~16.7k and bottoms out
at 13.38. Same weights, ~4.4× the thinking, and the difference lands entirely in the tail — the
hosting configuration is part of the measurement, not a detail.

**Shortcut answers are where every model fails.** The worst point of almost every run is also its
cheapest: DeepSeek's 6.18-digit answer took 128 tokens, GLM's 3.29-digit answer took 13, and Claude
Opus 5's 4.33-digit answer took **9**. A model that decides not to derive is a model about to be
wrong, at every tier.

### What the wider sweep added

**Model size and thinking budget are different axes, and neither predicts precision.**
The cleanest controlled pair is `gpt-oss` on one host: the **120b** scores 9.44 median on ~9k
tokens/point, the **20b** scores **6.37 on ~28.7k** — the smaller model spends 3x more to land 3
digits lower. It does not think less; it flails. The same shape appears across vendors: Claude
Haiku 4.5 spends *more* per point than Claude Fable 5 (10.8k vs 9.1k) and lands **ten digits**
lower (4.94 vs 14.44).

**Effort raises the median and does not fix the tail.** Claude Opus 5 at `--effort max` reaches a
clean 15.00 median on both functions, but its worst point stays broken — 5.56 digits on `exp`, and
on `sin` the floor actually *drops* (8.32 at default effort, 3.43 at max). Across every model,
tier, vendor and effort setting measured here, the worst answer is almost always the cheapest one,
and nothing tested eliminates it.

**Median accuracy is a model property; the tail is not.** Claude Fable 5 has the best tail measured
anywhere (10.49 worst over 62 answered `exp` points) and still shortcuts to 3.77 on `sin`.

**API defaults dominate small-model-vs-large-model differences.** Opus 4.7/4.8 do not think unless
asked: at their default they answer in 9-130 tokens and score ~4.5 digits, indistinguishable from
gemma4. Given `reasoning_effort` *and* a `max_tokens` cap, 4.8 reaches 11.04. The real 4.8-to-5 gap
is ~4 digits, not the ~10 a naive default-settings comparison suggests.

**Endpoint behaviour is part of the measurement.** The same Claude Opus 5 spends ~3x more tokens
through OpenRouter than through the Claude Code CLI and has a much better tail (13.38 vs 3.88).
Output caps decide what is even measurable: GLM-5.2 needs ~80k tokens/point and returns nothing
under OpenRouter's 65,536 ceiling. And NVIDIA NIM answered **8 of 101** points over 8.3 hours, the
rest `HTTP 504` — an endpoint result, not a model result.

**Capped runaways are the expensive failure mode.** An uncapped `effort=high` sweep on Opus 4.6
billed 655,360 tokens for **zero** answers. Always pass `--max-tokens` on a paid endpoint: the same
model, same effort, with a 32k cap finished in 6,849 tokens and answered.

No configuration gives "always answers and always correct": no thinking → always answers, never
accurate; unbounded thinking → exact or silent; throttled thinking → occasionally, quietly wrong.

### What the August 2026 sweep added (34 more runs, 6 new vendors)

**A single sentence of prompting moved a model from 0/101 to 15.00 digits.** `minimax-m3` scored
**0 of 101** points — every one burning the full budget and returning empty. Its traces show why:
it does *schoolbook column arithmetic* on 24-digit operands —

```
Col 7: 5 + 1 + 1 = 7, write 7, no carry
Result: 522815730554999028331286
```

— which costs ~24 lines per multiplication and cannot finish in any budget. DeepSeek never does
this; it splits algebraically (`A*0.12835 = A*0.1 + A*0.02 + A*0.008 + A*0.00035`), ~4–8 lines. So
we told minimax to do the same, on two points at its own 65,536 ceiling:

| arm | x = −0.9986 | x = 0.9246 |
|---|---|---|
| baseline | capped, **empty** | capped, **empty** |
| "you have a limited budget, commit to the shortest route" | 18,148 tok → **12.79 digits** | 61,759 tok → **14.98** |
| explicit method hint (reduce → series → split) | 46,036 tok → **15.00** | capped, empty |
| both | capped, empty | 32,132 tok → **15.00** |

Doubling the budget alone does **not** help (a 20-point run at the 65,536 cap scored 1/20). One
sentence does. **A meaningful part of what looks like capability here is strategy selection** — the
model can do the arithmetic and chooses a method that cannot finish. Reported separately from the
main table, which keeps the identical prompt for every model.

**The host matters more than the model — but only for models that reason.** The same
DeepSeek-V4-Flash-0731 weights, the same `reasoning_effort=low`, the same 65,536 cap (each host's
own ceiling):

| host | median tok/pt | hit the ceiling | answered |
|---|---|---|---|
| `api.deepseek.com` (native) | 24,370 | 3% | **97%** |
| OpenRouter | 25,338 | 0% | 75% |
| Ollama Cloud | **65,536 — the ceiling itself** | **80%** | **20%** |

Ollama silently ignores `reasoning_effort` for this model, so it reasons at full depth and gets
truncated four times in five. Native and OpenRouter agree within 4%. By contrast `gemma4:31b`
scores the same run locally (4.32) as on Ollama Cloud (4.76) — **a serving layer can only break
what the model actually uses.**

**Neither size nor architecture predicts anything.** Mistral shortcuts at **24B (3.46 digits)** and
at **675B (4.29)** — a 28× parameter increase buys ~0.8 digits and changes nothing structurally.
Google's gemma scores 4.76 dense and 4.91 as an MoE. And Poolside's *smaller* `laguna-xs` (6.57)
beats its larger sibling `laguna-s` (4.13). Eight vendors span 14B→675B inside the same 2.8–4.9
digit band, all at ~18 tokens per point.

**Beware the 3-point smoke.** `laguna-s` looked like a clean shortcutter from its smoke (18 tokens,
4.51 digits). Its full grid: **20/101 answered** — the 20 answers really are ~18-token shortcuts,
but the other 81 grind to the cap and die. Smokes set direction, not values; every headline number
here is from a full grid or an explicit `--sample`.

**Your own `--max-tokens` is an experimental variable, not a safety rail.** Two "the host is broken"
conclusions in this sweep were wrong — both were a 32k cap sitting below the model's natural spend,
which looks exactly like a host misconfiguration. If a run's *median* token count equals your cap,
you measured the cap.

## Run it

```bash
# hosted
DEEPSEEK_API_KEY=sk-... ./test1/run_test1.py --model deepseek-v4-flash
DEEPSEEK_API_KEY=sk-... ./test1v2/run_test1v2.py --func sin --model deepseek-v4-flash \
    --reasoning-effort low

OLLAMA_API_KEY=... ./test1v2/run_test1v2.py --func exp --base-url https://ollama.com/v1 \
    --model gemma4:31b --api-key-env OLLAMA_API_KEY

# local OpenAI-compatible server
./test1v2/run_test1v2.py --func exp --base-url http://127.0.0.1:8090/v1 --model qwen3.6 \
    --api-key dummy
```

```bash
# Claude Code CLI backend — no API key; zero tools, no CLAUDE.md loaded
./test1v2/run_test1v2.py --api claude-cli --func exp

# native Anthropic Messages API (adaptive thinking, summarized reasoning)
ANTHROPIC_API_KEY=sk-ant-... ./test1v2/run_test1v2.py --api anthropic --model claude-opus-5

# any aggregator; --sample runs a deterministic subset of the same grid
OPENROUTER_API_KEY=sk-or-v1-... ./test1v2/run_test1v2.py \
    --base-url https://openrouter.ai/api/v1 --model anthropic/claude-opus-5 \
    --api-key-env OPENROUTER_API_KEY --sample 10 --no-temperature
```

Three backends, selected with `--api`:

| `--api` | wire protocol | credentials |
|---|---|---|
| `openai` (default) | `POST {base}/chat/completions` | bearer token from `--api-key-env` |
| `anthropic` | `POST {base}/v1/messages`, adaptive thinking, `output_config.effort` | `ANTHROPIC_API_KEY` |
| `claude-cli` | execs `claude -p` with `--tools ""` and a throwaway `$HOME`/cwd | the machine's Claude Code login |

The `claude-cli` backend is the isolation-sensitive one: `--tools ""` disables the entire built-in
tool set (otherwise the model just computes the answer with `python`), and the throwaway `$HOME`
plus empty working directory keep every `CLAUDE.md` — user and project — out of the context.

Keys are read from the environment only — nothing is stored in the repo.

Useful flags: `--concurrency`, `--temperature`, `--reasoning-effort low|medium|high|max|none`,
`--max-tokens`, `--timeout`, `--retries`, `--lo/--hi/--step`, `--seed`, `--tag`.

## Plots

```bash
python3 -m venv .venv-plot && .venv-plot/bin/pip install matplotlib   # once
.venv-plot/bin/python test1/plot_results.py test1v2/results/<tag> [--dark]
```
Three figures per run: accuracy per point (with a second panel showing tokens spent on the same
point), the digit histogram, and effort vs accuracy. Light and dark variants.

## Output layout

```
test1[v2]/results/<tag>/
  raw.jsonl      one line per point: reply, full reasoning trace, usage, latency
  results.csv    x, expected, got, abs_err, rel_err, correct_digits, tokens
  summary.json   every metric, plus the exact prompts and run metadata
  plots/         rendered figures
  REPORT.md      written up per run (where present)
```

`test1/analysis/` holds a reasoning-trace study: what the model actually *does* inside the thinking
block (method markers, self-checks, doubt markers, and their correlation with accuracy), plus
verbatim traces of the least accurate points.

## Method notes

- temperature 0; one stateless request per point; no tools; no shared context
- reference values from Python `math` (correctly rounded doubles, < 1 ulp)
- a reply with no parseable number is recorded as a failure, never as a wrong value
- n = 1 per point per model; single function family; results are per-model-version and dated
- test1v2 also strengthened the digit request from "≥12" to "≥15, do not truncate" — recorded in
  `summary.json:meta` so the ablation is not silently confounded (in test1, "at least 12
  significant digits" made DeepSeek truncate an exact answer down to 12 digits)

## Prior work

Related but, as far as I found, not the same experiment: numerical-perturbation robustness studies
(ACL 2025), integer-arithmetic memorisation-vs-computation work (arXiv 2308.01154, 2504.05262,
2402.17709), and classical ULP error analyses of libm implementations. This combines transcendental
functions, digit-level grading on a dense grid, and a round-vs-recall-proof ablation. If it has been
done before, please point me at it.
