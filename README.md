# LLM-Tests — how many correct digits does an LLM actually produce?

Small, reproducible harness that asks a model for the value of an elementary function — **one
stateless API call per point, no tools, no code execution, no conversation** — and scores the reply
by how many significant digits agree with IEEE-754 double precision.

The metric is `correct digits = -log10(relative error)`, capped at 15. Pass/fail hides everything
interesting here; "4.8 digits" and "15 digits" are both "wrong" to a boolean grader.

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

**Recall-proof arguments (test1v2):**

| run | answered | median digits | worst | tokens |
|---|---|---|---|---|
| gemma4 `exp` | 101/101 | **4.76** | 2.97 | 1.8k |
| gemma4 `sin` | 101/101 | **4.09** | 2.43 | 1.9k |
| DeepSeek `exp` | 54/101 | 15.00 | 13.17 | 5.2M |
| DeepSeek `sin` | 39/101 | 15.00 | 11.05 | 5.9M |
| DeepSeek `exp`, `reasoning_effort=low` | 98/101 | 15.00 | 5.04 | 2.7M |
| DeepSeek `sin`, `reasoning_effort=low` | 98/101 | 14.38 | **−0.30** | 3.0M |

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

No configuration gives "always answers and always correct": no thinking → always answers, never
accurate; unbounded thinking → exact or silent; throttled thinking → occasionally, quietly wrong.

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
