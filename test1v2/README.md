# LLM-Tests / test1v2 — elementary functions on **recall-proof** arguments

## Why v2 exists

test1 used round arguments (`x = -1.00, -0.98, …, 1.00`). Its reasoning logs showed the models
**recalling** those values rather than computing them:

- DeepSeek answered `exp(0.1)` in 128 tokens from a memorized constant — and the memorized string
  was corrupt from the 7th digit (`1.105170185988091` vs `1.1051709180756477`). Its *worst* point
  was its *cheapest*.
- gemma4:31b emitted a correct 6–8 digit prefix plus fabricated tail digits on most points
  (a recurring `…115` / `…1115` motif), and once dropped the decimal point entirely.

So a round-argument grid measures the lookup table as much as the arithmetic. v2 removes the table.

## What changed

1. **Recall-proof arguments.** Each of v1's 101 grid points is nudged by a deterministic,
   sha256-derived offset with ~10 decimal places:

   ```
   x_i = -1 + 0.02*i + delta_i,    delta_i ∈ (-0.0095, 0.0095), rounded to 10 decimals
   ```
   e.g. `-0.9985798483, -0.9814799602, -0.9513299347, …`. Same coverage of [-1, 1], same 101
   points, same everything else — but no argument that could plausibly sit in a training corpus.
   The grid is a pure function of `--seed` (default `test1v2`), so runs are reproducible and every
   model sees identical arguments.

2. **The prompt no longer invites truncation.** v1 asked for "at least 12 significant digits";
   DeepSeek responded to that at x=-1 by truncating an exact answer down to exactly 12 digits,
   throwing away 3 correct digits. v2 asks for **≥15** and adds *"Give every digit you are
   confident in — do not round or truncate the result to fewer digits than you computed."*
   `--sigdigits 12 --no-antitrunc` reproduces the v1 wording.

Both changes are recorded in each run's `summary.json` under `meta`.

## Functions

`--func exp | sin | cos | log | sqrt`. `sin`/`cos` prompts say **RADIANS (not degrees)**
explicitly. `log`/`sqrt` shift the grid into `(0, 2]`. Reference values come from Python `math`
(IEEE-754 double, correctly rounded to <1 ulp).

`sin` is the interesting companion to `exp`: same "simple function, small argument" shape, but a
different algorithm (alternating series, so intermediate terms cancel) and a different memorized
table. If a model's v1 accuracy came from recall, `sin` on recall-proof arguments is where it
shows.

## Metrics

Identical to test1 — answered/failed, exact double matches, max/mean/RMS absolute error,
max/mean relative error, min/median/mean correct significant digits (`-log10(rel err)`, capped at
15), digit histogram (with a `<0 (gross)` bucket for answers off by more than the value itself),
worst point with its raw reply, latency, completion/reasoning tokens.

## Run

```bash
DEEPSEEK_API_KEY=sk-... ./run_test1v2.py --func exp --model deepseek-v4-flash
DEEPSEEK_API_KEY=sk-... ./run_test1v2.py --func sin --model deepseek-v4-flash

OLLAMA_API_KEY=...  ./run_test1v2.py --func sin --base-url https://ollama.com/v1 \
    --model gemma4:31b --api-key-env OLLAMA_API_KEY

# any local OpenAI-compatible server
./run_test1v2.py --func exp --base-url http://127.0.0.1:8090/v1 --model qwen3.6 --api-key dummy
```

Output layout matches test1 (`results/<tag>/raw.jsonl · results.csv · summary.json`), so
`../test1/plot_results.py results/<tag>` renders the same three figures.

Note: 10-decimal arguments are *harder to multiply by hand* than round ones, so reasoning models
spend more tokens per point in v2 than in v1 — budget accordingly.
