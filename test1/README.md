# LLM-Tests / test1 — `exp(x)` numeric accuracy, pure API

**Question it answers:** how accurately does a model compute `exp(x)` when it is given no tools,
no code execution, and no conversation context — one stateless API call per value?

## Design
- **Grid:** `x = -1.00 .. 1.00, step 0.02` → **101 points** (both endpoints inclusive; a 2.0-wide
  range at 0.02 is 100 intervals = 101 samples).
- **One independent request per point.** No agent loop, no tools, no shared session — every call
  is a fresh `POST /chat/completions` with the same 2-message prompt and a different `x`.
- **Prompt** (fixed, recorded in every result's `meta`):
  - system: *"You are a numerical calculator. Answer with the numeric result only: a single decimal
    number, no words, no units, no formatting, no explanation."*
  - user: *"Compute exp({x}), the exponential function e raised to the power {x}. Give the value as
    a plain decimal number with at least 12 significant digits. Output only the number."*
- **Parsing:** last standalone number in the reply (tolerates a restatement prefix); a reply with no
  number counts as a failure, never as a wrong value.
- **Reference:** Python `math.exp(x)` (IEEE-754 double, correctly rounded to <1 ulp).
- **Sampling:** `temperature 0` by default — this is a determinism test, not a creativity test.

## Metrics
| metric | meaning |
|---|---|
| `answered` / `unparsed_or_failed` | did the model return a parseable number at all |
| `exact_float_matches` | replies that round-trip to the exact same double as `math.exp(x)` |
| `max/mean/rms_abs_err` | absolute deviation from `math.exp(x)` |
| `max/mean_rel_err` | relative deviation |
| `min/mean_correct_digits` | `-log10(relative error)`, capped at 15 — significant digits that agree |
| `digit_histogram` | how many points landed at each accuracy level |
| `worst_abs` / `worst_rel` | the single worst point, with the model's actual answer |
| latency / tokens | per-call wall time, completion + reasoning tokens |

`min_correct_digits` is the headline: it is the accuracy you can *rely* on across the range.

## Run
```bash
DEEPSEEK_API_KEY=sk-... ./run_test1.py --model deepseek-v4-flash

# any OpenAI-compatible endpoint (local llama.cpp / SGLang / vLLM included)
./run_test1.py --base-url http://127.0.0.1:8090/v1 --model qwen3.6 --api-key dummy \
               --concurrency 4 --temperature 0
```
Useful flags: `--concurrency`, `--temperature`, `--reasoning-effort <level>` (passed through when the
provider supports it), `--max-tokens`, `--retries`, `--lo/--hi/--step`, `--tag`.

## Output
`results/<tag>/`
- `raw.jsonl` — one line per point: request result, raw reply text, reasoning text, usage
- `results.csv` — `x, expected, got, abs_err, rel_err, correct_digits, latency_s, tokens, error`
- `summary.json` — all metrics above plus the exact prompts and run metadata

Stdlib only; runs under any `python3`.
