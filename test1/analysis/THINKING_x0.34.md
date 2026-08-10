# How DeepSeek-V4-Flash computes `exp(0.34)` — reasoning-trace analysis

Single stateless call, temperature 0, test1 prompt. **6,063 reasoning tokens · 13,448 chars ·
175 lines · 37.6s · answer `1.404947590563593` (abs err 6.7e-16 ≈ 1 ulp, 15.3 correct digits).**

Full verbatim trace: [`thinking_x0.34_raw.md`](thinking_x0.34_raw.md).

## It solves the problem twice, by two independent methods

### Method A — argument splitting + recalled constants (~60% of the trace)
```
e^0.34 = e^(34/100) = e^(17/50)
I know e^0.3 = 1.3498588075760032, e^0.04 = 1.0408107741923882, product = e^0.34
```
Both recalled constants are correct to all 17 digits it states — the model is reading a memorized
table, not deriving them. It then does the multiplication **by hand, digit by digit**, recursively
splitting the multiplier into decreasing decades:

```
Break 1.0408107741923882 = 1 + 0.0408107741923882
1.3498588075760032 * 0.0408107741923882
  = *0.04 + *0.0008107741923882
  = *0.0008 + *0.0000107741923882
  = *0.00001 + *0.0000007741923882
  ... down to *0.0000000000006118
```
≈111 explicit multiplications, 297 `=` steps. Result: `1.4049475905635938`.

### The pivot — it distrusts its own product
```
So exp(0.34) ≈ 1.4049475905635938. Need check with actual known value.
Let's ensure multiplication precision. Could be slightly off due to approximations
of e^0.3 and e^0.04? ... Let's compute more accurately using high precision method maybe.
```
7 self-check phrases, 4 explicit cross-check pivots in this trace.

### Method B — Taylor series from scratch (~40% of the trace)
```
e^0.34 = sum_{n=0}^inf 0.34^n / n!
```
It computes terms **t1 … t13** individually — each `0.34^n` by repeated multiplication, each `n!`
division by long division — accumulating a running `S` after every term:

```
t10 = 5.688871677e-12   -> S = 1.4049475905634120825
t11 = 1.7583785e-13     -> S = 1.4049475905635878664
t12 = 4.98208e-15       -> S = 1.40494759056359285
t13 ~ 1.303e-16         -> S = 1.40494759056359298
So exp(0.34) ≈ 1.404947590563593. Good.
```
It stops when the term drops below the 16th digit — i.e. it tracks its own truncation error.

### Then a formatting side-quest (~5%)
The last ~10 lines are spent counting significant digits of its own answer to confirm it satisfies
"at least 12 significant digits", and re-deciding between `1.404947590563593` and
`1.40494759056359`.

## What this means for the test

- **The accuracy is real, not recall of the final answer.** Method A reuses memorized `e^0.3`/`e^0.04`,
  but Method B derives the value from the series with no lookup and lands on the same 16 digits. The
  two agreeing is why it's confident.
- **~6k thinking tokens per point is structural, not waste-per-se**: two full methods + a
  cross-check. A single method would plausibly cost ~2–3k. The `--reasoning-effort` /
  budget knob is the lever if the run cost matters.
- **Arithmetic is the bottleneck, not the maths.** It knows the algorithm immediately; the tokens go
  into hand-multiplication and long division. Failure modes to watch for at other grid points:
  a digit dropped mid-decomposition, or the series truncated a term too early.
- **Cost model:** 6,071 completion tokens × $0.28/M ≈ $0.0017 per point; 101 points ≈ **$0.17** and
  ~8 min at concurrency 8.

## Cross-point analysis

`analysis/analyze_thinking.py results/<tag>/raw.jsonl` profiles every point's trace: token volume,
which method markers appear (recalled constants / Taylor / squaring / log inversion /
long multiplication), self-check and doubt-marker counts, distinct candidate answers, and the
correlation between each of those and the final accuracy. It writes `THINKING_ANALYSIS.md`,
`thinking_profile.csv`, and dumps the full traces of the least accurate points for review.
