#!/usr/bin/env python3
"""test1v2 — elementary-function accuracy on RECALL-PROOF arguments, pure API.

test1 used round arguments (x = -1.00, -0.98, …). Its logs showed models *recalling* memorized
constants for those: DeepSeek answered exp(0.1) from memory in 128 tokens and was wrong from the
7th digit; gemma4 emitted fabricated digit tails on almost every point. A round argument therefore
measures the model's lookup table as much as its arithmetic.

v2 fixes that. Each grid point is perturbed by a deterministic pseudo-random offset with ~10
decimal places, so the argument cannot plausibly appear in any training corpus:

    x_i = -1 + 0.02*i + delta_i,   delta_i ∈ (-0.0095, 0.0095), 10 decimals, sha256-derived

Same coverage of [-1, 1] and the same 101 points as v1, so the two are directly comparable; the
only change is that the model must *compute* rather than recall.

Two differences from v1, both deliberate and both recorded in summary.json:
  1. recall-proof arguments (above);
  2. the prompt asks for >= 15 significant digits and explicitly forbids truncation. In v1
     "at least 12 significant digits" made DeepSeek truncate an exact answer at x=-1 down to
     12 digits, losing 3 digits of a correct result. Use --sigdigits 12 --no-antitrunc to
     reproduce the v1 wording.

Functions: exp, sin (radians), cos, log (natural, x shifted to (0,2]), sqrt.

Usage:
  DEEPSEEK_API_KEY=sk-... ./run_test1v2.py --func exp --model deepseek-v4-flash
  OLLAMA_API_KEY=... ./run_test1v2.py --func sin --base-url https://ollama.com/v1 \
      --model gemma4:31b --api-key-env OLLAMA_API_KEY
Stdlib only.
"""
import argparse, concurrent.futures as cf, hashlib, json, math, os, re, sys, time
import urllib.error, urllib.request
from datetime import datetime, timezone

FUNCS = {
    #  name: (python reference, prompt noun, domain mapper from the base grid)
    "exp":  (math.exp,  "exp({x}), the exponential function e raised to the power {x}", lambda t: t),
    "sin":  (math.sin,  "sin({x}), the sine of {x} RADIANS (not degrees)",              lambda t: t),
    "cos":  (math.cos,  "cos({x}), the cosine of {x} RADIANS (not degrees)",            lambda t: t),
    "log":  (math.log,  "ln({x}), the natural logarithm of {x}",                        lambda t: t + 1.0001),
    "sqrt": (math.sqrt, "sqrt({x}), the square root of {x}",                            lambda t: t + 1.0001),
}

SYSTEM = ("You are a numerical calculator. Answer with the numeric result only: a single decimal "
          "number, no words, no units, no formatting, no explanation.")
NUM_RE = re.compile(r'[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?')


def recall_proof_grid(lo, hi, step, jitter, seed):
    """v1's grid, each point nudged by a deterministic ~10-decimal offset."""
    n = int(round((hi - lo) / step))
    xs = []
    for i in range(n + 1):
        base = lo + i * step
        h = hashlib.sha256(f"{seed}:{i}".encode()).digest()
        u = int.from_bytes(h[:8], "big") / 2**64 * 2 - 1          # u in (-1, 1)
        if i == 0:
            u = abs(u)                                            # keep inside [lo, hi]
        elif i == n:
            u = -abs(u)
        xs.append(round(base + u * jitter, 10))
    return xs


def post(url, key, payload, timeout):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
        method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def parse_number(text):
    if text is None:
        return None
    m = NUM_RE.findall(text.replace(",", ""))
    if not m:
        return None
    try:
        return float(m[-1])
    except ValueError:
        return None


def build_user(args, x):
    noun = FUNCS[args.func][1].format(x=repr(x))
    tail = (f"Give the value as a plain decimal number with at least {args.sigdigits} significant "
            f"digits.")
    if not args.no_antitrunc:
        tail += (" Give every digit you are confident in — do not round or truncate the result to "
                 "fewer digits than you computed.")
    return f"Compute {noun}. {tail} Output only the number."


def ask(args, key, x):
    ref = FUNCS[args.func][0]
    payload = {"model": args.model, "temperature": args.temperature,
               "messages": [{"role": "system", "content": SYSTEM},
                            {"role": "user", "content": build_user(args, x)}]}
    if args.max_tokens:
        payload["max_tokens"] = args.max_tokens
    if args.reasoning_effort:
        payload["reasoning_effort"] = args.reasoning_effort

    last_err = None
    for attempt in range(args.retries + 1):
        t0 = time.time()
        try:
            resp = post(args.base_url.rstrip("/") + "/chat/completions", key, payload, args.timeout)
            dt = time.time() - t0
            msg = (resp.get("choices") or [{}])[0].get("message", {}) or {}
            usage = resp.get("usage", {}) or {}
            return {"x": x, "expected": ref(x), "got": parse_number(msg.get("content")),
                    "raw": msg.get("content"),
                    "reasoning": msg.get("reasoning_content") or msg.get("reasoning"),
                    "latency_s": round(dt, 3), "attempt": attempt,
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "reasoning_tokens": ((usage.get("completion_tokens_details") or {})
                                         .get("reasoning_tokens", 0)),
                    "cache_hit_tokens": usage.get("prompt_cache_hit_tokens", 0),
                    "error": None}
        except Exception as e:                                   # noqa: BLE001 - record, retry
            last_err = f"{type(e).__name__}: {e}"
            if isinstance(e, urllib.error.HTTPError):
                try: last_err += " :: " + e.read().decode()[:300]
                except Exception: pass
            time.sleep(1.5 * (attempt + 1))
    return {"x": x, "expected": ref(x), "got": None, "raw": None, "reasoning": None,
            "latency_s": None, "attempt": args.retries, "prompt_tokens": 0, "completion_tokens": 0,
            "reasoning_tokens": 0, "cache_hit_tokens": 0, "error": last_err}


def correct_digits(expected, got):
    if got is None or expected == 0:
        return None
    rel = abs(got - expected) / abs(expected)
    return 15.0 if rel == 0 else min(15.0, -math.log10(rel))


def summarize(rows, meta):
    ok = [r for r in rows if r["got"] is not None]
    fails = [r for r in rows if r["got"] is None]
    abs_err = [abs(r["got"] - r["expected"]) for r in ok]
    rel_err = [abs(r["got"] - r["expected"]) / abs(r["expected"]) for r in ok]
    digits = [correct_digits(r["expected"], r["got"]) for r in ok]
    lat = [r["latency_s"] for r in ok if r["latency_s"] is not None]

    def worst(keys):
        if not ok:
            return None
        i = max(range(len(ok)), key=lambda j: keys[j])
        return {"x": ok[i]["x"], "expected": ok[i]["expected"], "got": ok[i]["got"],
                "abs_err": abs_err[i], "rel_err": rel_err[i], "raw": ok[i]["raw"]}

    s = {"meta": meta, "points": len(rows), "answered": len(ok), "unparsed_or_failed": len(fails),
         "exact_float_matches": sum(1 for r in ok if r["got"] == r["expected"]),
         "max_abs_err": max(abs_err) if abs_err else None,
         "mean_abs_err": sum(abs_err) / len(abs_err) if abs_err else None,
         "rms_abs_err": math.sqrt(sum(e * e for e in abs_err) / len(abs_err)) if abs_err else None,
         "max_rel_err": max(rel_err) if rel_err else None,
         "mean_rel_err": sum(rel_err) / len(rel_err) if rel_err else None,
         "min_correct_digits": min(digits) if digits else None,
         "mean_correct_digits": sum(digits) / len(digits) if digits else None,
         "median_correct_digits": sorted(digits)[len(digits) // 2] if digits else None,
         "digit_histogram": {},
         "worst_abs": worst(abs_err), "worst_rel": worst(rel_err),
         "latency_s": {"mean": sum(lat) / len(lat) if lat else None,
                       "min": min(lat) if lat else None, "max": max(lat) if lat else None},
         "tokens": {"prompt": sum(r["prompt_tokens"] for r in rows),
                    "completion": sum(r["completion_tokens"] for r in rows),
                    "reasoning": sum(r["reasoning_tokens"] for r in rows),
                    "cache_hit": sum(r["cache_hit_tokens"] for r in rows)},
         "failures": [{"x": r["x"], "error": r["error"], "raw": r["raw"]} for r in fails]}
    for d in digits:
        b = "15 (exact)" if d >= 15 else (f"{int(d)}" if d >= 0 else "<0 (gross)")
        s["digit_histogram"][b] = s["digit_histogram"].get(b, 0) + 1
    return s


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--func", default="exp", choices=sorted(FUNCS))
    p.add_argument("--base-url", default="https://api.deepseek.com/v1")
    p.add_argument("--model", default="deepseek-v4-flash")
    p.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    p.add_argument("--api-key", default=None)
    p.add_argument("--lo", type=float, default=-1.0)
    p.add_argument("--hi", type=float, default=1.0)
    p.add_argument("--step", type=float, default=0.02)
    p.add_argument("--jitter", type=float, default=0.0095, help="max |offset| applied to each point")
    p.add_argument("--seed", default="test1v2", help="grid seed — same seed = same arguments")
    p.add_argument("--sigdigits", type=int, default=15)
    p.add_argument("--no-antitrunc", action="store_true",
                   help="drop the 'do not truncate' clause (reproduces the v1 prompt)")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=0)
    p.add_argument("--reasoning-effort", default=None)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--timeout", type=float, default=900)
    p.add_argument("--retries", type=int, default=2)
    p.add_argument("--tag", default=None)
    p.add_argument("--out-root",
                   default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
    args = p.parse_args()

    key = args.api_key or os.environ.get(args.api_key_env)
    if not key:
        sys.exit(f"no API key: set ${args.api_key_env} or pass --api-key")

    base = recall_proof_grid(args.lo, args.hi, args.step, args.jitter, args.seed)
    xs = [round(FUNCS[args.func][2](t), 10) for t in base]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    tag = args.tag or f"{args.func}_{args.model.replace('/', '_').replace(':', '-')}_{stamp}"
    outdir = os.path.join(args.out_root, tag)
    os.makedirs(outdir, exist_ok=True)

    print(f"test1v2/{args.func}: {len(xs)} recall-proof points "
          f"[{min(xs)} .. {max(xs)}], model={args.model} temp={args.temperature} "
          f"concurrency={args.concurrency}")
    print(f"  sample args: {xs[:3]} …")
    t0 = time.time()
    rows = [None] * len(xs)
    done = 0
    with cf.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = {ex.submit(ask, args, key, x): i for i, x in enumerate(xs)}
        for f in cf.as_completed(futs):
            rows[futs[f]] = f.result()
            done += 1
            if done % 10 == 0 or done == len(xs):
                print(f"  {done}/{len(xs)} done ({time.time()-t0:.0f}s)", flush=True)
    wall = time.time() - t0

    meta = {"test": "test1v2", "func": args.func, "model": args.model, "base_url": args.base_url,
            "temperature": args.temperature, "reasoning_effort": args.reasoning_effort,
            "concurrency": args.concurrency, "recall_proof": True, "seed": args.seed,
            "jitter": args.jitter, "sigdigits_requested": args.sigdigits,
            "antitruncation_clause": not args.no_antitrunc,
            "grid": {"lo": args.lo, "hi": args.hi, "step": args.step, "points": len(xs)},
            "wall_s": round(wall, 1), "utc": stamp, "system_prompt": SYSTEM,
            "user_prompt_example": build_user(args, xs[0])}

    with open(os.path.join(outdir, "raw.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    with open(os.path.join(outdir, "results.csv"), "w") as f:
        f.write("x,expected,got,abs_err,rel_err,correct_digits,latency_s,"
                "completion_tokens,reasoning_tokens,error\n")
        for r in rows:
            ae = "" if r["got"] is None else f'{abs(r["got"]-r["expected"]):.6e}'
            re_ = "" if r["got"] is None else f'{abs(r["got"]-r["expected"])/abs(r["expected"]):.6e}'
            cd = correct_digits(r["expected"], r["got"])
            f.write(f'{r["x"]},{r["expected"]:.17g},{"" if r["got"] is None else repr(r["got"])},'
                    f'{ae},{re_},{"" if cd is None else f"{cd:.2f}"},{r["latency_s"] or ""},'
                    f'{r["completion_tokens"]},{r["reasoning_tokens"]},'
                    f'{(r["error"] or "").replace(chr(44), " ")}\n')
    s = summarize(rows, meta)
    with open(os.path.join(outdir, "summary.json"), "w") as f:
        json.dump(s, f, indent=2)

    print(f"\n== test1v2 {tag}")
    print(f"  answered            {s['answered']}/{s['points']}  (failed {s['unparsed_or_failed']})")
    print(f"  exact float matches {s['exact_float_matches']}")
    if s["min_correct_digits"] is not None:
        print(f"  correct digits      min {s['min_correct_digits']:.2f} · "
              f"median {s['median_correct_digits']:.2f} · mean {s['mean_correct_digits']:.2f}")
        print(f"  max abs err         {s['max_abs_err']:.3e}   max rel err {s['max_rel_err']:.3e}")
        w = s["worst_abs"]
        print(f"  worst point         x={w['x']} expected={w['expected']:.17g} got={w['got']!r}")
    print(f"  wall {wall:.0f}s  completion_tokens {s['tokens']['completion']} "
          f"(reasoning {s['tokens']['reasoning']})")
    print(f"  -> {outdir}")


if __name__ == "__main__":
    main()
