#!/usr/bin/env python3
"""test1 — numeric accuracy of exp(x) over a grid, one independent API call per point.

No agent, no tools, no conversation: each grid point is its own stateless chat-completion
request. The model must answer with a bare decimal number; the score is how far that number
is from math.exp(x).

Grid: x = -1.00 .. 1.00 step 0.02  -> 101 points (inclusive of both endpoints).

Usage:
  DEEPSEEK_API_KEY=sk-... ./run_test1.py --model deepseek-v4-flash
  ./run_test1.py --base-url https://api.deepseek.com/v1 --model deepseek-v4-flash \
                 --api-key-env DEEPSEEK_API_KEY --concurrency 8 --temperature 0
Stdlib only.
"""
import argparse, concurrent.futures as cf, json, math, os, re, sys, time, urllib.error, urllib.request
from datetime import datetime, timezone

SYSTEM = ("You are a numerical calculator. Answer with the numeric result only: "
          "a single decimal number, no words, no units, no formatting, no explanation.")
USER = ("Compute exp({x}), the exponential function e raised to the power {x}. "
        "Give the value as a plain decimal number with at least 12 significant digits. "
        "Output only the number.")

NUM_RE = re.compile(r'[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?')


def grid(lo, hi, step):
    n = int(round((hi - lo) / step))
    return [round(lo + i * step, 10) for i in range(n + 1)]


def post(url, key, payload, timeout):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
        method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def parse_number(text):
    """Last standalone number in the reply (models sometimes prefix a restatement)."""
    if text is None:
        return None
    m = NUM_RE.findall(text.replace(",", ""))
    if not m:
        return None
    try:
        return float(m[-1])
    except ValueError:
        return None


def ask(args, key, x):
    payload = {"model": args.model, "temperature": args.temperature,
               "messages": [{"role": "system", "content": SYSTEM},
                            {"role": "user", "content": USER.format(x=f"{x:.2f}")}]}
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
            content = msg.get("content")
            usage = resp.get("usage", {}) or {}
            got = parse_number(content)
            return {"x": x, "expected": math.exp(x), "got": got, "raw": content,
                    "reasoning": msg.get("reasoning_content"),
                    "latency_s": round(dt, 3), "attempt": attempt,
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "reasoning_tokens": ((usage.get("completion_tokens_details") or {})
                                         .get("reasoning_tokens", 0)),
                    "cache_hit_tokens": usage.get("prompt_cache_hit_tokens", 0),
                    "error": None}
        except Exception as e:                                  # noqa: BLE001 - report, then retry
            last_err = f"{type(e).__name__}: {e}"
            if isinstance(e, urllib.error.HTTPError):
                try: last_err += " :: " + e.read().decode()[:300]
                except Exception: pass
            time.sleep(1.5 * (attempt + 1))
    return {"x": x, "expected": math.exp(x), "got": None, "raw": None, "reasoning": None,
            "latency_s": None, "attempt": args.retries, "prompt_tokens": 0,
            "completion_tokens": 0, "reasoning_tokens": 0, "cache_hit_tokens": 0,
            "error": last_err}


def correct_digits(expected, got):
    """Significant decimal digits that agree: -log10(relative error), capped at 15."""
    if got is None or expected == 0:
        return None
    rel = abs(got - expected) / abs(expected)
    if rel == 0:
        return 15.0
    return min(15.0, -math.log10(rel))


def summarize(rows, meta):
    ok = [r for r in rows if r["got"] is not None]
    fails = [r for r in rows if r["got"] is None]
    abs_err = [abs(r["got"] - r["expected"]) for r in ok]
    rel_err = [abs(r["got"] - r["expected"]) / abs(r["expected"]) for r in ok]
    digits = [correct_digits(r["expected"], r["got"]) for r in ok]
    exact = sum(1 for r in ok if r["got"] == r["expected"])
    lat = [r["latency_s"] for r in ok if r["latency_s"] is not None]

    def worst(key_list):
        i = max(range(len(ok)), key=lambda j: key_list[j]) if ok else None
        return None if i is None else {"x": ok[i]["x"], "expected": ok[i]["expected"],
                                       "got": ok[i]["got"], "abs_err": abs_err[i],
                                       "rel_err": rel_err[i]}
    s = {
        "meta": meta,
        "points": len(rows), "answered": len(ok), "unparsed_or_failed": len(fails),
        "exact_float_matches": exact,
        "max_abs_err": max(abs_err) if abs_err else None,
        "mean_abs_err": sum(abs_err) / len(abs_err) if abs_err else None,
        "rms_abs_err": math.sqrt(sum(e * e for e in abs_err) / len(abs_err)) if abs_err else None,
        "max_rel_err": max(rel_err) if rel_err else None,
        "mean_rel_err": sum(rel_err) / len(rel_err) if rel_err else None,
        "min_correct_digits": min(digits) if digits else None,
        "mean_correct_digits": sum(digits) / len(digits) if digits else None,
        "digit_histogram": {},
        "worst_abs": worst(abs_err), "worst_rel": worst(rel_err),
        "latency_s": {"mean": sum(lat) / len(lat) if lat else None,
                      "min": min(lat) if lat else None, "max": max(lat) if lat else None},
        "tokens": {"prompt": sum(r["prompt_tokens"] for r in rows),
                   "completion": sum(r["completion_tokens"] for r in rows),
                   "reasoning": sum(r["reasoning_tokens"] for r in rows),
                   "cache_hit": sum(r["cache_hit_tokens"] for r in rows)},
        "failures": [{"x": r["x"], "error": r["error"], "raw": r["raw"]} for r in fails],
    }
    for d in digits:
        b = "15 (exact)" if d >= 15 else f"{int(d)}"
        s["digit_histogram"][b] = s["digit_histogram"].get(b, 0) + 1
    return s


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="https://api.deepseek.com/v1")
    p.add_argument("--model", default="deepseek-v4-flash")
    p.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    p.add_argument("--api-key", default=None)
    p.add_argument("--lo", type=float, default=-1.0)
    p.add_argument("--hi", type=float, default=1.0)
    p.add_argument("--step", type=float, default=0.02)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=0)
    p.add_argument("--reasoning-effort", default=None,
                   help="passed through as reasoning_effort if the provider supports it")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--timeout", type=float, default=180)
    p.add_argument("--retries", type=int, default=2)
    p.add_argument("--tag", default=None, help="output dir name (default <model>_<UTC stamp>)")
    p.add_argument("--out-root", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
    args = p.parse_args()

    key = args.api_key or os.environ.get(args.api_key_env)
    if not key:
        sys.exit(f"no API key: set ${args.api_key_env} or pass --api-key")

    xs = grid(args.lo, args.hi, args.step)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    tag = args.tag or f"{args.model.replace('/', '_')}_{stamp}"
    outdir = os.path.join(args.out_root, tag)
    os.makedirs(outdir, exist_ok=True)

    print(f"test1: exp(x) on {len(xs)} points [{args.lo}..{args.hi} step {args.step}], "
          f"model={args.model} temp={args.temperature} concurrency={args.concurrency}")
    t0 = time.time()
    rows = [None] * len(xs)
    done = 0
    with cf.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = {ex.submit(ask, args, key, x): i for i, x in enumerate(xs)}
        for f in cf.as_completed(futs):
            i = futs[f]
            rows[i] = f.result()
            done += 1
            if done % 10 == 0 or done == len(xs):
                print(f"  {done}/{len(xs)} done ({time.time()-t0:.0f}s)", flush=True)
    wall = time.time() - t0

    meta = {"model": args.model, "base_url": args.base_url, "temperature": args.temperature,
            "reasoning_effort": args.reasoning_effort, "concurrency": args.concurrency,
            "grid": {"lo": args.lo, "hi": args.hi, "step": args.step, "points": len(xs)},
            "wall_s": round(wall, 1), "utc": stamp, "system_prompt": SYSTEM,
            "user_prompt_template": USER}

    with open(os.path.join(outdir, "raw.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    with open(os.path.join(outdir, "results.csv"), "w") as f:
        f.write("x,expected,got,abs_err,rel_err,correct_digits,latency_s,completion_tokens,reasoning_tokens,error\n")
        for r in rows:
            ae = "" if r["got"] is None else f'{abs(r["got"]-r["expected"]):.6e}'
            re_ = "" if r["got"] is None else f'{abs(r["got"]-r["expected"])/abs(r["expected"]):.6e}'
            cd = correct_digits(r["expected"], r["got"])
            f.write(f'{r["x"]},{r["expected"]:.15g},{"" if r["got"] is None else repr(r["got"])},'
                    f'{ae},{re_},{"" if cd is None else f"{cd:.2f}"},'
                    f'{r["latency_s"] or ""},{r["completion_tokens"]},{r["reasoning_tokens"]},'
                    f'{(r["error"] or "").replace(chr(44)," ")}\n')
    s = summarize(rows, meta)
    with open(os.path.join(outdir, "summary.json"), "w") as f:
        json.dump(s, f, indent=2)

    print(f"\n== test1 {tag}")
    print(f"  answered            {s['answered']}/{s['points']}  (failed/unparsed {s['unparsed_or_failed']})")
    print(f"  exact float matches {s['exact_float_matches']}")
    print(f"  max abs err         {s['max_abs_err']:.3e}" if s['max_abs_err'] is not None else "")
    print(f"  mean abs err        {s['mean_abs_err']:.3e}" if s['mean_abs_err'] is not None else "")
    print(f"  rms abs err         {s['rms_abs_err']:.3e}" if s['rms_abs_err'] is not None else "")
    print(f"  max rel err         {s['max_rel_err']:.3e}" if s['max_rel_err'] is not None else "")
    print(f"  min correct digits  {s['min_correct_digits']:.2f}" if s['min_correct_digits'] is not None else "")
    print(f"  mean correct digits {s['mean_correct_digits']:.2f}" if s['mean_correct_digits'] is not None else "")
    if s["worst_abs"]:
        w = s["worst_abs"]
        print(f"  worst point         x={w['x']} expected={w['expected']:.15g} got={w['got']!r} "
              f"abs={w['abs_err']:.3e}")
    print(f"  wall {wall:.0f}s  completion_tokens {s['tokens']['completion']} "
          f"(reasoning {s['tokens']['reasoning']})")
    print(f"  -> {outdir}")


if __name__ == "__main__":
    main()
