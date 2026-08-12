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


def post(url, headers, payload, timeout):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers=headers, method="POST")
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


def build_anthropic_payload(args, x):
    """Native Anthropic Messages API body (POST {base}/v1/messages).

    Differences from the OpenAI-compatible path, per the current API:
      * `system` is a top-level field, not a message with role="system";
      * `max_tokens` is REQUIRED and caps thinking + visible text together;
      * `temperature` is rejected (400) on current models - we never send it here;
      * thinking depth is `output_config.effort`, not `reasoning_effort`;
        `thinking.budget_tokens` is removed on current models (400).
      * `display: "summarized"` is needed to get any reasoning text back at all -
        the default ("omitted") streams thinking blocks with empty content.
    """
    payload = {"model": args.model,
               "max_tokens": args.max_tokens or 16000,
               "system": SYSTEM,
               "messages": [{"role": "user", "content": build_user(args, x)}]}
    eff = (args.reasoning_effort or "").lower()
    if eff in ("none", "off", "disabled"):
        payload["thinking"] = {"type": "disabled"}
    else:
        payload["thinking"] = {"type": "adaptive", "display": "summarized"}
        if eff:
            payload["output_config"] = {"effort": eff}
    return payload


def parse_anthropic(resp):
    """-> (answer_text, reasoning_text, usage_dict). Content is a block list."""
    text, thinking = [], []
    for b in resp.get("content") or []:
        if not isinstance(b, dict):
            continue
        if b.get("type") == "text":
            text.append(b.get("text") or "")
        elif b.get("type") == "thinking":
            thinking.append(b.get("thinking") or "")
    u = resp.get("usage", {}) or {}
    return ("".join(text) or None, "".join(thinking) or None, u)


# --------------------------------------------------------------------------
# claude-cli backend: drive the Claude Code CLI instead of an HTTP API.
# Uses the machine's Claude Code credentials, so it needs no ANTHROPIC_API_KEY.
# Two isolation requirements, both enforced here:
#   * ZERO TOOLS      - `--tools ""` disables the entire built-in tool set, so
#                       the model cannot compute the answer with bash/python.
#   * NO CLAUDE.md    - `claude` walks cwd->root for CLAUDE.md and reads
#                       $HOME/.claude/CLAUDE.md. We run in an empty temp dir
#                       under a temp HOME holding only a copy of the
#                       credentials, so no project or user memory is loaded.
# `--system-prompt` (not --append-system-prompt) REPLACES Claude Code's agent
# prompt with ours, and --exclude-dynamic-system-prompt-sections drops the
# per-machine cwd/env/memory-path preamble.
# --------------------------------------------------------------------------
CLAUDE_ENV = {"home": None, "cwd": None}


def claude_cli_env():
    """Create (once per run) the throwaway HOME + cwd the CLI is invoked in."""
    if CLAUDE_ENV["home"]:
        return CLAUDE_ENV["home"], CLAUDE_ENV["cwd"]
    import shutil, tempfile
    root = tempfile.mkdtemp(prefix="test1v2_claude_")
    home, cwd = os.path.join(root, "home"), os.path.join(root, "ws")
    os.makedirs(os.path.join(home, ".claude"), exist_ok=True)
    os.makedirs(cwd, exist_ok=True)
    src = os.path.expanduser("~/.claude/.credentials.json")
    if os.path.exists(src):
        shutil.copy2(src, os.path.join(home, ".claude", ".credentials.json"))
    with open(os.path.join(home, ".claude.json"), "w") as f:
        f.write('{"hasCompletedOnboarding":true,"theme":"dark"}\n')
    CLAUDE_ENV["home"], CLAUDE_ENV["cwd"] = home, cwd
    return home, cwd


def ask_claude_cli(args, x):
    """One stateless `claude -p` invocation. Returns the same row shape as ask()."""
    import subprocess
    ref = FUNCS[args.func][0]
    home, cwd = claude_cli_env()
    cmd = [args.claude_bin, "-p", build_user(args, x),
           "--system-prompt", SYSTEM,
           "--tools", "",                       # no tools at all
           "--output-format", "json",
           "--exclude-dynamic-system-prompt-sections"]
    if args.model:
        cmd += ["--model", args.model]
    if args.reasoning_effort:
        # CLI exposes the same ladder as the API: low|medium|high|xhigh|max
        cmd += ["--effort", args.reasoning_effort]
    env = dict(os.environ, HOME=home, CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC="1")
    env.pop("ANTHROPIC_API_KEY", None)          # use the CLI's own credentials
    last_err = None
    for attempt in range(args.retries + 1):
        t0 = time.time()
        try:
            r = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True,
                               text=True, timeout=args.timeout)
            dt = time.time() - t0
            if r.returncode != 0:
                last_err = f"claude rc={r.returncode}: {r.stderr.strip()[:200]}"
                time.sleep(1.5 * (attempt + 1)); continue
            d = json.loads(r.stdout)
            u = d.get("usage", {}) or {}
            content = d.get("result") if not d.get("is_error") else None
            return {"x": x, "expected": ref(x), "got": parse_number(content),
                    "raw": content, "reasoning": None,
                    "latency_s": round(dt, 3), "attempt": attempt,
                    "prompt_tokens": u.get("input_tokens", 0),
                    "completion_tokens": u.get("output_tokens", 0),
                    # `claude -p` does not expose thinking text or a separate
                    # reasoning-token count; thinking is inside output_tokens.
                    "reasoning_tokens": 0,
                    "cache_hit_tokens": u.get("cache_read_input_tokens", 0),
                    "cost_usd": d.get("total_cost_usd"),
                    "stop_reason": d.get("stop_reason"),
                    # num_turns == 1 proves no tool round-trip happened (a tool call
                    # needs >= 2 turns). Recorded per point so the zero-tools claim
                    # is auditable from raw.jsonl rather than taken on trust.
                    "num_turns": d.get("num_turns"),
                    "permission_denials": len(d.get("permission_denials") or []),
                    "error": d.get("api_error_status") if d.get("is_error") else None}
        except Exception as e:                                   # noqa: BLE001
            last_err = f"{type(e).__name__}: {e}"
            time.sleep(1.5 * (attempt + 1))
    return {"x": x, "expected": ref(x), "got": None, "raw": None, "reasoning": None,
            "latency_s": None, "attempt": args.retries, "prompt_tokens": 0,
            "completion_tokens": 0, "reasoning_tokens": 0, "cache_hit_tokens": 0,
            "error": last_err}


def ask(args, key, x):
    if args.api == "claude-cli":
        return ask_claude_cli(args, x)
    ref = FUNCS[args.func][0]
    anthropic = args.api == "anthropic"
    if anthropic:
        payload = build_anthropic_payload(args, x)
        url = args.base_url.rstrip("/") + "/v1/messages"
        headers = {"Content-Type": "application/json", "x-api-key": key,
                   "anthropic-version": "2023-06-01"}
    else:
        payload = {"model": args.model,
                   "messages": [{"role": "system", "content": SYSTEM},
                                {"role": "user", "content": build_user(args, x)}]}
        if not args.no_temperature:
            payload["temperature"] = args.temperature
        if args.max_tokens:
            payload["max_tokens"] = args.max_tokens
        if args.reasoning_effort:
            payload["reasoning_effort"] = args.reasoning_effort
        url = args.base_url.rstrip("/") + "/chat/completions"
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {key}"}

    last_err = None
    for attempt in range(args.retries + 1):
        t0 = time.time()
        try:
            resp = post(url, headers, payload, args.timeout)
            dt = time.time() - t0
            if anthropic:
                content, reasoning, u = parse_anthropic(resp)
                # A safety refusal is HTTP 200 with stop_reason="refusal" and no
                # usable content - record it as a failure, not as a wrong value.
                err = None
                if resp.get("stop_reason") in ("refusal", "max_tokens") and not content:
                    err = f'stop_reason={resp.get("stop_reason")}'
                return {"x": x, "expected": ref(x), "got": parse_number(content),
                        "raw": content, "reasoning": reasoning,
                        "latency_s": round(dt, 3), "attempt": attempt,
                        "prompt_tokens": u.get("input_tokens", 0),
                        "completion_tokens": u.get("output_tokens", 0),
                        # thinking is billed inside output_tokens; the API reports
                        # no separate reasoning-token count.
                        "reasoning_tokens": 0,
                        "cache_hit_tokens": u.get("cache_read_input_tokens", 0),
                        "stop_reason": resp.get("stop_reason"), "error": err}
            msg = (resp.get("choices") or [{}])[0].get("message", {}) or {}
            usage = resp.get("usage", {}) or {}
            ptok = usage.get("prompt_tokens", 0)
            ctok = usage.get("completion_tokens", 0)
            rtok = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0)
            ttok = usage.get("total_tokens", 0)
            # Some hosts bill thinking inside total_tokens but expose neither reasoning_tokens
            # nor a completion_tokens that includes it (near.ai + Gemini: 113 + 18 visible,
            # total 15,126). Recover the hidden spend from the total rather than under-reporting
            # the point by ~800x, and record it so the inference is visible in the data.
            hidden = ttok - ptok - ctok if ttok else 0
            if hidden > 0 and rtok == 0:
                rtok = hidden
                ctok += hidden
            return {"x": x, "expected": ref(x), "got": parse_number(msg.get("content")),
                    "raw": msg.get("content"),
                    "reasoning": msg.get("reasoning_content") or msg.get("reasoning"),
                    "latency_s": round(dt, 3), "attempt": attempt,
                    "prompt_tokens": ptok,
                    "completion_tokens": ctok,
                    "reasoning_tokens": rtok,
                    "total_tokens": ttok,
                    "hidden_reasoning_tokens": max(hidden, 0),
                    "cache_hit_tokens": usage.get("prompt_cache_hit_tokens", 0),
                    "error": None}
        except Exception as e:                                   # noqa: BLE001 - record, retry
            last_err = f"{type(e).__name__}: {e}"
            wait = 1.5 * (attempt + 1)
            if isinstance(e, urllib.error.HTTPError):
                try: last_err += " :: " + e.read().decode()[:300]
                except Exception: pass
                if e.code == 429:
                    # rate limited: honour Retry-After when given, else back off hard.
                    # Free/shared endpoints 429 constantly; a 1.5s retry just burns the budget.
                    ra = None
                    try: ra = float(e.headers.get("Retry-After", ""))
                    except Exception: pass
                    wait = ra if ra else min(120.0, 10.0 * (attempt + 1) ** 2)
            time.sleep(wait)
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



def render_plots(outdir):
    """Render the standard figures. Needs matplotlib; the repo keeps it in .venv-plot."""
    import subprocess, sys
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.dirname(here) if os.path.basename(here) != "LLM-Tests" else here
    script = os.path.join(repo, "test1", "plot_results.py")
    venv = os.path.join(repo, ".venv-plot", "bin", "python")
    python = venv if os.path.exists(venv) else sys.executable
    if not os.path.exists(script):
        print(f"  (no plot script at {script} - skipping figures)")
        return
    for extra in ([], ["--dark"]):
        r = subprocess.run([python, script, outdir] + extra, capture_output=True, text=True)
        if r.returncode != 0:
            print("  (plotting skipped: " + (r.stderr.strip().splitlines() or ["?"])[-1][:120] + ")")
            print(f"  render manually: {venv} {script} {outdir}")
            return
    print(f"  plots -> {os.path.join(outdir, 'plots')}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--func", default="exp", choices=sorted(FUNCS))
    p.add_argument("--api", default="openai", choices=("openai", "anthropic", "claude-cli"),
                   help="openai: OpenAI-compatible /chat/completions (default); "
                        "anthropic: native /v1/messages; claude-cli: exec the Claude Code "
                        "CLI with zero tools and no CLAUDE.md (no API key needed)")
    p.add_argument("--claude-bin", default="claude", help="claude-cli: CLI binary")
    p.add_argument("--base-url", default=None,
                   help="default: https://api.deepseek.com/v1 (openai) or "
                        "https://api.anthropic.com (anthropic)")
    p.add_argument("--model", default=None,
                   help="default: deepseek-v4-flash (openai) or claude-opus-5 (anthropic)")
    p.add_argument("--api-key-env", default=None,
                   help="default: DEEPSEEK_API_KEY (openai) or ANTHROPIC_API_KEY (anthropic)")
    p.add_argument("--api-key", default=None)
    p.add_argument("--lo", type=float, default=-1.0)
    p.add_argument("--hi", type=float, default=1.0)
    p.add_argument("--step", type=float, default=0.02)
    p.add_argument("--jitter", type=float, default=0.0095, help="max |offset| applied to each point")
    p.add_argument("--seed", default="test1v2", help="grid seed — same seed = same arguments")
    p.add_argument("--sample", type=int, default=0,
                   help="run a deterministic random subset of N grid points (0 = all)")
    p.add_argument("--no-temperature", action="store_true",
                   help="omit `temperature` entirely — current Claude models reject it")
    p.add_argument("--sigdigits", type=int, default=15)
    p.add_argument("--no-antitrunc", action="store_true",
                   help="drop the 'do not truncate' clause (reproduces the v1 prompt)")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=0)
    p.add_argument("--reasoning-effort", default=None)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--timeout", type=float, default=900)
    p.add_argument("--retries", type=int, default=2)
    p.add_argument("--no-plot", action="store_true",
                   help="skip rendering figures at the end of the run")
    p.add_argument("--tag", default=None)
    p.add_argument("--out-root",
                   default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
    args = p.parse_args()
    if args.api == "claude-cli":
        args.base_url = args.base_url or "claude-code-cli (local)"
        args.api_key_env = args.api_key_env or "-"
        if args.temperature:
            sys.exit("--temperature is not supported by the claude CLI backend")
    elif args.api == "anthropic":
        args.base_url = args.base_url or "https://api.anthropic.com"
        args.model = args.model or "claude-opus-5"
        args.api_key_env = args.api_key_env or "ANTHROPIC_API_KEY"
        if args.temperature:
            # current Claude models reject temperature/top_p/top_k with a 400
            sys.exit("--temperature is not accepted by the Anthropic API; drop it "
                     "(sampling params were removed on current models)")
    else:
        args.base_url = args.base_url or "https://api.deepseek.com/v1"
        args.model = args.model or "deepseek-v4-flash"
        args.api_key_env = args.api_key_env or "DEEPSEEK_API_KEY"

    key = args.api_key or os.environ.get(args.api_key_env)
    if args.api == "claude-cli":
        key = key or "cli"          # credentials come from the CLI, not an env var
    if not key:
        sys.exit(f"no API key: set ${args.api_key_env} or pass --api-key")

    base = recall_proof_grid(args.lo, args.hi, args.step, args.jitter, args.seed)
    xs = [round(FUNCS[args.func][2](t), 10) for t in base]
    if args.sample and args.sample < len(xs):
        # deterministic subset: rank points by a seeded hash, take the first N,
        # then restore grid order so plots stay left-to-right.
        rank = sorted(xs, key=lambda v: hashlib.sha256(f"{args.seed}:sample:{v}".encode()).digest())
        xs = sorted(rank[:args.sample])
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
            "api": args.api,
            "temperature": args.temperature, "reasoning_effort": args.reasoning_effort,
            "concurrency": args.concurrency, "recall_proof": True, "seed": args.seed,
            "jitter": args.jitter, "sigdigits_requested": args.sigdigits,
            "antitruncation_clause": not args.no_antitrunc,
            "grid": {"lo": args.lo, "hi": args.hi, "step": args.step, "points": len(xs),
                     "sampled_from": len(base) if args.sample else None},
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

    if not args.no_plot:
        render_plots(outdir)



if __name__ == "__main__":
    main()
