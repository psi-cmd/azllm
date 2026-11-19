"""Fast check for one optimisation round: one fixed greedy request against a running tqserve.

  fastloop.py URL REF [--save] [--reps 3] [--max-tokens 1024] [--prompt FILE]

Sends the same request --reps times (after a short warm-up). Every rep must produce exactly the same text as the
others and, unless --save, as REF (saved once from a known-good build). Prints decode tok/s (best and median), the
accepted-draft length, and exits 1 on any mismatch. Greedy output is bit-identical across tree widths and launch
groupings by design, so one long request catches a kernel that changed a single logit's argmax.
"""
import json, os, statistics, sys, time, urllib.request

args = sys.argv[1:]
url, ref = args[0], args[1]
opt = lambda name, dflt: args[args.index(name) + 1] if name in args else dflt
save, reps, max_tokens = "--save" in args, int(opt("--reps", "3")), int(opt("--max-tokens", "1024"))
temp = float(opt("--temp", "0"))  # > 0: the sampled path (top-p 0.95, top-k 20); exactness only vs its own reference
here = os.path.dirname(os.path.abspath(__file__))
prompt = open(opt("--prompt", os.path.join(here, "data", "fastloop-prompt.txt"))).read()


def send(text, n):
    body = {"model": "x", "messages": [{"role": "user", "content": text}], "max_tokens": n, "temperature": temp,
            **({"top_p": 0.95, "top_k": 20, "seed": 1234} if temp > 0 else {}),
            "chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": "xhigh"}}
    req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=3600).read())


def text_of(r):
    m = r["choices"][0]["message"]
    return (m.get("reasoning_content") or "") + "\n<<content>>\n" + (m.get("content") or "")


send("Hello", 16)
outs, speeds, accept = [], [], []
for _ in range(reps):
    r = send(prompt, max_tokens)
    u, tm = r.get("usage", {}), r.get("timings", {})
    comp, acc = int(u.get("completion_tokens", 0)), int(tm.get("draft_n_accepted", 0))
    outs.append(text_of(r)), speeds.append(float(tm.get("predicted_per_second", 0)))
    accept.append(comp / max(1, comp - acc))

def first_diff(a, b):
    i = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
    return i if i < max(len(a), len(b)) else -1

ok = all(o == outs[0] for o in outs)
msg = "reps agree" if ok else f"reps DIFFER at char {first_diff(outs[0], next(o for o in outs if o != outs[0]))}"
if save:
    with open(ref, "w") as f:
        f.write(outs[0])
    msg += f"; reference saved ({len(outs[0])} chars)"
elif os.path.exists(ref):
    want = open(ref).read()
    d = first_diff(outs[0], want)
    ok &= d < 0
    msg += "; matches reference" if d < 0 else f"; DIFFERS from reference at char {d} of {len(want)}"
else:
    ok = False
    msg += "; no reference (run once with --save on a known-good build)"
print(f"{'OK  ' if ok else 'FAIL'} {msg}")
print(f"tokens {comp}  accept_len {statistics.median(accept):.2f}  "
      f"tok/s best {max(speeds):.1f} median {statistics.median(speeds):.1f}  ({reps} reps)")
sys.exit(0 if ok else 1)
