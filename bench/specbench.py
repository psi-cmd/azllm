"""Acceptance and speed over a fixed prompt set against a running tqserve, greedy or sampled (fixed seeds).
  specbench.py URL PROMPTS.json [--per 6] [--temp 0.6] [--max-tokens 1024]
PROMPTS.json: {"dataset": [prompt, ...], ...} (the DFlash2 benchmark set). Pooled accept = tokens / verify steps;
sampled runs differ per path, so compare pooled numbers over many prompts, never one request."""
import json, sys, urllib.request

args = sys.argv[1:]
url, pf = args[0], args[1]
opt = lambda n, d: args[args.index(n) + 1] if n in args else d
per, temp, mt = int(opt("--per", "6")), float(opt("--temp", "0.6")), int(opt("--max-tokens", "1024"))
P = json.load(open(pf))


def send(text, n, seed):
    body = {"model": "x", "messages": [{"role": "user", "content": text}], "max_tokens": n, "temperature": temp,
            "chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": "xhigh"}}
    if temp > 0: body.update(top_p=0.95, top_k=20, seed=seed)
    req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=7200).read())


send("Hello", 16, 1)
tot_tok = tot_steps = 0
tot_time = 0.0
rows = []
for ds, items in P.items():
    tk = st = 0
    tt = 0.0
    for i, text in enumerate(items[:per]):
        r = send(text, mt, 1000 + i)
        u, tm = r.get("usage", {}), r.get("timings", {})
        comp, acc = int(u.get("completion_tokens", 0)), int(tm.get("draft_n_accepted", 0))
        tk += comp
        st += max(1, comp - acc)
        tt += comp / max(1e-9, float(tm.get("predicted_per_second", 0) or 1e-9))
    rows.append((ds, tk / st, tk / tt))
    tot_tok, tot_steps, tot_time = tot_tok + tk, tot_steps + st, tot_time + tt
print("  ".join(f"{d} {a:.2f}/{s:.0f}" for d, a, s in rows))
print(f"pooled accept {tot_tok / tot_steps:.3f}  decode tok/s {tot_tok / tot_time:.1f}  ({tot_tok} tokens)")
