"""Decode-only kernel window for bench/kprof.sh: streams one greedy request and signals the traced server
(SIGUSR2 = reset at the first generated token, i.e. after prefill; SIGUSR1 = dump at the end).
  kprof_client.py URL PID PROMPT_FILE [max_tokens]"""
import json, os, signal, sys, time, urllib.request

url, pid, pf = sys.argv[1], int(sys.argv[2]), sys.argv[3]
n = int(sys.argv[4]) if len(sys.argv) > 4 else 1024
body = {"model": "x", "messages": [{"role": "user", "content": open(pf).read()}], "max_tokens": n, "temperature": 0,
        "stream": True, "chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": "xhigh"}}
req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
t0, first, toks, timings = time.time(), None, 0, {}
with urllib.request.urlopen(req, timeout=7200) as r:
    for raw in r:
        line = raw.decode("utf-8", "replace").strip()
        if not line.startswith("data:") or line == "data: [DONE]":
            continue
        ev = json.loads(line[5:])
        timings = ev.get("timings", timings) or timings
        ch = ev.get("choices") or [{}]
        d = ch[0].get("delta", {})
        if d.get("content") or d.get("reasoning_content"):
            toks += 1
            if first is None:
                first = time.time()
                os.kill(pid, signal.SIGUSR2)
t1 = time.time()
time.sleep(0.3)
os.kill(pid, signal.SIGUSR1)
dec = timings.get("predicted_per_second")
print(f"prefill+first {first - t0:.2f}s  decode {t1 - first:.2f}s  ~{toks} chunks  server decode tok/s {dec}")
