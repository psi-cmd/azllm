"""Opt-in real-server regression for prompt reuse (--prompt-cache on, or --kv-compact; --np 1).

TQSERVE_TEST_URL names the running server. A follow-up turn is answered twice: right after the first turn (the shared
prefix is reused) and after an unrelated request has replaced the cache (full prefill). Greedy answers must agree.
"""
import json
import os
import time
import unittest
import urllib.request

URL = os.environ.get("TQSERVE_TEST_URL", "")
FILLER = "The archive lists shipments by date, port and weight; most entries repeat the same three carriers. " * 40


def ask_full(msgs, max_tokens=64, temperature=0):
    body = {"model": "x", "max_tokens": max_tokens, "temperature": temperature, "messages": msgs,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(URL.rstrip("/") + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=900) as r:
        m = json.load(r)
    return m, time.time() - t0


def ask(msgs, max_tokens=64):
    m, dt = ask_full(msgs, max_tokens)
    return m["choices"][0]["message"]["content"] or "", dt


@unittest.skipUnless(URL, "requires a running tqserve with prompt reuse (TQSERVE_TEST_URL)")
class PromptCache(unittest.TestCase):
    def test_follow_up_matches_full_prefill(self):
        m1 = [{"role": "user", "content": FILLER + "\nThe pallet code is 5172.\nQuestion: which three things does the "
                                                   "archive list for each shipment?"}]
        a1, _ = ask(m1)
        m2 = m1 + [{"role": "assistant", "content": a1}, {"role": "user", "content": "What is the pallet code?"}]
        warm, t_warm = ask(m2)
        ask([{"role": "user", "content": "Say hello."}])  # replaces the cached sequence
        cold, t_cold = ask(m2)
        self.assertEqual(warm, cold, "reused-prefix answer differs from the full-prefill answer")
        self.assertIn("5172", warm)
        print("follow-up %.2f s with reuse, %.2f s without" % (t_warm, t_cold))

    def test_rewind_after_many_short_turns(self):
        # An agent loop: a long start, then many short turns, each a request that saves a snapshot. A compaction summary
        # then resends the history up to a message well before the end. The snapshots must still reach back there:
        # evicting the oldest kept only the last 16 turns, and the rewind re-read everything.
        msgs = [{"role": "user", "content": FILLER * 3 + "\nRead the archive, then answer each step briefly."}]
        turns = []
        for i in range(24):
            a, _ = ask(msgs, max_tokens=24)
            msgs = msgs + [{"role": "assistant", "content": a},
                           {"role": "user", "content": ("Step %d: note this log line. " % i) + FILLER[:600]}]
            turns.append(len(msgs))
        ask(msgs, max_tokens=24)
        cut = msgs[:turns[5]] + [{"role": "user", "content": "Summarize the conversation above in one sentence."}]
        m, dt = ask_full(cut, max_tokens=32)
        u = m["usage"]
        cached = u.get("prompt_tokens_details", {}).get("cached_tokens", 0)
        print("rewind: %d of %d prompt tokens reused, %.2f s" % (cached, u["prompt_tokens"], dt))
        self.assertGreater(cached, u["prompt_tokens"] // 2, "the rewind found no snapshot near the cut")

    def test_sampled_first_token_matches_greedy(self):
        # The prompt's last chunk is the few-token generation prompt (split at its <|im_start|> for the message-boundary
        # snapshot). A sampled first token must come from that chunk's LAST position, as greedy's does: reading its
        # first one drew the token after <|im_start|>, so sampled answers began "assistant\n" or "user\n".
        msgs = [{"role": "user", "content": FILLER + "\nIn one short sentence: what does the archive list?"}]
        greedy, _ = ask(msgs, max_tokens=24)
        m, _ = ask_full(msgs, max_tokens=24, temperature=0.01)
        sampled = m["choices"][0]["message"]["content"] or ""
        print("greedy %r / sampled %r" % (greedy[:60], sampled[:60]))
        self.assertFalse(sampled.lstrip().startswith(("assistant", "user", "system")), "sampled answer opens with a role")
        self.assertEqual(sampled.split()[:3], greedy.split()[:3], "near-zero temperature diverges from greedy at once")


if __name__ == "__main__":
    unittest.main()
