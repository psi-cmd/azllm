"""Opt-in real-server regression for prompt reuse (--prompt-cache on, or --kv-compact; --np 1).

TQSERVE_TEST_URL names the running server. A follow-up turn is answered twice: right after the first turn (the shared
prefix is reused) and after an unrelated request has replaced the cache (full prefill). Greedy answers must agree.
test_image_prompt_reuse needs the server started with --mmproj (skipped otherwise).
"""
import base64
import json
import os
import struct
import time
import unittest
import urllib.error
import urllib.request
import zlib

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


def png(w, h, pixel):
    """data: URI of a w x h RGB PNG, pixel(x, y) -> (r, g, b)"""
    rows = b"".join(b"\0" + bytes(c for x in range(w) for c in pixel(x, y)) for y in range(h))
    chunk = lambda t, d: struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d))
    data = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) +
            chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))
    return "data:image/png;base64," + base64.b64encode(data).decode()


def image_msg(uri, text):
    return {"role": "user", "content": [{"type": "image_url", "image_url": {"url": uri}}, {"type": "text", "text": text}]}


def cached(m):
    return m["usage"].get("prompt_tokens_details", {}).get("cached_tokens", 0)


def evict():
    """An unrelated request long enough to take the cache over (at most --prompt-cache-probe tokens, 128 by default,
    is a probe that leaves the cached conversation alone)."""
    m, _ = ask_full([{"role": "user", "content": "Unrelated. " * 200 + "Reply with OK."}], max_tokens=4)
    return m


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
        evict()
        cold_m, t_cold = ask_full(m2)
        cold = cold_m["choices"][0]["message"]["content"] or ""
        self.assertLess(cached(cold_m), 32, "the eviction request did not replace the cache")
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

    def test_image_prompt_reuse(self):
        # Every image cell is the same <|image_pad|> token: the cache must key cells by image content. A follow-up about
        # the same image reuses the image; the same text with a different image must not (it stops at that image).
        red_blue = png(256, 256, lambda x, y: (220, 30, 30) if x < 128 else (30, 30, 220))
        green = png(256, 256, lambda x, y: (30, 200, 60))
        m1 = [image_msg(red_blue, FILLER[:400] + "\nWhat two colors does the image show? One short sentence.")]
        try:
            r1, _ = ask_full(m1)
        except urllib.error.HTTPError as e:
            if e.code in (400, 422, 501):
                self.skipTest("the server has no vision encoder (--mmproj)")
            raise
        p1 = r1["usage"]["prompt_tokens"]
        a1 = r1["choices"][0]["message"]["content"] or ""
        m2 = m1 + [{"role": "assistant", "content": a1}, {"role": "user", "content": "Which color is on the left?"}]
        warm, t_warm = ask_full(m2)
        self.assertGreaterEqual(cached(warm), p1 - 8, "the follow-up did not reuse the first turn and its image")
        other = [image_msg(green, m1[0]["content"][1]["text"])] + m2[1:]
        swapped, _ = ask_full(other)
        self.assertLess(cached(swapped), 32, "a different image reused the cached one's cells")
        evict()
        cold, t_cold = ask_full(m2)
        self.assertLess(cached(cold), 32, "the eviction request did not replace the cache")
        print("image follow-up: %d of %d reused, %.2f s; cold %.2f s; first answer %r" %
              (cached(warm), warm["usage"]["prompt_tokens"], t_warm, t_cold, a1[:60]))
        self.assertEqual(warm["choices"][0]["message"]["content"], cold["choices"][0]["message"]["content"],
                         "image follow-up answered differently with reuse")


if __name__ == "__main__":
    unittest.main()
