#!/usr/bin/env python3
"""Deep-context probe: prefill a long document, then measure decode speed.

Builds a prompt of roughly the requested token count from a text file (the
first request prefills it, later runs hit the server's prompt cache), asks for
a summary, and reports decode tok/s the same way probe.py does (client-side
clock over streamed tokens, time to first token excluded) plus the server's
own print_timing figures and draft acceptance when present.

Usage:
    python3 deep_probe.py <server_url> <text_file> [--tokens 35000] [--runs 3] [--max-tokens 300]
"""
from __future__ import annotations

import json
import statistics as st
import sys
import time
import urllib.request


def build_prompt(text_file: str, target_tokens: int) -> str:
    # about 4 characters per token for English prose and source code
    text = open(text_file, encoding="utf-8", errors="replace").read()
    need = target_tokens * 4
    while len(text) < need:
        text = text + "\n\n" + text
    text = text[:need]
    return ("Below is a long document. Read it, then write a 250 word summary of its main points "
            "in plain prose.\n\n<document>\n" + text + "\n</document>\n\nSummary:")


def tokenize_count(url: str, prompt: str) -> int:
    req = urllib.request.Request(url.rstrip("/") + "/tokenize", json.dumps({"content": prompt}).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return len(json.load(r)["tokens"])


def run(url: str, prompt: str, max_tokens: int) -> dict:
    body = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
        "timings_per_token": False,
    }
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.time()
    ttft = None
    n = 0
    last = t0
    timings = None
    with urllib.request.urlopen(req, timeout=3600) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            d = json.loads(line[6:])
            if d.get("timings"):
                timings = d["timings"]
            choices = d.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta", {})
            if delta.get("content") or delta.get("reasoning_content"):
                now = time.time()
                if ttft is None:
                    ttft = now - t0
                last = now
                n += 1
    span = last - t0 - (ttft or 0)
    return {"client_tps": n / span if span > 0 else 0.0, "n": n, "ttft_s": ttft, "timings": timings}


def main(argv: list[str]) -> int:
    args = [a for a in argv if not a.startswith("--")]
    opts = {argv[i]: argv[i + 1] for i in range(len(argv) - 1) if argv[i].startswith("--")}
    if len(args) < 2:
        print(__doc__)
        return 2
    url, text_file = args[0], args[1]
    target = int(opts.get("--tokens", 35000))
    runs = int(opts.get("--runs", 3))
    max_tokens = int(opts.get("--max-tokens", 300))
    prompt = build_prompt(text_file, target)
    n_tok = tokenize_count(url, prompt)
    print(f"prompt: {len(prompt)} chars, {n_tok} tokens (server /tokenize)")
    results = []
    for i in range(runs):
        r = run(url, prompt, max_tokens)
        t = r["timings"] or {}
        acc = t.get("draft_n_accepted")
        gen = t.get("draft_n")
        extra = f" draft {acc}/{gen} ({acc / gen:.3f})" if gen else ""
        print(f"run {i + 1}: client {r['client_tps']:.2f} tok/s over {r['n']} deltas, ttft {r['ttft_s']:.1f} s, "
              f"server predicted_per_second {t.get('predicted_per_second', 0):.2f} tok/s ({t.get('predicted_n')} tokens), "
              f"prompt_n {t.get('prompt_n')} cache_n {t.get('cache_n')}{extra}")
        results.append(r)
    ctps = [r["client_tps"] for r in results]
    stps = [(r["timings"] or {}).get("predicted_per_second", 0) for r in results]
    print(f"median client {st.median(ctps):.2f} tok/s, median server {st.median(stps):.2f} tok/s, depth {n_tok} tokens")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
