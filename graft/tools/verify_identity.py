#!/usr/bin/env python3
"""Identity check for speculative decoding: same prompt, greedy, flag off vs on.

Speculative decoding is verify-then-accept, so at temperature 0 the text must be
identical with and without the draft head. This script runs the three probe
prompts greedily against a live llama-server and saves each answer, then a
second invocation compares the saved answers of two arms byte for byte.

Usage:
    python3 verify_identity.py run     <server_url> <out_dir> <arm>    # arm: off | on
    python3 verify_identity.py compare <out_dir> <arm_a> <arm_b>

Each run writes <out_dir>/<prompt_id>.<arm>.txt (the text) and
<out_dir>/<prompt_id>.<arm>.json (server timings, token count, sha256 of the text).
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import urllib.request

PROMPTS = {
    "code": "write a python function that merges two sorted lists into one sorted list, with docstring.",
    "prose": "explain the difference between mmap and read for loading large files, one paragraph.",
    "bash": "write a bash script that watches a directory and prints new files as they appear.",
}
MAX_TOKENS = 300


def complete(url: str, prompt: str, max_tokens: int = MAX_TOKENS) -> dict:
    body = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "top_k": 1,
        "top_p": 1.0,
        "seed": 42,
        "stream": False,
        "logprobs": True,
        "top_logprobs": 4,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)


def run(url: str, out_dir: str, arm: str) -> int:
    os.makedirs(out_dir, exist_ok=True)
    for pid, prompt in PROMPTS.items():
        d = complete(url, prompt)
        text = d["choices"][0]["message"].get("content") or ""
        reasoning = d["choices"][0]["message"].get("reasoning_content") or ""
        timings = d.get("timings", {})
        lp = ((d["choices"][0].get("logprobs") or {}).get("content")) or []
        tokens = [{"token": x.get("token"), "logprob": x.get("logprob"),
                   "top": [(t.get("token"), t.get("logprob")) for t in x.get("top_logprobs", [])]} for x in lp]
        meta = {
            "prompt_id": pid, "arm": arm, "prompt": prompt, "max_tokens": MAX_TOKENS,
            "finish_reason": d["choices"][0].get("finish_reason"),
            "completion_tokens": d.get("usage", {}).get("completion_tokens"),
            "reasoning_chars": len(reasoning),
            "sha256": hashlib.sha256(text.encode()).hexdigest(),
            "timings": timings,
            "tokens": tokens,
        }
        with open(os.path.join(out_dir, f"{pid}.{arm}.txt"), "w") as f:
            f.write(text)
        with open(os.path.join(out_dir, f"{pid}.{arm}.json"), "w") as f:
            json.dump(meta, f, indent=1)
        tps = timings.get("predicted_per_second")
        acc = timings.get("draft_n_accepted")
        gen = timings.get("draft_n")
        extra = f", draft {acc}/{gen} accepted" if gen else ""
        print(f"{pid:6s} {arm:4s} {meta['completion_tokens']} tokens, {tps:.2f} tok/s (server print_timing){extra}, sha256 {meta['sha256'][:16]}")
    return 0


def compare(out_dir: str, a: str, b: str) -> int:
    rc = 0
    for pid in PROMPTS:
        pa = os.path.join(out_dir, f"{pid}.{a}.txt")
        pb = os.path.join(out_dir, f"{pid}.{b}.txt")
        ta = open(pa, "rb").read()
        tb = open(pb, "rb").read()
        if ta == tb:
            print(f"{pid:6s} IDENTICAL ({len(ta)} bytes, {a} vs {b})")
            continue
        rc = 1
        n = next((i for i, (x, y) in enumerate(zip(ta, tb)) if x != y), min(len(ta), len(tb)))
        print(f"{pid:6s} DIFFERS at byte {n} of {len(ta)}/{len(tb)}: {a}={ta[max(0, n - 30):n + 30]!r} {b}={tb[max(0, n - 30):n + 30]!r}")
        token_diff(out_dir, pid, a, b)
    return rc


def token_diff(out_dir: str, pid: str, a: str, b: str) -> None:
    """Print the first differing token and the top candidates on both arms at that step."""
    try:
        ja = json.load(open(os.path.join(out_dir, f"{pid}.{a}.json")))
        jb = json.load(open(os.path.join(out_dir, f"{pid}.{b}.json")))
    except FileNotFoundError:
        return
    ta, tb = ja.get("tokens") or [], jb.get("tokens") or []
    if not ta or not tb:
        return
    i = next((k for k, (x, y) in enumerate(zip(ta, tb)) if x["token"] != y["token"]), None)
    if i is None:
        print(f"       tokens identical for the first {min(len(ta), len(tb))} steps, lengths {len(ta)} vs {len(tb)}")
        return
    print(f"       first differing token at step {i}:")
    for arm, toks in ((a, ta), (b, tb)):
        top = ", ".join(f"{t!r}:{lp:.4f}" for t, lp in toks[i]["top"])
        print(f"         {arm:4s} chose {toks[i]['token']!r} ({toks[i]['logprob']:.4f}); top: {top}")


def main(argv: list[str]) -> int:
    if len(argv) == 4 and argv[0] == "run":
        return run(argv[1], argv[2], argv[3])
    if len(argv) == 4 and argv[0] == "compare":
        return compare(argv[1], argv[2], argv[3])
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
