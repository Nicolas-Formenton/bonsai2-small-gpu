#!/usr/bin/env python3
"""Show how the next-token distribution depends on the decode batch size.

Speculative decoding verifies drafts by running the target model on 2 or 3
tokens at once, while plain decoding runs it on 1 token at a time. llama.cpp
picks different CUDA kernels for different batch sizes, so the logits are not
bit-identical across paths. This script makes that visible without any
speculative machinery: it takes a greedy transcript saved by
verify_identity.py, replays the prompt plus the first N generated tokens
through the server's prompt cache so that exactly 1, 2, 3, ... new tokens are
evaluated in the final call, and prints the top candidates the model sees for
token N on each path.

Usage:
    python3 batch_numerics.py <server_url> <identity_dir> <arm> <prompt_id> <step> [batch sizes, default 1 2 3 8]

Example:
    python3 batch_numerics.py http://127.0.0.1:8899 results/identity off code 52
"""
from __future__ import annotations

import json
import sys
import urllib.request


def post(url: str, path: str, body: dict) -> dict:
    req = urllib.request.Request(url.rstrip("/") + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)


def main(argv: list[str]) -> int:
    if len(argv) < 5:
        print(__doc__)
        return 2
    url, ident_dir, arm, pid, step = argv[0], argv[1], argv[2], argv[3], int(argv[4])
    sizes = [int(x) for x in argv[5:]] or [1, 2, 3, 8]
    meta = json.load(open(f"{ident_dir}/{pid}.{arm}.json"))
    tokens = meta["tokens"]
    tpl = post(url, "/apply-template", {"messages": [{"role": "user", "content": meta["prompt"]}],
                                       "chat_template_kwargs": {"enable_thinking": False}})["prompt"]
    prefix = [t["token"] for t in tokens[:step]]
    print(f"{pid} step {step}: transcript arm {arm!r} chose {tokens[step]['token']!r}; its top: "
          + ", ".join(f"{t!r}:{lp:.4f}" for t, lp in tokens[step]["top"]))
    for n_new in sizes:
        if n_new > step:
            continue
        # warm the cache up to step - n_new tokens, then evaluate the last n_new in one batch
        warm = tpl + "".join(prefix[:step - n_new])
        post(url, "/completion", {"prompt": warm, "n_predict": 0, "cache_prompt": True})
        full = tpl + "".join(prefix)
        r = post(url, "/completion", {"prompt": full, "n_predict": 1, "n_probs": 4, "temperature": 0, "top_k": 1,
                                       "cache_prompt": True, "samplers": ["top_k"]})
        t = r.get("timings") or {}
        evaluated = f"prompt_n {t.get('prompt_n')}, cache_n {t.get('cache_n')}"
        top = (r.get("completion_probabilities") or [{}])[0].get("top_logprobs") or []
        print(f"  batch of {n_new} new token(s) (server {evaluated}): chose {r['content']!r}; top: "
              + ", ".join(f"{x.get('token')!r}:{x.get('logprob', 0):.4f}" for x in top))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
