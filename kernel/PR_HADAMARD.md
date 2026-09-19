# PR: qwen35: apply the Hadamard inverse to token embeddings in the MTP draft graph

Branch: `pr-hadamard-mtp`, one commit on top of `prism` (`9a9394a89`), file `src/models/qwen35.cpp`, 15 insertions.

## What

`llama_model_qwen35::graph_mtp` looks the draft token's embedding up with a raw
`ggml_get_rows(model.tok_embd, ...)`. When the model folds a Hadamard rotation into its weights
and lists `token_embd.weight` in `prism.hadamard.inverse_weight_names` (every Ternary Bonsai 2
file does), the rows of that table are stored rotated. The main graph restores the primal basis
right after the lookup in `llm_graph_context::build_inp_embd()`; the MTP graph did not. This
change applies the same `llama_mul_mat_hadamard` and sign vector after the lookup when the
table is in `hadamard_inverses`. Models without `prism.hadamard.*` keys take the path they took
before (the lookup finds nothing).

## Why

No PrismML export carries an MTP block today, so the case never came up. It does the moment
someone grafts the Qwen 3.8 `blk.64.nextn.*` tensors onto Bonsai 2 to get `--spec-type draft-mtp`
(the graft tools and measurements are in github.com/sudoingX/bonsai2-small-gpu). Without the fix the
draft context is refused at load:

```
llama_verify_hadamard_graph: latent lookup 'mtp_tok_embd-64' consumed by op=RMS_NORM name='norm-64' src0 hint=0
llama_init_from_model: failed to initialize the context: Hadamard-latent table 'token_embd.weight' is read without the inverse transform
common_speculative_init_result: failed to create MTP context
srv    load_model: failed to create MTP context
```

The workaround is to ship a second, unrotated embedding table inside the head
(`blk.64.nextn.embed_tokens.weight`, a Q4_K copy of the donor's `token_embd`, 682 MiB of VRAM).
With the fix the head reads the trunk's own table and the copy is not needed: 9,956 MiB instead
of 10,638 MiB at 131072 context on an RTX 3060 12GB, and 196608 context fits (11,990 MiB) where
the fat file OOMs on the MTP compute buffer.

## Reproduction

1. `Ternary-Bonsai-2-27B-PTQ1_0.gguf` plus the 15 `blk.64.*` tensors of any Qwen3.8-27B GGUF,
   with `qwen35.block_count` set to 65 and `qwen35.nextn_predict_layers = 1`
   (`tools/extract_head.py --no-embed-tokens` and `tools/merge.py` in the repo above; the
   merged file is 6,297,658,848 bytes, sha256 `1e33c571...5685`).
2. `llama-server -m Ternary-Bonsai-2-27B-PTQ1_0-mtp-lean.gguf -ngl 99 -fa on -c 32768 -np 1 -ctk q4_0 -ctv q4_0 --jinja --spec-type draft-mtp --spec-draft-n-max 1`
3. Before: the three lines above and exit. After: `spec common_specu: adding speculative implementation 'draft-mtp'`, `speculative decoding context initialized`.

## Measured (RTX 3060 12GB, 131072 context, q4_0 K/V, one slot)

| | fat file, prebuilt prism-b10685 | lean file, this fix |
| --- | ---: | ---: |
| VRAM, n-max 1 | 10,638 MiB | 9,956 MiB |
| draft acceptance, code / prose / bash (identity prompts) | 0.91 / 0.65 / 0.80 | 0.89 / 0.67 / 0.77 |
| greedy text with the flag | same three texts | same three texts |
| largest context with the head resident | 163840 | 196608 |

The draft quality with the trunk's ternary, Hadamard-rotated embedding table equals the donor's
fp table: acceptance per run 0.85 to 0.94 on Python, 0.45 to 0.55 on prose, 0.73 to 0.79 on bash.

## Known limits

- Only the `qwen35` MTP graph is changed. Other architectures with an MTP graph
  (`qwen3next`, `glm4-moe`, `deepseek2`, ...) do the same raw lookup; none of them has a
  Hadamard-folded export today, so they are left alone here.
- A unit test would need a Hadamard-folded model with a nextn block; there is none in the tree.
  `llama_verify_hadamard_graph` is the guard that catches the bug, and it now passes on the
  file above.
