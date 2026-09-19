# Ternary-Bonsai-2-27B-PTQ1_0-mtp (draft model card, not published)

Ternary Bonsai 2 27B (PrismML, PTQ1_0, 1.75 bpw ternary, group 128) with the
Qwen 3.8 27B multi-token-prediction head grafted back on as block 64, so that
the PrismML llama.cpp fork can run `--spec-type draft-mtp` on it.

## Files

| file | bytes | tensors | sha256 | needs |
| --- | ---: | ---: | --- | --- |
| Ternary-Bonsai-2-27B-PTQ1_0-mtp.gguf | 7,012,820,512 | 867 | 83a396ee218c36e5ed88205eccb940a71549d9a72a3d020cc2713f94a78f70f0 | PrismML fork, prism-b10685 or later |
| Ternary-Bonsai-2-27B-PTQ1_0-mtp-lean.gguf | 6,297,658,848 | 866 | 1e33c571a5ce7a9a3e42474d66192923d5a6d77da7fb3a22986dc809522b5685 | PrismML fork plus the 15 line qwen35 MTP patch |

Parents:

- PrismML `Ternary-Bonsai-2-27B-PTQ1_0.gguf`, 5,946,648,928 bytes,
  sha256 53107f530aa52eb00912263ab1ee29bd199261c87cd7b4ad4ca1318c1fe33ee3.
  Every one of its 851 tensors is present with identical bytes and offsets;
  `tools/merge.py --strip` on the merged file reproduces this sha256.
- unsloth `Qwen3.8-27B-UD-Q4_K_M.gguf`, 16,464,440,224 bytes,
  sha256 322e194ff79741c7baa497c240f677f54b201b0efab44ca8e50f122b39123482.
  Source of the 15 `blk.64.*` tensors (Q6_K, Q8_0, F32, 335 MiB) and, in the
  fat file, of `blk.64.nextn.embed_tokens.weight` (Q4_K copy of its token_embd, 682 MiB).

Header changes versus Bonsai 2: `qwen35.block_count` 64 -> 65,
`qwen35.nextn_predict_layers = 1`, plus `graft.donor.name`, `graft.head_blocks`,
`graft.head_tensor_count`, `graft.tool` for provenance. Everything else,
including the `prism.hadamard.*` keys and the tokenizer, is byte-identical.

## What it does and does not do

- Loads and serves with `--spec-type draft-mtp` on the PrismML fork. Without
  the flag it behaves exactly like Bonsai 2 (the head is skipped).
- The head was trained by Qwen against fp16 hidden states. Here it reads a
  ternary trunk. Measured draft acceptance on an RTX 3060 12GB: 0.85 to 0.95
  on Python, 0.73 to 0.81 on bash, 0.45 to 0.68 on prose, 0.56 to 0.65 at
  41.8K tokens of context.
- Speed on that card (probe.py, thinking off, 131072 context, one slot):
  25.0 tok/s without the flag, 27.1 tok/s with `--spec-draft-n-max 1` (+8%),
  22.5 with n-max 2 (-10%). The target's PTQ1_0 kernels make batched
  verification expensive; see `results/rtx3060.md`.
- Greedy output with the flag on is not byte-identical to greedy output with
  the flag off on this fork (batch-size dependent kernels). If you need
  reproducible greedy text, do not use the flag.

## Serving

    llama-server -m Ternary-Bonsai-2-27B-PTQ1_0-mtp.gguf -ngl 99 -fa on -c 131072 -np 1 \
      -ctk q4_0 -ctv q4_0 --jinja --spec-type draft-mtp --spec-draft-n-max 1

VRAM on an RTX 3060 12GB: 10,638 MiB at 131072, 11,726 MiB at 163840 (with
`-ctkd q4_0 -ctvd q4_0`). The lean file with the patched fork: 9,956 MiB at
131072, 11,990 MiB at 196608. Stock ggml-org llama.cpp cannot read PTQ1_0 and
produces gibberish on any Bonsai 2 file.

## Licence

Both parents are Apache 2.0 (Qwen/Qwen3.8-27B by Alibaba Cloud, Ternary Bonsai
2 27B by PrismML, unsloth's GGUF export of the former). This file redistributes
their weights unchanged under the same licence. Not affiliated with PrismML,
Qwen or unsloth.
