# RUN_REPORT: Qwen 3.8 MTP head grafted onto Ternary Bonsai 2 27B, RTX 3060 12GB

Agent: grafter. Run: Sep 18 2026 20:30 UTC to Sep 18 2026 22:50 UTC (03:30 to 05:50 ICT), unattended. Superseded on the speed and identity questions by KERNEL_REPORT.md (Sep 19).

## Facts

Merged files (both on the node in `/root/models/bonsai2-27b/`, neither copied back):

| file | bytes | tensors | sha256 |
| --- | ---: | ---: | --- |
| Ternary-Bonsai-2-27B-PTQ1_0-mtp.gguf | 7,012,820,512 | 867 | 83a396ee218c36e5ed88205eccb940a71549d9a72a3d020cc2713f94a78f70f0 |
| Ternary-Bonsai-2-27B-PTQ1_0-mtp-lean.gguf | 6,297,658,848 | 866 | 1e33c571a5ce7a9a3e42474d66192923d5a6d77da7fb3a22986dc809522b5685 |

Inputs: Bonsai 2 `Ternary-Bonsai-2-27B-PTQ1_0.gguf` 5,946,648,928 bytes, 851 tensors,
sha256 `53107f53...ee3` (unchanged, verified before and after). Donor pulled on the node:
`unsloth/Qwen3.8-27B-GGUF` `Qwen3.8-27B-UD-Q4_K_M.gguf`, 16,464,440,224 bytes,
sha256 `322e194ff79741c7baa497c240f677f54b201b0efab44ca8e50f122b39123482` (the plain
Q4_K_M that beast holds is no longer in the repo; the UD file has the same 15 head tensors in
Q6_K/Q8_0/F32 instead of Q4_K/Q6_K/Q8_0). Head files: `qwen38-27b-nextn-head.gguf`
1,066,172,160 bytes (16 tensors, sha256 `89a3144a...956`), `qwen38-27b-nextn-head-lean.gguf`
351,010,464 bytes (15 tensors, sha256 `4c696d34...d64`), both in `/root/mtp-graft/`.

Round trip: `merge.py --strip` on the merged file reproduces the original Bonsai 2 sha256
byte for byte (results/node/node_strip.txt). pytest: 16 passed on the node (with both real
files), 15 passed and 1 skipped on beast (no Bonsai file there).

Largest context with the head resident:

| context | file, binary | VRAM | note |
| --- | --- | ---: | --- |
| 131072 | mtp, prebuilt prism-b10685 | 10,638 MiB (n-max 1), 10,788 MiB (n-max 2), 10,938 MiB (n-max 3) | f16 draft K/V |
| 163840 | mtp, prebuilt | 11,726 MiB | `-ctkd q4_0 -ctvd q4_0`, generates fine |
| 196608 | mtp, prebuilt | OOM | see failures |
| 131072 | lean, patched 254725c6a | 9,956 MiB | |
| 196608 | lean, patched | 11,976 MiB idle, 11,990 MiB peak | generates, 41.8K prefill fine |

So: 163840 on the prebuilt fork, 196608 with the 15 line patch. 262144 cannot fit with any head
(flag off alone is 11,698 MiB).

Identity (temperature 0, top_k 1, 300 tokens, thinking off, same merged file):

| prompt | off vs off (2 runs) | off vs on | first divergence | flag-off top two at that step |
| --- | --- | --- | --- | --- |
| code | identical | differs | byte 216, token 52 | ` combines` -0.884, ` merges` -1.009 |
| prose | identical | differs | byte 58, token 14 | ` data` -1.386, ` memory` -1.399 |
| bash | identical | differs | byte 99, token 24 | `\n` -0.827, `\n\n` -0.931 |

Flag on picked the runner up in all three. All flag-on arms (n-max 1, 2, 3, prebuilt fat file,
patched lean file) produced byte-identical text to each other (sha256 `2ff5fe93`, `591a3caa`,
`53ac46fa`), and different seeds do not change it, so the flag-on arm is deterministic too.
Identity fails. Transcripts and per-token logprobs are in `results/identity/`.

A/B at 131072 (probe.py client tok/s, 3 runs x 3 prompts, `-np 1`, thinking off; acceptance from
the server's `draft acceptance` line per run):

| arm | prompt | run 1 | run 2 | run 3 | median | acceptance |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| off, prebuilt | code | 25.4 | 25.2 | 24.8 | 25.2 | |
| off, prebuilt | prose | 24.9 | 25.0 | 25.0 | 25.0 | |
| off, prebuilt | bash | 25.0 | 25.0 | 25.0 | 25.0 | |
| on n-max 1, prebuilt | code | 30.1 | 29.5 | 29.4 | 29.5 | 0.946, 0.913, 0.913 |
| on n-max 1, prebuilt | prose | 25.9 | 23.5 | 23.6 | 23.6 | 0.681, 0.517, 0.521 |
| on n-max 1, prebuilt | bash | 28.0 | 27.1 | 26.5 | 27.1 | 0.814, 0.765, 0.727 |
| on n-max 2, prebuilt | code | 26.7 | 26.9 | 26.9 | 26.9 | 0.860, 0.872, 0.898 |
| on n-max 2, prebuilt | prose | 17.9 | 17.3 | 17.4 | 17.4 | 0.398, 0.380, 0.382 |
| on n-max 2, prebuilt | bash | 24.3 | 22.5 | 22.3 | 22.5 | 0.736, 0.648, 0.634 |
| on n-max 3, prebuilt | code | 25.6 | 26.4 | 26.2 | 26.2 | 0.777, 0.810, 0.799 |
| on n-max 3, prebuilt | prose | 15.7 | 15.3 | 15.3 | 15.3 | 0.346, 0.331, 0.337 |
| on n-max 3, prebuilt | bash | 22.2 | 20.7 | 22.3 | 22.2 | 0.640, 0.563, 0.634 |
| on n-max 1, patched lean | code | 29.7 | 28.6 | 30.0 | 29.7 | 0.909, 0.851, 0.938 |
| on n-max 1, patched lean | prose | 23.5 | 22.3 | 24.0 | 23.5 | 0.515, 0.448, 0.554 |
| on n-max 1, patched lean | bash | 27.0 | 27.5 | 26.4 | 27.0 | 0.753, 0.793, 0.730 |

Overall medians: off 25.0, n-max 1 27.1 (+8%), n-max 2 22.5 (-10%), n-max 3 22.2 (-11%),
patched lean n-max 1 27.0 (+8%). The flag-off number matches the pre-run baseline in the SPEC
(25.81 tok/s server print_timing on a 400 token probe).

Deep row (41,812 token prompt from tools/deep_probe.py, 300 token answer, 131072 context,
client tok/s, server predicted_per_second in parentheses, medians):

| arm | tok/s | acceptance |
| --- | ---: | --- |
| off, prebuilt | 16.85 (16.79) | |
| on n-max 2, prebuilt | 16.43 (16.38) | 0.469, 0.528, 0.555 |
| on n-max 1, patched lean | 18.75 (18.68) | 0.628, 0.625, 0.646 |
| on n-max 1, patched lean, 196608 context, 2 runs | 17.90 (17.87) | 0.562, 0.620 |

Prefill of the 41.8K prompt ran at 236 to 239 tok/s in every arm.

## What was built

- `tools/gguf_raw.py`: GGUF v3 header parser and writer that copies tensors as byte spans
  (offset to next offset, last to end of file), so it never needs the element size of
  PTQ1_0 (id 143, 128 elements in 28 bytes) or PQ2_0 (id 142, 128 in 34 bytes); those sizes
  are only used for sanity checks and came from the fork's `ggml/src/ggml-common.h`.
- `tools/extract_head.py`: reads `<arch>.block_count` and `<arch>.nextn_predict_layers`,
  takes every `blk.64.*` tensor (15 in the donor), and by default adds a 16th,
  `blk.64.nextn.embed_tokens.weight`, as a renamed copy of the donor's `token_embd.weight`.
- `tools/merge.py`: Bonsai kv in original order with `qwen35.block_count` 64 -> 65,
  `qwen35.nextn_predict_layers = 1` inserted after it, four `graft.*` keys appended; Bonsai
  tensors verbatim at their original offsets, head tensors appended (renamed to the trunk's
  next free block index, 64 here). `--strip` reverses it.
- Fork branch `bonsai2-mtp` on the node in `/root/llama.cpp-sudo` (remote renamed to `prism`,
  base `7dffb15` = the prebuilt release), one commit `254725c6a`, built to
  `/root/llama.cpp-sudo/build/bin/llama-server` (reports build 10686). `~/bin/llama-server`
  still points at the prebuilt binary. Patch: `results/0001-qwen35-mtp-hadamard-inverse.patch`.

## What the source said (Method step 1 and fallback 1)

`src/models/qwen35.cpp` at 7dffb15: `n_layer_nextn` is read from `%s.nextn_predict_layers`
and must be `< block_count`; `n_layer()` = `block_count - n_layer_nextn`; blocks
`n_layer .. block_count-1` load through `load_block_mtp` (attn_norm, post_attention_norm,
attn_q/k/v or attn_qkv, attn_output, attn_q_norm, attn_k_norm, ffn_gate/down/up,
nextn.eh_proj [2*n_embd, n_embd], nextn.enorm, nextn.hnorm, and optional nextn.embed_tokens,
nextn.shared_head_head, nextn.shared_head_norm). MTP layers are flagged non-recurrent
(`is_recr_impl[i] = i < n_layer() && (i+1) % 4 != 0`). The MTP graph (`graph_mtp`) uses
`model.tok_embd` when `nextn.embed_tokens` is absent, and `model.output` (with its Hadamard
rotation through `build_lora_mm`) when `nextn.shared_head_head` is absent. The draft context is
created against the target model (`common_speculative_init_result`, "creating MTP draft
context against the target model"), a separate draft file is not used for MTP. The arch label is
`qwen35` on both files, so no renaming was needed (fallback 4 not triggered).

## What failed, exact text

1. Lean merged file (no embedding copy) on the prebuilt fork, `-c 32768 --spec-type draft-mtp`:

       llama_verify_hadamard_graph: latent lookup 'mtp_tok_embd-64' consumed by op=RMS_NORM name='norm-64' src0 hint=0
       llama_init_from_model: failed to initialize the context: Hadamard-latent table 'token_embd.weight' is read without the inverse transform
       common_speculative_init_result: failed to create MTP context
       srv    load_model: failed to create MTP context
       srv  llama_server: exiting due to model loading error

   Cause: `graph_mtp` does `ggml_get_rows(model.tok_embd, ...)` with no inverse rotation,
   and Bonsai 2 lists `token_embd.weight` in `prism.hadamard.inverse_weight_names`. Two fixes,
   both used: (a) GGUF side, ship the donor's embedding table as `blk.64.nextn.embed_tokens.weight`
   (fat file, prebuilt fork), (b) Phase B, 15 lines in `src/models/qwen35.cpp` applying
   `llama_mul_mat_hadamard` and the sign vector after the lookup, mirroring `build_inp_embd()`.
   Both give the same acceptance and speed; (b) saves 682 MiB and unlocks 196608.

2. 196608 context, fat file, prebuilt fork, `-ctkd q4_0 -ctvd q4_0 --spec-draft-n-max 1`:

       ggml_backend_cuda_buffer_type_alloc_buffer: allocating 1040.28 MiB on device 0: cudaMalloc failed: out of memory
       ggml_gallocr_reserve_n_impl: failed to allocate CUDA0 buffer of size 1090816128
       graph_reserve: failed to allocate compute buffers
       llama_init_from_model: failed to initialize the context: failed to allocate compute pp buffers
       srv    load_model: failed to create MTP context

   Same with `-ub 256` (compute buffer 904.27 MiB, still OOM). Main KV 3,456 MiB, main compute
   1,050 MiB, model 6,412 MiB, draft KV 216 MiB, then the 1,040 MiB MTP compute buffer does not
   fit. 163840 fits at 11,726 MiB. With the lean file and the patch, 196608 fits at 11,990 MiB.

3. Identity at temperature 0 (Method step 5): fails on all three prompts, see the table above.
   Fallback 5 in order: `--spec-draft-n-max 1` gives the same flag-on text; the head quant is
   irrelevant to identity because the target verifies every draft (the patched arm with the
   ternary embedding table gives the same text too); flag off is deterministic across two runs;
   different seeds do not change the flag-on text. The cause is in the target's kernels, not the
   head: `tools/batch_numerics.py` replays the greedy prefix so that the last N tokens are
   evaluated in one batch (server `prompt_n` = N, rest from the prompt cache) and reads the
   top candidates for the next token with no speculative decoding involved. Prose, step 14:
   N=1 ` data` by 0.003 nats, N=2 and N=3 ` memory` by 0.021, N=4 ` data` by 0.0005, N=8
   ` memory` by 0.009. Code, step 52: N=1,2,3,8 ` combines`, N=4 ` merges`. The fork's CUDA
   path (PTQ1_0 mmvq for 1 column, the `vec_dot_ptq1_0_q8_1_multi` path for 2 to 3 columns,
   generic mmvq for 4 to 7, MMQ from 8) gives logits that differ by up to about 0.1 nats
   between batch sizes on this model, and speculative verification always runs 2 or 3 token
   batches. Nothing on the GGUF side can change that. Also tried, fallback 3: `-fa off` with
   f16 K/V (16384 context, both arms, flag arm n-max 1): prose identical over 157 tokens, code
   differs at token 74 (` efficient` -1.241 vs ` an` -1.263), bash at token 179 (` close`
   -0.814 vs ` create` -0.827), so 2 of 3 still fail, later and again at near ties. The
   flag-off text itself changes between the FA-on q4_0 and FA-off f16 configurations on all
   three prompts (`results/identity/*.off_faoff.*`), the same effect from a different knob.
   Head quant variants (fallback 2: q8_0, f16) were not run for identity: the target verifies
   every draft with its own logits, so the head can only change which drafts are proposed, not
   which tokens survive. The measured evidence agrees: swapping the head's embedding table from
   the donor's Q4_K copy to Bonsai's own ternary table (fat file vs patched lean file) changes
   the per-run acceptance but leaves all three flag-on texts byte-identical. A q8_0 or f16 head
   would need the 29 GB Q8_0 or the 54 GB bf16 donor on a node with 33 GB free and would
   answer only the acceptance question, which the Q6_K/Q8_0 head already answers at 0.9.

4. No speed win at n-max 2 despite 0.86 to 0.90 acceptance on code: `llama-bench` on the
   merged file (prebuilt, 4096 context, `-fa 1 -ctk q4_0 -ctv q4_0`): tg32 26.02 tok/s
   (38.4 ms per token), pp2 61.7 ms per batch (1.6x one token), pp3 93.0 ms (2.4x), pp4 115.6 ms
   (3.0x), pp8 151.9 ms (4.0x), pp16 159.9 ms (4.2x). Verifying 2 drafts costs 2.4 steps and
   returns at most 3 tokens, so the best case is +25% and prose loses. For scale, the same
   binary on a Q4_0 Qwen3.5 9B (same qwen35 architecture, pulled only for this check):
   pp1 18.8 ms, pp2 19.9 ms (1.06x), pp3 20.2 ms (1.07x), pp4 22.4 ms (1.19x). The poor
   small-batch scaling belongs to the PTQ1_0 CUDA mat-vec path.

5. Operational: one 196608 deep-run attempt hung when my own restart script sent C-c to the
   tmux window while a 41.8K prefill was 93% done; the server stayed alive but unresponsive
   (GPU 0%, /slots timed out) and was killed with `pkill -x llama-server`, then the run was
   repeated cleanly. An earlier `pkill -f llama-server` killed the ssh session that issued it
   (the pattern matched the script's own command line); use `pkill -x`.

## Not done

- q8_0 and f16 heads (fallback 2): see failure 3 for why they cannot move identity; the
  Q6_K/Q8_0 head loads and drafts at 0.9 acceptance, and the disk did not have room for the
  bf16 checkpoint next to the donor and the merged files.
- `--spec-draft-n-max 4`: n-max 3 was already 11% below the flag-off, 4 would be worse on
  the pp4 = 3.0x cost curve.
- Publishing (Phase C): `results/MODEL_CARD.md` and `results/PR_NOTES.md` drafted, no uploads,
  no PR, no push.

## State of the node at the end

Canonical serve line restored in tmux `server` (prebuilt fork, original Bonsai 2 file,
262144 context, flag off), `/health` returns `{"status":"ok"}`, slot idle, 11,698 MiB. Extra
files left on the node: the two merged GGUFs and two head GGUFs listed above, the donor
(16.5 GB) in `/root/models/qwen3.8-27b/`, the 9B diagnostic model (5.4 GB) in
`/root/models/diag/`, the fork source and build in `/root/llama.cpp-sudo` (branch `bonsai2-mtp`,
uncommitted nothing), scripts and results in `/root/mtp-graft/`, server logs in `~/server_*.log`.
33 GB disk free. Original Bonsai 2 file and its SHA256SUMS untouched. `~/.hermes` untouched.

## The read

The graft itself works: sixteen tensors and two header edits are enough for the PrismML fork to
build the MTP draft context on Bonsai 2, and the fp16-trained Qwen 3.8 head agrees with the
ternary trunk often enough to be a good drafter (0.9 acceptance on code, 0.5 on prose, 0.6 at
42K deep). It does not give 12GB owners a real speed win today: the best setting, n-max 1, is
+8% overall (+17% on code, -6% on prose) and +11% at 42K depth, while n-max 2 and 3 are
slower than the flag off. The cost is 1.9 GB of VRAM with the prebuilt fork (largest context
163840 instead of 262144) or 1.2 GB with the one-commit patch (196608). The reason is the fork's
PTQ1_0 mat-vec kernels, which charge 1.6 and 2.4 single-token steps for 2 and 3 token batches
where a Q4_0 quant pays 1.06 and 1.07; until that path amortizes across columns, the head can
draft as well as it likes and the target will eat the gain. And greedy output with the flag is
not byte-identical to greedy output without it, for the same kernel reason, so none of the
above is a lossless speedup claim. Answer: no, not yet, and the fix is in ggml-cuda, not in the
GGUF.
