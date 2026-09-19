# KERNEL_REPORT: PTQ1_0 small-batch decoding on the PrismML fork, RTX 3060 12GB

Agent: grafter, continuing RUN_REPORT.md. Run: Sep 19 2026 06:15 UTC to 11:00 UTC (13:15 to 18:00 ICT).

Branch on the node: `/root/llama.cpp-sudo`, `ptq1-batched-mmv`, seven commits on top of `bonsai2-mtp`
(`254725c6a`, base `7dffb15` = prebuilt prism-b10685). Head `ecaa8b374`; the measurements below were taken at
`597c7f0bc` (build 10692), the last commit only removes a temporary env switch and pins the fold to an explicit
`__fmaf_rn` (outputs and timings verified unchanged: same sha256 on all identity texts, tg32 39.91).
Patches: `results/kernel/patches/0001..0007`. Diff: 8 files, 596 insertions, 21 deletions.
`~/bin/llama-server` still points at the prebuilt binary. Commits local only, nothing pushed.

## Facts

### 1. Cost curve (`llama-bench`, merged Bonsai 2 file, 4096 context, `-fa 1 -ctk q4_0 -ctv q4_0`, 8 repetitions)

| test | prebuilt tok/s | patched tok/s | prebuilt ms per batch | patched ms per batch | prebuilt vs pp1 | patched vs pp1 | target |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| pp1 | 25.26 | 37.29 | 39.6 | 26.8 | 1.00 | 1.00 | |
| pp2 | 34.10 | 61.08 | 58.7 | 32.7 | 1.48 | 1.22 | 1.15 |
| pp3 | 33.29 | 72.04 | 90.1 | 41.6 | 2.28 | 1.55 | 1.30 |
| pp4 | 35.26 | 81.03 | 113.5 | 49.4 | 2.87 | 1.84 | 1.50 |
| pp8 | 53.81 | 73.06 | 148.7 | 109.5 | 3.76 | 4.09 | |
| pp16 | 102.61 | 102.16 | 155.9 | 156.6 | | | MMQ path, unchanged |
| tg32 | 26.14 | 39.76 | 38.3 | 25.2 | | | within 2% of 26.0 |

Same numbers with `GGML_CUDA_BATCH_INVARIANT=1`: pp1 37.24, pp2 61.42, pp3 71.54, pp4 80.52, pp8 72.63, tg32 39.70.
Single-token decode is 1.52x the prebuilt fork. A 3-token verification batch costs 41.6 ms instead of 90.1 ms.
The ratio targets for pp2, pp3 and pp4 are not met: each extra column still costs about 22% of a single pass.
Note pp1 carries about 1.6 ms of prompt-path overhead over tg (26.8 vs 25.2 ms), so the ratios against tg
are 1.30, 1.65 and 1.96.

Kernel level (`test-backend-ops perf`, Bonsai 2 shapes, us per call, prebuilt path = the fork's PTQ1_0 mmvq
with the `_multi` variant; final = commit `597c7f0bc`):

| shape (K x M) | cols | prebuilt path | final | ratio |
| --- | ---: | ---: | ---: | ---: |
| attn_qkv 5120 x 10240 | 1 | 72.4 | 41.8 | 0.58 |
| | 2 | 75.0 | 52.5 | 0.70 |
| | 3 | 100.4 | 71.8 | 0.72 |
| | 4 | 150.7 | 85.4 | 0.57 |
| | 8 | 219.0 | 193.9 | 0.89 |
| ffn_up 5120 x 17408 | 1 | 120.0 | 66.7 | 0.56 |
| | 2 | 124.6 | 84.7 | 0.68 |
| | 3 | 165.9 | 116.4 | 0.70 |
| | 4 | 254.3 | 140.9 | 0.55 |
| ffn_down 17408 x 5120 | 1 | 75.1 | 68.5 | 0.91 |
| | 2 | 85.3 | 81.8 | 0.96 |
| | 3 | 107.4 | 112.1 | 1.04 |
| | 4 | 150.6 | 145.7 | 0.97 |
| attn_gate 5120 x 6144 | 1 | 44.8 | 27.7 | 0.62 |
| | 4 | 92.3 | 54.2 | 0.59 |

(The "prebuilt path" column was measured on the stage 1 build with the generic mmvq path, which is the
prebuilt kernel plus the activation layout change; on the 4096 x 14336 test shape the prebuilt kernel itself
measured 47.3 / 103.8 / 154.3 / 201.0 us for 1 / 2 / 3 / 4 columns and the final kernel 45.9 / 56.5 / 74.2 /
83.9.) All 47 PTQ1_0 MUL_MAT and 77 MUL_MAT_ID correctness cases pass against the CPU reference.

### 2. Batch invariance (`tools/batch_numerics.py`, the same prefix, the last N tokens in one batch)

Default build: top-1 identical for N = 1, 2, 3, 4, 8 at all three steps; logprobs differ by up to 0.02 nats
between N = 1 and N = 2 (the flash-attention and bf16 gate kernels change with N).

`GGML_CUDA_BATCH_INVARIANT=1`: N = 1, 2, 3, 4 give bit-identical logprobs at all three steps, for example
prose step 14: ` memory` -1.3994, ` data` -1.4060, ` file` -1.6064, ` large` -1.6643 for every N in 1..4.
N = 5, 6, 7, 8 are identical to each other but differ from 1..4 by up to 0.03 nats (` memory` -1.3689);
top-1 is the same. The kernel that switches between 4 and 5 columns was not found (see failures).

### 3. Identity (temperature 0, top_k 1, 300 tokens, thinking off, 131072 context, `GGML_CUDA_BATCH_INVARIANT=1`)

| prompt | flag off vs n-max 1 | vs n-max 2 | vs n-max 3 | sha256 |
| --- | --- | --- | --- | --- |
| code | identical, 1023 bytes | identical | identical | 9948638b |
| prose | identical, 789 bytes | identical | identical | 591a3caa |
| bash | identical, 1073 bytes | identical | identical | 8e08e427 |

Without the knob (default kernels), n-max 1 and 2 differ from flag off on code (token 52, ` combines` vs
` merges`) and prose (token ~60), bash identical: throughput-only configurations.

Flag-off patched vs flag-off prebuilt: the greedy text changed on all three prompts (code from token 74 in the
knob build, prose from token 14, bash from token ~45), each at a near tie (top two within 0.13 nats). Expected:
the summation order over K changed. The batched-prefill check that showed the prebuilt fork's own decode and
prefill paths disagreeing by up to 0.1 nats still holds for prompt processing (MMQ at 9+ tokens); within
1..4 tokens the patched build is now exact.

### 4. A/B (`probe.py`, 131072 context, `-np 1`, thinking off, 3 runs x 3 prompts, client tok/s, medians)

Patched build `597c7f0bc`, `GGML_CUDA_BATCH_INVARIANT=1` (lossless configuration):

| arm | code | prose | bash | overall median | vs flag off | acceptance (probe runs) |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| flag off | 39.8 | 40.0 | 39.7 | 39.8 | | |
| n-max 1 | 53.2 | 41.8 | 50.1 | 50.1 | +26% | code 0.90-0.94, prose 0.48-0.55, bash 0.75-0.83 |
| n-max 2 | 55.1 | 36.8 | 45.4 | 45.4 | +14% | code 0.85-0.88, prose 0.38-0.42, bash 0.56-0.66 |
| n-max 3 | 55.9 | 32.9 | 41.6 | 41.6 | +5% | code 0.80-0.86, prose 0.27-0.37, bash 0.47-0.54 |

Same build, default kernels (not lossless): n-max 1 53.7 / 43.5 / 50.0 (50.0), n-max 2 56.9 / 36.9 / 48.7 (48.7).

Against the prebuilt fork's 25.0 tok/s flag-off baseline from RUN_REPORT.md: flag off 39.8 (+59%),
lossless n-max 1 50.1 (2.0x), code with n-max 2 55.1 (2.2x). Target check: n-max 2 code +38% (target +40%),
overall +14% (target +20%); n-max 1 overall +26% meets the overall target. Per-run figures are in
`results/kernel/final/probe_*.txt`, acceptance per request in `results/kernel/final/final_runs_log.txt`.

### 5. Deep row (41,812 token prompt, 300 token answer, 131072 context, medians of 3)

| arm | tok/s | acceptance |
| --- | ---: | --- |
| prebuilt, flag off (RUN_REPORT) | 16.85 | |
| patched, knob, flag off | 22.23 | |
| patched, knob, n-max 1 | 26.86 (+21%) | 0.67, 0.74, 0.76 |
| patched, knob, n-max 2 | 21.08 (-5%) | 0.48, 0.60, 0.62 |
| patched, default kernels, n-max 2 | 28.31 (+27%) | 0.48, 0.52, 0.55 |

At depth the knob's cost shows: the vector flash-attention kernel with 3 queries over 42K keys is much slower
than the tensor-core kernel, so lossless n-max 2 loses at depth while the default n-max 2 gains 27%. n-max 1
gains either way.

### 6. VRAM at 131072: flag off 8,738 MiB, n-max 1 10,622 MiB, n-max 2 10,772 MiB, n-max 3 10,924 MiB (fat file).

## What the source said (method step 1)

- `mmvq.cu`: for PTQ1_0, `vdr = 4`, `qi = 4`, so `qi/vdr = 1`: one thread owns a whole 128-weight block, 128
  threads per CTA cover 128 blocks per iteration. Columns 2 and 3 used `vec_dot_ptq1_0_q8_1_multi<ncols>`
  (`vecdotq.cuh`), which decodes the trits once but still loads the activations as 32 four-byte
  `get_int_b4` reads per column from 36-byte `block_q8_1` structs (4-byte alignment, no vector loads).
  Columns 4..7 used the generic per-column `vec_dot`, 8+ the MMQ tile path (`ggml_cuda_should_use_mmvq`
  returned `ne11 <= 7` for PTQ1_0). nwarps came from the GENERIC table on sm_86 (4 for 1..4 columns, 2 for
  5..8), so the K partition and the fp32 sum order changed at 5 columns.
- No Ada or Hopper specific PTQ1_0 path exists in the fork beyond the shared mmvq/MMQ code; PrismML's L40S
  numbers come from the same kernels on a card with 2.4x the bandwidth.
- The trit decode: 5 trits per byte via `v*3` in 16-bit lanes, `__byte_perm` to gather the high bytes,
  `__vsub4` for the `-1`; `qh` holds 8 more trits, `d` is the last half of the 28-byte block.
- `quantize_q8_1` writes 36-byte blocks `{half2 (d, sum of x), int8 qs[32]}`; only mmvq consumes it
  (`ggml_cuda_op_mul_mat_vec_q`, the split-mode variant, has no callers).
- Measured before writing anything: `test-backend-ops perf` PTQ1_0 4096 x 14336: 47.3 / 103.8 / 154.3 /
  201.0 / 237.5 / 395.2 us for 1 / 2 / 3 / 4 / 5 / 8 columns; Q4_0 on the same shape 101.5 / 102.2 / 114.6 /
  151.0 / 135.0 / 208.5. `ncu` cannot run in the container (`ERR_NVGPUCTRPERM`), so a per-op cudaEvent hook
  (not committed) was used for attribution, which showed the PTQ1_0 mat-vecs at 115-160 GB/s effective on
  the K = 5120 projections at 1 column, a third of the card's bandwidth.

## What was built, in order, with what it did

1. PT activation layout + shared decode across columns (commit 1): 4096 x 14336, 1..4 columns:
   45.7 / 54.9 / 65.2 / 74.8 us (was 47.3 / 103.8 / 154.3 / 201.0). Whole model pp2 1.48x -> 1.10x of pp1,
   pp3 2.28x -> 1.36x, tg 26.0 -> 27.4.
2. Experiments on that kernel (numbers in `results/kernel/shape_perf_*.txt` and the shell logs): 4 rows per
   thread in the generic kernel (no gain, registers 154 -> 233); removing the activation loads entirely
   (perf only, wrong results) made 1..8 columns cost 48 / 48 / 55 / 59 / 91 us, so the loads are the cost;
   halving the dp4a count changed nothing, so the integer dot products are not.
3. Dedicated 2D kernel with flattened (row group, K block) work items (commit 2): attn_qkv 72.4 -> 43.5 us,
   ffn_up 120 -> 69 us at 1 column; tg 27.4 -> 38.9 tok/s. First version used 32 KiB static shared memory
   and lost occupancy (n=1 56.9 us, n=2 112 us); dynamic shared memory sized per launch fixed it.
4. Warp tile of 8 rows x 4 K blocks so that 8 lanes share each activation piece (broadcast loads): worse on
   K = 5120 (attn_qkv 2 cols 63.1 vs 55.5 us, 4 cols 126 vs 90.5), slightly better on K = 17408. Reverted.
5. Tighter `__launch_bounds__` (3 or 4 CTAs per SM): registers 188 -> 160 at 4 columns, no spills, timing
   within 2%. Kept the bounds.
6. 4 rows per work item (commit 6): -3% to -13% across shapes; 8 rows spill (STACK 160) and lose 2x.
7. Invariance knob (commit 4) and the bf16 mat-vec rule (commit 5), found with `batch_numerics.py`: after
   the PTQ1_0 kernels were invariant, N = 1 vs N = 2 still differed by 0.01-0.02 nats; the bf16
   [5120 x 48] gate projections took `mul_mat_vec_f` at 1 column and a 10.5 us tensor-core path at 2+, and
   flash attention took the vector kernel at 1 query and the MMA kernel at 2+. Routing both to the
   1-column kernels for up to 8 columns, and pinning the flash-attention KV split to the 1-query geometry,
   made N = 1..4 bit-identical.

## What failed or stayed open, exact text

1. pp2 / pp3 / pp4 ratio targets (1.15 / 1.30 / 1.50): reached 1.22 / 1.55 / 1.84 (vs pp1). The per-column
   cost of the dedicated kernel on the K = 5120 shapes is 10-12 us per column per 13 MB matrix, about 1
   SM cycle per (block, column). Neither halving the activation traffic (4 rows per item, warp tiles with
   broadcast loads) nor raising occupancy (launch bounds) nor halving the dp4a count moved it by more than
   10%, while deleting the loads altogether removed it; the remaining explanation is the latency chain of
   the 9 dependent 16-byte loads per (block, column) per thread, which more rows per thread only partly
   hide. The tensor-core move was then tried in two forms, both correct (47 of 47 cases against the CPU
   reference) and both slower than the dp4a kernel at 2..4 columns, so neither is in the branch
   (diffs in `results/kernel/experiments/`):
   - `mma-16row-redundant-decode.diff`: `mma.sync.m16n8k16.s8` on 16 (then 32) rows x 1 K block per
     warp, each lane decoding the a-fragment words of its two (four) rows, B fragments loaded once for
     all 8 columns, exact int32 sub-block sums folded with the same `d * sum_k d8_k * sumi_k`
     expression and the same shared-memory reduction (bit-identical partials by construction, 64..91
     registers). attn_qkv at 2 / 3 / 4 / 8 columns: 103.9 / 109.5 / 134.2 / 161.0 us (16 rows) and
     86.7 / 98.0 / 101.7 / fallback (32 rows) against 52.5 / 71.8 / 85.4 / 193.9 for the dp4a kernel.
     Flat in the column count as intended, but the per-item cost is about 2x a single dp4a pass:
     each lane decodes three groups' worth of trits for 8 words (1.65x the decode work of the dp4a
     kernel) and the item loop is a long dependent chain at 8 to 20 warps per SM.
   - `mma-cooperative-decode.diff`: lanes 0..15 and 16..31 decode one row each of two items into shared
     memory (32 words per row, stride 36 for conflict-free fragment reads), then the warp runs both
     items' mma. Decode work equal to the dp4a kernel, but the extra 18 KiB of static shared memory
     and the store, syncwarp, load phases left 2 CTAs per SM: attn_qkv 2 columns 127.4 us (2.4x dp4a).
   The tensor-core path only pays at 8 columns (0.83x) on this card. A version that wins at 2 columns
   would need the decode shared without shared memory (register shuffles cost 4 per word) or a
   pre-transposed weight layout that lets a warp read 16 rows' blocks coalesced; both are beyond a
   small patch.
2. Invariance at 5..8 columns: N = 5..8 agree with each other and differ from N = 1..4 by up to 0.03 nats
   with the knob on. Checked and excluded: the PTQ1_0 kernels (same per-block partials, same reduction for
   every N), `mul_mat_vec_f` (K partition depends on K only), the FWHT Hadamard path (per row), the GDN
   kernel (sequential loop, precompute only at 32+ tokens), flash attention kernel choice (vector kernel
   for 1..8 under the knob) and its KV split (pinned). Not found in the time available; it is outside the
   n-max 1..3 range used for identity.
3. n-max 2 A/B target (+40% code, +20% overall): +38% code, +14% overall lossless; +43% code, +22% overall
   with the default kernels (not lossless). Where the step goes at n-max 2 on code: a 3-token verify batch
   costs 41.6 ms and yields 2.7 tokens on average (acceptance 0.86 with the bonus token), 2 MTP drafts add
   about 6 ms (server `dur(g)`), so 2.7 tokens per 48 ms against 25.2 ms per token single: +41%. On prose
   the acceptance of 0.40 yields 1.65 tokens per step, break-even before the draft cost.
4. `ncu`: `==ERROR== ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA GPU Performance
   Counters on the target device 0.` Attribution was done with a temporary cudaEvent hook instead (its
   per-op times include host launch gaps, which made cuBLAS-path ops look 30x worse than under CUDA graphs;
   the `test-backend-ops perf` numbers were used for decisions).
5. Build errors on the way: `quantize.cu(55): error: attributes are not allowed here` (`__launch_bounds__`
   before `template`, reordered); `mmvq.cuh(12): error: redefinition of default argument` (`mmvq.cuh` has no
   include guard, so the new header does not include it and a `static_assert` in `mmvq.cu` ties the two
   column limits together).
6. Operational: one 196608 deep run on the earlier day hung after a stray C-c; nothing of the kind today.
   The canonical serve line was down during all kernel benchmarking and restored at the end.

## Not done

- MoE PTQ1_0 was compiled and passes MUL_MAT_ID correctness but has no benchmark (no ternary MoE file).
- `--spec-draft-n-max 4` and p-min gating were not swept; n-max 3 was already below n-max 1 overall.
- A tensor-core small-batch variant that beats dp4a at 2 columns (failure 1 lists the two that were tried,
  the addendum a third; none did).
- The pre-transposed weight layout (addendum C: needs the MMQ loader, get_rows and buffer get/set to
  know the layout).

## State of the node at the end

Canonical serve line restored in tmux `server` (prebuilt fork, original Bonsai 2 file, 262144 context, flag
off), `/health` ok. Fork source and build in `/root/llama.cpp-sudo` (branch `ptq1-batched-mmv`, clean tree,
build dir with the patched `llama-server`, `llama-bench`, `test-backend-ops`); PR series `pr-hadamard-mtp`
and `pr-ptq1-mmv` on `prism/prism` in the same repo, built in the worktrees `/root/llama.cpp-pr-hadamard`
and `/root/llama.cpp-pr-mmv`. Scripts, logs and results in `/root/mtp-graft/`. Original Bonsai 2 file,
`~/bin` and `~/.hermes` untouched. Nothing pushed.

## Addendum (Sep 19 evening): PR series, depth ladder, third tensor-core attempt

### A. PR-ready branches on the fork (local, never pushed)

Rebased the `ptq1-batched-mmv` work onto `prism/prism` (`9a9394a89`, current fork head) into two series
in `/root/llama.cpp-sudo`, both built (`cmake --build`, sm_86) and tested on the GPU once it was free:

- `pr-hadamard-mtp` (1 commit on prism/prism): `76106714f Fix: apply the Hadamard inverse to token
  embeddings in the qwen35 MTP graph`. Patch and description: `results/pr/0001-*.patch`,
  `results/PR_HADAMARD.md`.
- `pr-ptq1-mmv` (5 commits on prism/prism): planar-transposed q8_1 layout, dedicated PTQ1_0 mat-vec
  kernel, Bonsai 2 shapes in `test-backend-ops`, `GGML_CUDA_BATCH_INVARIANT`, the bf16 under-64-rows
  rule. Patches `results/pr/kernel/0001..0005-*.patch`, description `results/PR_KERNEL.md`.

Tests on the `pr-ptq1-mmv` build (`test-backend-ops test -b CUDA0`, CPU reference): MUL_MAT ptq1_0 47/47,
MUL_MAT_ID ptq1_0 77/77, FLASH_ATTN_EXT 2938/2938 with the knob off and 2938/2938 with it on, MUL_MAT
f16/bf16 with the knob on 426/426. sudo's own A/B on the original file, same flags, r=3: prebuilt tg128
26.32, patched tg128 40.47, pp512 unchanged at about 268.

### B. Depth ladder on the patched build, original Bonsai 2 file, flag off

`llama-bench -m Ternary-Bonsai-2-27B-PTQ1_0.gguf -ngl 99 -fa 1 -ctk q4_0 -ctv q4_0 -d 0,16384,65536,131072
-r 2`, `pr-ptq1-mmv` build (`588318660`), raw output `results/kernel/partb/depth_ladder_{patched,prebuilt}.txt`:

| depth | pp512 prebuilt | pp512 patched | tg128 prebuilt | tg128 patched | tg gain |
|---|---|---|---|---|---|
| 0 | 269.1 | 267.8 | 26.43 | 40.54 | 1.53x |
| 16384 | 243.9 | 243.5 | 21.73 | 30.85 | 1.42x |
| 65536 | 191.7 | 191.1 | 14.34 | 17.86 | 1.25x |
| 131072 | 149.5 | 149.6 | 9.81 | 11.35 | 1.16x |

The gain shrinks with depth because the kernel change only touches the weight mat-vecs: at 131072 the
16 full-attention layers read 4.8 GB of q4_0 K/V per token and the flash-attention vector kernel, not the
projections, sets the pace (88 ms per token, of which the projections are about 25). This is also why the
MTP head helps more at depth in relative terms: a 3-token verify batch reads the K/V once for all three.

Served VRAM, patched build, original file, flag off, `-np 1 -ctk q4_0 -ctv q4_0 -fa on` (nvidia-smi
after `/health` ok, then after a 120-token answer): 65536 context 7,266 / 7,284 MiB (prebuilt fork: 7,282);
262144 context 11,682 / 11,700 MiB (prebuilt: 11,698). Decode on a short prompt 39.8 tok/s at both
(`results/kernel/partb/served_vram_patched.txt`). The kernel adds no VRAM: the planar activation buffer
replaces the q8_1 buffer of the same size.

### C. Third tensor-core attempt (time-boxed, not kept)

The per-column cost was attacked once more with the register-shuffle idea from failure 1, in three steps,
each correct against the CPU reference (47/47 MUL_MAT ptq1_0 with the path forced on):

1. Permuted cooperative decode. A quad (4 lanes) holds one row's 32 words in a fixed permutation of K,
   applied to the A and B fragments alike: lane t decodes trit group t alone (5 levels) and the 12 words
   of groups 4, 5 and qh are split so that even lanes run the pair group to level 2 while odd lanes run
   qh to level 1, the half-finished remainders cross the quad with 3 shuffles, and both finish with two
   more steps. 10 trit steps per row per lane instead of 14 (the redundant-decode variant) with no shared
   memory. Slots that mix sub-blocks across lanes are handled with masked fragments so every int32 sum is
   still a whole sub-block and the fold expression stays the dp4a kernel's. Padded partials (stride
   bpr + 1) remove the 8-way bank conflicts of the previous variant.
2. Software pipeline: the next item's 22 loads are issued before the current item's decode.
3. Two K blocks per item (two independent decode to mma chains per warp), 32 and 64 rows per CTA.

`test-backend-ops perf`, us per op (dp4a = the kernel in the branch):

| shape | n | dp4a | permuted decode | + pipeline | + 2 blocks per item |
|---|---|---|---|---|---|
| attn_qkv 10240x5120 | 2 | 52.3 | 90.4 | 83.4 | 78.8 |
| | 3 | 70.0 | 106.5 | 94.0 | 89.0 |
| | 4 | 85.0 | 104.1 | 102.9 | 102.4 |
| ffn_up 17408x5120 | 2 | 84.9 | 144.3 | 135.1 | 128.0 |
| | 3 | 115.1 | 175.0 | 153.0 | 145.2 |
| | 4 | 139.5 | 168.3 | 166.5 | 167.8 |

Best case 1.5x slower at 2 columns and 1.2x at 4, so it fails the keep rule (beat dp4a at 2 to 4 on these
two shapes) and the branch keeps the dp4a kernel. What the decomposition said (a diag switch that removed
one component at a time, attn_qkv, 2 columns, 101.5 us baseline with the switch in): no decode 82.7, no
weight loads 82.0, no mma 104.5, no B loads 97.7, no partial stores 98.9. Decode and weight loads are
serialized (removing either saves the same 19 us) and the tensor-core instructions themselves are free;
a 60 us floor stays that none of the components explain, which points at the dependent chain per item
(loads, 10 decode steps, 10 mma, 4 folds) with 16 warps per SM to hide it, since more chains per warp
(step 3) and prefetch (step 2) each bought 5 to 8%. The dp4a kernel has the same instruction budget per
row block but 4 independent row chains per lane and one dp4a per column-word instead of a fold, which is
why it stays ahead at 2 to 4 columns. Diff of the whole experiment:
`results/kernel/partc/mma_permuted_decode_experiment.patch` (opt-in via `GGML_CUDA_PTQ1_0_MMA=1`, not in
any branch). The pre-transposed weight layout was not attempted: it needs the MMQ tile loader, `get_rows`
and the buffer get/set path to understand the layout, beyond the box.

Working tree of `/root/llama.cpp-sudo` restored to `ecaa8b374` (clean), binaries rebuilt from it.

## The read

The wall was two things, and the fork's PTQ1_0 mat-vec had both: it read the quantized activations through
32 scattered 4-byte loads per column, and it let 88 of every 128 threads idle on the K = 5120 projections
that hold three quarters of the model's weights. Fixing the layout and the work split gives every 12GB
owner of Bonsai 2 a 1.52x faster single-token decode (26.1 to 39.8 tok/s) with no change to the file, and
makes a 3-token verify batch cost 1.55 single passes instead of 2.28. With `GGML_CUDA_BATCH_INVARIANT=1`
the logits of a token are the same bits whether it is decoded alone or verified inside a batch of up to 4,
so the grafted MTP head is now lossless: greedy output with the flag equals greedy output without it on all
three prompts at n-max 1, 2 and 3, and n-max 1 runs at 50.1 tok/s, twice the prebuilt fork's 25.0, with
+21% at 42K context. The per-column cost is still about 22% per extra token, above the 15% asked, so n-max 2
does not reach +40% on code lossless (+38%) and loses at depth under the knob; the fast-but-not-lossless
n-max 2 configuration gets +43% on code and +27% at depth. Answer: yes, the graft is now a real, lossless
speed win for 12GB owners at n-max 1, and the remaining gap to the n-max 2 targets is the per-column cost
of the dp4a mat-vec, which two tensor-core attempts did not beat at 2 to 4 columns on this card.
