# PR: CUDA: fast, batch-invariant small-batch PTQ1_0 mat-vec

Branch: `pr-ptq1-mmv`, five commits on top of `prism` (`9a9394a89`), 8 files, 596 insertions, 21 deletions.
Measured on one RTX 3060 12GB (sm_86, CUDA 12.4, driver 550.144.03) with Ternary Bonsai 2 27B.

## What

1. `Add: planar-transposed activation layout for the PTQ1_0 mat-vec path` (`quantize.cu`, `mmvq.cu`,
   new `mmvq-ptq1_0.cuh`). PTQ1_0 activations are quantized into a layout where the 128 quants a
   thread needs are 8 aligned 16-byte pieces plus one 16-byte piece of scales, adjacent lanes on
   adjacent pieces; same bytes per row and the same column stride as `block_q8_1`, so nothing in the
   launchers changes. Trit decode unchanged; the byte-wise `-1` is `(q + 0x7F7F7F7F) ^ 0x80808080`.
   nwarps pinned to 4 for all column counts; mmvq handles PTQ1_0 up to 8 columns. HIP unchanged.
2. `Add: dedicated PTQ1_0 mat-vec kernel with full lane utilization` (`mmvq-ptq1_0.cuh`,
   `mul_mat_vec_ptq1_0_pt`). For plain 2D MUL_MAT: (row group, K block) work items flattened so
   every thread has work, 4 rows per thread per K block, one fp32 partial per (row, column, K block)
   in dynamic shared memory, one warp per (row, column) reducing in a fixed order. The per-block
   partial is `d * sum_k d8_k * sumi_k` with exact integer `sumi_k` (`__fmaf_rn`, `__fmul_rn`), the
   order depends only on the weight shape: a column's result is bit-identical for every column count
   1 to 8. Fusion for one column as in the generic kernel; batched and MoE calls keep the generic
   kernel (which also reads the new layout).
3. `Test: add Bonsai 2 projection shapes to the mul_mat perf cases` (`tests/test-backend-ops.cpp`):
   PTQ1_0 and Q4_0 at the six Bonsai 2 projection shapes for 1, 2, 3, 4, 8 columns, plus the bf16
   [5120 x 48] gate projection.
4. `Add: GGML_CUDA_BATCH_INVARIANT for batch-invariant small-batch kernels` (`common.cuh`, `mmvf.cu`,
   `fattn.cu`, `fattn-common.cuh`). Off by default. When set: small F16/BF16 matrices take
   `mul_mat_vec_f` for 1 to 8 columns, flash attention with up to 8 queries takes the vector kernel,
   and its KV split is sized as for one query tile.
5. `Fix: use the mat-vec kernel for bf16 matrices under 64 rows at 2 to 8 columns` (`mmvf.cu`).

## Why

The fork's PTQ1_0 mat-vec charged 1.5 single-token passes for a 2-token batch and 2.3 for a
3-token batch, so speculative decoding (`--spec-type draft-mtp`) could not pay for its drafts on a
ternary model, and its logits changed with the batch size, so greedy output with the draft head
differed from greedy output without it. Two causes, both in the same kernel: the activations were
read as 32 scattered 4-byte loads per column out of 36-byte structs, and on the K = 5120 projections
(three quarters of the 27B's weights) 88 of every 128 threads sat idle because one thread owned a
whole 128-element block. `test-backend-ops perf` on a 4096 x 14336 PTQ1_0 matrix: 47 / 104 / 154 /
201 us for 1 / 2 / 3 / 4 columns; Q4_0 on the same shape 101 / 102 / 115 / 151.

## Before / after

`llama-bench`, `Ternary-Bonsai-2-27B-PTQ1_0` with the MTP head appended, 4096 context,
`-fa 1 -ctk q4_0 -ctv q4_0`, 8 repetitions:

| test | before, tok/s | after, tok/s | before, ms per batch | after, ms per batch | before vs pp1 | after vs pp1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| pp1 | 25.26 | 37.29 | 39.6 | 26.8 | 1.00 | 1.00 |
| pp2 | 34.10 | 61.08 | 58.7 | 32.7 | 1.48 | 1.22 |
| pp3 | 33.29 | 72.04 | 90.1 | 41.6 | 2.28 | 1.55 |
| pp4 | 35.26 | 81.03 | 113.5 | 49.4 | 2.87 | 1.84 |
| pp8 | 53.81 | 73.06 | 148.7 | 109.5 | 3.76 | 4.09 |
| pp16 | 102.61 | 102.16 | | | | MMQ path, unchanged |
| tg32 | 26.14 | 39.76 | 38.3 | 25.2 | | |

`GGML_CUDA_BATCH_INVARIANT=1` measures the same within noise (tg32 39.70).

Kernel level, `test-backend-ops perf`, us per call (before = the fork's kernel on the same layout
change, so this isolates the dedicated kernel; the layout change alone took the 4096 x 14336 case
from 47 / 104 / 154 / 201 to 46 / 55 / 65 / 75 us):

| shape (K x M) | 1 col | 2 cols | 3 cols | 4 cols | 8 cols |
| --- | ---: | ---: | ---: | ---: | ---: |
| attn_qkv 5120 x 10240, before | 72.4 | 75.0 | 100.4 | 150.7 | 219.0 |
| attn_qkv, after | 41.8 | 52.5 | 71.8 | 85.4 | 193.9 |
| ffn_up 5120 x 17408, before | 120.0 | 124.6 | 165.9 | 254.3 | 368.6 |
| ffn_up, after | 66.7 | 84.7 | 116.4 | 140.9 | 324.5 |
| ffn_down 17408 x 5120, before | 75.1 | 85.3 | 107.4 | 150.6 | 423.4 |
| ffn_down, after | 68.5 | 81.8 | 112.1 | 145.7 | 431.1 |
| attn_gate 5120 x 6144, before | 44.8 | 46.7 | 61.0 | 92.3 | 134.6 |
| attn_gate, after | 27.7 | 34.1 | 45.3 | 54.2 | 120.1 |

Speculative decoding end to end (`llama-server`, 131072 context, one slot, q4_0 K/V, thinking off,
client-side tok/s over streamed tokens, medians of 3 runs on 3 prompts):

| arm | code | prose | bash | overall | greedy text equals flag off |
| --- | ---: | ---: | ---: | ---: | --- |
| prebuilt, flag off | 25.2 | 25.0 | 25.0 | 25.0 | |
| this branch, flag off | 39.8 | 40.0 | 39.7 | 39.8 | |
| this branch, `draft-mtp` n-max 1, `GGML_CUDA_BATCH_INVARIANT=1` | 53.2 | 41.8 | 50.1 | 50.1 | yes, all three prompts |
| this branch, n-max 2, `GGML_CUDA_BATCH_INVARIANT=1` | 55.1 | 36.8 | 45.4 | 45.4 | yes |
| this branch, n-max 2, default kernels | 56.9 | 36.9 | 48.7 | 48.7 | no |

At 41.8K tokens of context: 22.2 tok/s flag off, 26.9 with n-max 1 (invariant mode), 28.3 with
n-max 2 on the default kernels.

## Correctness

`test-backend-ops test -o MUL_MAT -p type_a=ptq1_0`: 47 of 47 against the CPU reference, including
1 to 9 columns and batched shapes (generic kernel). `-o MUL_MAT_ID`: 77 of 77. The flag-off greedy
text changes versus the prebuilt build at near ties (the K summation order changed), as expected for
any kernel change; the batched-prefill check that exposed the prebuilt fork's own decode and prefill
paths disagreeing by up to 0.1 nats now shows bit-identical logprobs for 1 to 4 tokens.

## How to reproduce

```
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 -DLLAMA_BUILD_TESTS=ON
cmake --build build --target llama-bench test-backend-ops llama-server -j
./build/bin/test-backend-ops test -o MUL_MAT -b CUDA0 -p "type_a=ptq1_0"
./build/bin/test-backend-ops perf -o MUL_MAT -b CUDA0 -p "type_a=ptq1_0,type_b=f32,m=10240,n=(1|2|3|4|8),k=5120"
./build/bin/llama-bench -m Ternary-Bonsai-2-27B-PTQ1_0.gguf -ngl 99 -fa 1 -ctk q4_0 -ctv q4_0 -p 1,2,3,4,8,16 -n 32 -r 8
```

The invariance check (same prefix, last N tokens in one batch, compare the next-token logprobs) is
`tools/batch_numerics.py` in github.com/sudoingX/bonsai2-mtp against a running `llama-server`.

## Known limits

- The per-column cost is still about 22% of a single-token pass per extra column (pp2 1.22x, pp3
  1.55x of pp1). Removing the activation loads from the kernel (perf only) makes 1 to 8 columns cost
  the same, so the loads are the cost; broadcasting them across 8 lanes (warp tiles), more rows per
  thread, tighter launch bounds and two `mma.sync.m16n8k16.s8` variants (correct, bit-identical by
  construction) did not beat the dp4a kernel at 2 to 4 columns on this card. Details and diffs of
  the attempts are in the repo's `KERNEL_REPORT.md` and `results/kernel/experiments/`.
- `GGML_CUDA_BATCH_INVARIANT=1` makes batches of 1 to 4 bit-identical; batches of 5 to 8 agree with
  each other but differ from 1 to 4 by up to 0.03 nats on the tested steps (top-1 unchanged). The
  kernel that switches between 4 and 5 columns was not identified; the PTQ1_0 path, `mul_mat_vec_f`,
  the FWHT path, the gated-delta-net kernel and the flash-attention kernel choice and KV split were
  checked and excluded. Prompt processing (MMQ at 9+ tokens) is not covered either.
- In invariant mode the vector flash-attention kernel with 3 queries over 42K keys is slow, so
  n-max 2 loses at depth (21.1 vs 22.2 tok/s flag off) where the default kernels gain (28.3).
- MoE PTQ1_0 goes through the generic kernel with the new layout and passes MUL_MAT_ID correctness;
  it was not benchmarked (no ternary MoE file at hand).
- HIP keeps the previous layout and vec_dot (`ptq1_0_pt_enabled()` is false there); the new code is
  compiled out with `!defined(GGML_USE_HIP)`.
