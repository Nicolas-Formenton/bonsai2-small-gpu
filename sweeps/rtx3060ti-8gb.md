# Ternary Bonsai 2 27B, RTX 3060 Ti 8GB

All numbers from one RTX 3060 Ti 8GB on Windows 11, one machine, 2026-09-19, single
slot, q4_0 K/V cache, flash attention on, `-ngl 99`, driver 616.92, CUDA UMD 13.4.
Numbers only from runs done on this machine.

This card is **not** the RTX 3060 12GB. It is the 3060 **Ti**: 8GB of VRAM but
**448 GB/s** of memory bandwidth against the 3060's 360 GB/s. The repo lists the 8GB
tier row as pending (`serve/README.md`, measured on 12GB). This is that row, on the
Ti, with both binaries measured on the same card, in the same session.

## Binaries

- **stock**: PrismML llama.cpp fork, prebuilt release `prism-b10685` (`7dffb158d`,
  build 10685), windows-cuda-13.3-x64. The same fork revision the 12GB sweep used.
- **patched**: the fork at branch `pr-ptq1-mmv` of `sudoingX/llama.cpp` (`5883186`,
  build 1), built on this machine: MSVC 14.44.35207, CUDA Toolkit 13.4.59,
  `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86`. Three traps hit and fixed while
  building, all in Notes.

## Files

- `Ternary-Bonsai-2-27B-PTQ1_0.gguf`, 5,946,648,928 bytes. Size matches the value
  in `graft/recipe.txt`.

## A/B on the same card

Same model, same flags, same session, only the binary differs.

```
llama-bench -m Ternary-Bonsai-2-27B-PTQ1_0.gguf -ngl 99 -fa 1 -ctk q4_0 -ctv q4_0 -p 512 -n 128 -r 3 -d 0,16384,65536
```

| test | stock | patched | delta |
| --- | ---: | ---: | ---: |
| pp512 | 335.07 ± 2.01 | 329.43 ± 2.01 | -1.7% |
| tg128 | 32.44 ± 0.05 | **43.40 ± 0.07** | **+33.8%** |
| pp512 @ d16384 | 311.76 ± 1.17 | 303.82 ± 0.66 | -2.5% |
| tg128 @ d16384 | 26.61 ± 0.03 | **34.47 ± 0.02** | **+29.5%** |
| pp512 @ d65536 | 210.12 ± 0.87 | 231.75 ± 0.89 | +10.3% |
| tg128 @ d65536 | 17.13 ± 0.09 | **20.96 ± 0.04** | **+22.4%** |

Decode gains land below the 12GB row's figures (+53 / +42 / +25) while the patched
absolute numbers land above it (43.40 against 40.47 on the flat row). Not
contradictory: the stock row scales with memory bandwidth (448/360 = 1.2444, measured
1.233 at tg128) and the patched row does not (1.072). A kernel that stops leaving two
thirds of the lanes idle is no longer purely memory-bound, so extra bandwidth buys
less of it.

Verbatim, patched binary:

| model | size | params | backend | ngl | type_k | type_v | fa | test | t/s |
| --- | ---: | ---: | --- | --: | ---: | ---: | --: | --- | ---: |
| qwen35 27B PTQ1_0 - 1.75 bpw ternary (group 128) | 5.53 GiB | 26.90 B | CUDA | 99 | q4_0 | q4_0 | 1 | pp512 | 329.43 ± 2.01 |
| qwen35 27B PTQ1_0 - 1.75 bpw ternary (group 128) | 5.53 GiB | 26.90 B | CUDA | 99 | q4_0 | q4_0 | 1 | tg128 | 43.40 ± 0.07 |
| qwen35 27B PTQ1_0 - 1.75 bpw ternary (group 128) | 5.53 GiB | 26.90 B | CUDA | 99 | q4_0 | q4_0 | 1 | pp512 @ d16384 | 303.82 ± 0.66 |
| qwen35 27B PTQ1_0 - 1.75 bpw ternary (group 128) | 5.53 GiB | 26.90 B | CUDA | 99 | q4_0 | q4_0 | 1 | tg128 @ d16384 | 34.47 ± 0.02 |
| qwen35 27B PTQ1_0 - 1.75 bpw ternary (group 128) | 5.53 GiB | 26.90 B | CUDA | 99 | q4_0 | q4_0 | 1 | pp512 @ d65536 | 231.75 ± 0.89 |
| qwen35 27B PTQ1_0 - 1.75 bpw ternary (group 128) | 5.53 GiB | 26.90 B | CUDA | 99 | q4_0 | q4_0 | 1 | tg128 @ d65536 | 20.96 ± 0.04 |

## By allocated context, not prompt depth

`llama-bench -d` fills the context with tokens, which costs a full prefill per
measurement. On this card a 131072-token prefill runs at about 21 tok/s, so a single
depth row costs roughly 78 minutes and a full `-d 131072` sweep would have taken hours.
A different question is cheaper and closer to how the model is actually served: keep
the context **allocated** and send a short prompt. That is what an agent does. Three
identical requests per arm, 150 generated tokens each:

| arm | ctx | run1 | run2 | run3 | median |
| --- | ---: | ---: | ---: | ---: | ---: |
| stock | 65536 | 32.65 | 31.86 | 31.28 | 31.86 |
| patched | 65536 | **46.03** | **44.89** | **46.08** | **45.67** (**+43.0%**) |
| stock | 131072 | 16.87 | 17.23 | 17.24 | 17.23 |
| patched | 131072 | 17.46 | 17.56 | 17.31 | 17.46 (+1.3%) |
| stock | 262144 | 15.21 | 15.45 | 15.04 | 15.23 |
| patched | 262144 | 15.34 | 15.68 | 15.54 | 15.52 (+1.9%) |

| ctx | stock | patched | delta |
| ---: | ---: | ---: | ---: |
| 65536 | 31.86 | **45.67** | **+43.0%** |
| 131072 | 17.23 | 17.46 | +1.3% |
| 262144 | 15.23 | 15.52 | +1.9% |

The kernel change is worth **+43% at a 65536 window** and effectively nothing at 131072
and 262144, even though the prompt stays short in all three. Note this table is not
comparable row-for-row with the `-d` table above: there, "d65536" means a **filled**
prompt of 65536 tokens, so attention reads 65536 positions per token; here it means a
65536 **window** with a short prompt, so attention reads about 40. Different work, and
it is why the same depth reads +22% in one table and +43% in the other.

Practical consequence for a 8GB card: the same model decodes at **45.7 tok/s in a 64K
window** and **15.5 tok/s in a 256K window**. Context length is the dominant cost, more
than any kernel change available today.

**A single-run caution worth recording:** the first pass of this table was one run per
arm, and it reported the patched binary **11.5% slower** at 262144 (14.18 against
16.03). Repeating it three times showed that reading to be an outlier; the patched arm
is consistently slightly faster. One sample per arm is not enough on this machine.

## Same method as the repo (their `graft/probe.py`)

Run against a live server with the repo's own client (`graft/probe.py`, sha256
`d9ea0c9fb1043593`), unmodified: 3 prompts (code / prose / bash), 3 runs each, 400
max tokens, `enable_thinking: false`, client tok/s over streamed deltas with
time-to-first-token excluded, per-prompt medians. This is the arm that is directly
comparable with `sweeps/rtx3060.md`.

| window | stock | patched | delta | VRAM |
| ---: | ---: | ---: | ---: | ---: |
| 65536 | 31.6 | **40.3** | **+27.5%** | 7711 MiB |
| 131072 | 14.3 | 14.8 | +3.5% | 7711 MiB |
| 262144 | 14.3 | **14.8** | +3.5% | 7711 MiB |

Per-prompt detail, which also shows the prompt type barely matters here:

| arm | ctx | code | prose | bash |
| --- | ---: | ---: | ---: | ---: |
| stock | 65536 | 31.7 | 31.6 | 31.3 |
| stock | 131072 | 14.3 | 14.3 | 14.1 |
| stock | 262144 | 14.3 | 14.4 | 14.1 |
| patched | 65536 | 40.3 | 40.3 | 40.1 |
| patched | 131072 | 14.8 | 14.8 | 14.6 |
| patched | 262144 | 14.8 | 14.8 | 14.7 |

Note the delta here (+27.5% at 65536) is smaller than the same card, same binary pair
measured through `llama-bench` tg128 (+33.8%) and through the server's own `tg`
(+43.0%). The three measure different things: `llama-bench` times a fixed 128-token
run, the server `tg` is its internal decode rate, and `probe.py` is what a client
actually receives with TTFT excluded. Where they disagree, prefer the client number.

## The window step at ~114688

A scan at 65536 / 81920 / 98304 / 114688, same client, same flags, both binaries, same
session:

| ctx | stock VRAM (used / free) | stock tok/s | patched VRAM (used / free) | patched tok/s |
| ---: | --- | ---: | --- | ---: |
| 65536 | 7737 / 288 MiB | 31.8 | 7477 / 548 MiB | **42.9** |
| 81920 | 7823 / 202 MiB | 31.5 | 7845 / 180 MiB | **42.8** |
| 98304 | 7936 / 89 MiB | 30.6 | 7897 / 128 MiB | **42.8** |
| 114688 | 7958 / 67 MiB | 18.0 | 7879 / 146 MiB | 17.5 |
| 131072 (from the tables above) | 7712 / ~480 MiB | 14.3 | 7711 / ~480 MiB | 14.8 |

Four things fall out of this:

**1. The step is at ~114688, not at 64K.** Going from 65536 to 98304 costs 4% (31.8 to
30.6). The 41% drop happens one step later. A 96K window is close to free on this card.

**2. Both binaries fall at the same place.** 42.8 at 98304 to 17.5 at 114688 is a 59%
drop, at the same window where the stock binary drops 41%.

**3. They converge at the bottom.** Above the step, 18.0 and 17.5 are the same number.
The kernel's +27 to +43% does not survive it, which is exactly why the 131072 and
262144 rows read +3.5% and not +27.5%.

**4. Free VRAM does not explain it.** This is the finding that killed my own
hypothesis, and the patched arm is the proof: at 114688 it has **146 MiB free** — more
headroom than the stock arm had at 98304 (89 MiB) while running *fast*. A memory
ceiling cannot be the cause if the arm with more memory collapses at the same window.

**I proposed the VRAM explanation before running this scan, and the scan killed it.**
The cause is unidentified. Two candidate mechanisms were checked and do not fit: no
fallback or unsupported-path line appears in the server log at either 98304 or 114688,
and 18 tok/s is far above what a CPU decode of a 5.5 GiB model would produce, so a
plain attention-on-CPU fallback does not match either. What is measured is the step,
its position, that it is binary-independent, and that the two binaries converge above
it. The mechanism is left open.

**Practical advice for this card: serve 98304.** It costs 4% against 65536 and buys
50% more window. Everything above ~112K costs most of the speed, and with either
binary.

## Served VRAM

`nvidia-smi --query-gpu=memory.used --format=csv,noheader`, after load, before requests:

| arm | ctx=65536 | ctx=131072 | ctx=262144 |
| --- | ---: | ---: | ---: |
| stock | 7706 MiB | 7712 MiB | 7718 MiB |
| patched | 7711 MiB | 7711 MiB | 7711 MiB |

**262144 fits in under 7.7 GB of a 8 GB card** with `-ctk q4_0 -ctv q4_0`. The 12GB
address space is not needed for the full native window: Bonsai 2 is hybrid, with 16
full-attention layers out of 64, so only those hold a K/V cache. The 8GB tier does not
have to be a 64K tier.

"Fits", though, is not the same as "serves well", and the two must not be read
together as a recommendation. Loading at 262144 leaves very little free and decode
drops from 31.8 tok/s at a 65536 window to 14.3. The card can hold the full native
window; it cannot hold it *and* keep its shallow-window speed. See "The window step at
~114688" above. Pick the window for the workload, not for the address space.

## Stock vs the 12GB reference row

| test | 3060 12GB (repo) | 3060 Ti 8GB, stock | ratio |
| --- | ---: | ---: | ---: |
| pp512 | 269.6 | 335.07 | 1.243 |
| tg128 | 26.32 | 32.44 | 1.233 |
| tg128 @ d16384 | 21.7 | 26.61 | 1.226 |
| tg128 @ d65536 | 14.3 | 17.13 | 1.198 |

Memory bandwidth ratio 448/360 = **1.2444**. Every stock row tracks it within noise.
On the stock binary the 8GB Ti is 20 to 24 percent faster than the 12GB 3060 in decode
and prefill, despite having less VRAM.

**But that ordering does not hold at deep windows.** Same client, same method, ours
against the repo's 12GB row, both flag-off:

| window | 3060 12GB (repo) | 3060 Ti 8GB (here) |
| ---: | ---: | ---: |
| flat | 26.32 | 32.44 |
| 131072 | 25.0 | 14.3 |

The 12GB card holds 25.0 tok/s at 131072 against 26.32 flat, essentially no loss. This
card loses 55 percent. Bandwidth explains the flat row and nothing after it; the
12GB row's advantage at depth is not bandwidth, and the likely variable is VRAM
headroom (71% used there, 94% here). Anyone choosing between these two cards for
Bonsai 2 at long context should read that table before the first one.

## Reproducibility

The stock arm was measured twice in the llama-bench table, in two separate sessions
hours apart:

| test | first run | A/B run | spread |
| --- | ---: | ---: | ---: |
| tg128 | 32.72 | 32.44 | 0.9% |
| tg128 @ d16384 | 26.28 | 26.61 | 1.3% |
| tg128 @ d65536 | 17.29 | 17.13 | 0.9% |

And the allocated-context table was measured twice, by two different harnesses (a
single run per arm, then three runs per arm):

| arm / ctx | single pass | 3-run median | spread |
| --- | ---: | ---: | ---: |
| stock @ 65536 | 31.57 | 31.86 | 0.9% |
| stock @ 131072 | - | 17.23 | - |
| stock @ 262144 | 16.03 | 15.23 | 5.0% |
| patched @ 65536 | 45.86 | 45.67 | 0.4% |
| patched @ 131072 | - | 17.46 | - |
| patched @ 262144 | 14.18 | 15.52 | 9.5% |

Everything reproduces inside 1.3 percent except the patched 262144 figure, which is
the same outlier discussed above. Treat the 262144 decode numbers as within about 5
percent, and everything else as within about 1.

## Notes

- **Build trap 1**: `Could not find nvcc executable in any searched paths`. Passing the
  toolkit path is not enough, and the path is not accepted in POSIX form
  (`/c/Program Files/...`) because cmake is a native binary. Pass
  `-DCUDAToolkit_ROOT=C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.4` and
  `-DCMAKE_CUDA_COMPILER=<toolkit>/bin/nvcc.exe`.
- **Build trap 2**: with the toolkit found, MSBuild then failed with
  `The CUDA Toolkit directory '' does not exist` from `CUDA 13.4.targets`, which reads
  the `CUDA_PATH` environment variable. A shell started before the CUDA install never
  inherited it, and the message reads as if CUDA were missing. Export `CUDA_PATH`
  before configuring.
- **Runtime trap**: the built executables need `cudart64_13.dll`, `cublas64_13.dll`
  and `cublasLt64_13.dll`. In CUDA 13 these live in `bin/x64/`, not `bin/`. Without
  them the exe exits with code 127 and no message. The prebuilt release bundles them.
- `-fa 1` is accepted by llama-bench on this build and is recorded as `fa = 1`.
- No `--no-mmap`, no `-ub` change, no power limit change, no overclock.
- The prefill row at d65536 reads **+10.3%** for the patched binary. The kernel PR does
  not touch the prefill path, so this is either noise or a side effect of the
  batch-invariant change with a full cache. Reported as an open observation, not a
  claim.
- The `-d 131072` row is deliberately absent. The `-d` levels are "if you can" in
  `CONTRIBUTING.md`, and on this card 131072 costs about 78 minutes of prefill per
  measurement because llama-bench re-fills from scratch for each one. The
  allocated-context table above answers the same question at that depth for a fraction
  of the cost.
