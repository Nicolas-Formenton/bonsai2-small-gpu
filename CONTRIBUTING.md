# add your card

One row per card, same method, so rows compare.

1. Serve with the line for your tier from `serve/`, the PrismML fork release named in the README (or the
   `pr-ptq1-mmv` branch, say which).
2. Run `llama-bench -m Ternary-Bonsai-2-27B-PTQ1_0.gguf -ngl 99 -fa 1 -ctk q4_0 -ctv q4_0 -p 512 -n 128 -r 3`
   and, if you can, `-d 0,16384,65536,131072` for the depth curve.
3. Note the served VRAM at your context (`nvidia-smi --query-gpu=memory.used --format=csv,noheader`).
4. Open a PR that adds `sweeps/<your card>.md` with: card, VRAM, driver, build tag, the llama-bench lines
   verbatim, served VRAM, and one sentence on anything odd. Numbers only from runs you did yourself.
