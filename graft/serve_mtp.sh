#!/bin/bash
# Ternary Bonsai 2 27B with the grafted Qwen 3.8 MTP head on a 12GB card.
# PrismML llama.cpp fork required (stock llama.cpp cannot read PTQ1_0).
# n-max 1 is the only setting that beat the flag off on an RTX 3060 12GB,
# see results/rtx3060.md. 131072 context fits in 10,638 MiB; 163840 fits with
# -ctkd q4_0 -ctvd q4_0 (11,726 MiB); 196608 needs the lean file and the patch.
MODEL="${1:-$HOME/models/bonsai2-27b/Ternary-Bonsai-2-27B-PTQ1_0-mtp.gguf}"
CTX="${2:-131072}"

llama-server -m "$MODEL" \
  -c "$CTX" -ngl 99 -fa on -np 1 \
  -ctk q4_0 -ctv q4_0 --jinja \
  --temp 1.0 --top-p 0.95 --top-k 20 \
  --spec-type draft-mtp --spec-draft-n-max 1 \
  --host 127.0.0.1 --port 8899
