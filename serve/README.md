# serve lines, one per vram tier

All lines use the PrismML llama.cpp fork (release prism-b10685 or newer, or the `pr-ptq1-mmv` branch of
github.com/sudoingX/llama.cpp for the faster PTQ1_0 decode). Stock llama.cpp does not run these files.
Every line: `-np 1` (one slot; four slots cost 450 MiB for nothing), `--jinja` (tool calls), the thinking
defaults `--temp 1.0 --top-p 0.95 --top-k 20`. Measured on the cards named in `../sweeps/`.

| tier | script | context | served VRAM | notes |
| --- | --- | --- | --- | --- |
| 8GB | `8gb.sh` | 65536 | about 7.3 GB | the Hermes floor; measured on 12GB, 8GB row pending |
| 12GB | `12gb.sh` | 262144 | 11.7 GB | the full native window fits, 0.6 GB spare |
| 12GB + MTP head | `12gb-mtp.sh` | 131072 | 10.6 GB | needs the merged file from `../graft/`; lossless with `GGML_CUDA_BATCH_INVARIANT=1` on the fork branch |
| 16GB + vision | `16gb-vision.sh` | 131072 | pending | adds `--mmproj`; row pending |
