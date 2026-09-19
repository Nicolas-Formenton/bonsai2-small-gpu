import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

# Real files, present on some machines only. Tests that need them skip otherwise.
DONOR_PATHS = [
    os.path.expanduser("~/models/qwen3.8-27b-dense/Qwen3.8-27B-Q4_K_M.gguf"),
    "/root/models/qwen3.8-27b/Qwen3.8-27B-UD-Q4_K_M.gguf",
]
BONSAI_PATHS = [
    "/root/models/bonsai2-27b/Ternary-Bonsai-2-27B-PTQ1_0.gguf",
    os.path.expanduser("~/models/bonsai2-27b/Ternary-Bonsai-2-27B-PTQ1_0.gguf"),
]


def first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None


@pytest.fixture(scope="session")
def donor_path():
    p = first_existing(DONOR_PATHS)
    if p is None:
        pytest.skip("donor GGUF not present on this machine")
    return p


@pytest.fixture(scope="session")
def bonsai_path():
    p = first_existing(BONSAI_PATHS)
    if p is None:
        pytest.skip("Bonsai 2 GGUF not present on this machine")
    return p


def make_synthetic_gguf(path, arch="qwen35", n_blocks=2, with_nextn=False, seed=0):
    """Small GGUF written with gguf-py (an independent writer) for round-trip tests.

    Layout mirrors the real files: token_embd, output_norm, output, blk.N.* with a
    mix of F32, F16, Q8_0 and Q4_K raw blobs, and an optional nextn block.
    """
    from gguf import GGUFWriter, GGMLQuantizationType

    rng = np.random.default_rng(seed)
    n_embd, n_vocab, n_ff = 256, 512, 384
    w = GGUFWriter(path, arch)
    w.add_name("synthetic")
    w.add_block_count(n_blocks + (1 if with_nextn else 0))
    w.add_uint32(f"{arch}.embedding_length", n_embd)
    w.add_float32(f"{arch}.attention.layer_norm_rms_epsilon", 1e-6)
    w.add_array(f"{arch}.rope.dimension_sections", [11, 11, 10, 0])
    w.add_array("prism.hadamard.weight_names", ["output.weight", "blk.0.ffn_up.weight"])
    w.add_bool("tokenizer.ggml.add_bos_token", False)
    w.add_string("tokenizer.chat_template", "{{ messages }}\n")
    w.add_array("tokenizer.ggml.tokens", [f"tok{i}" for i in range(64)])
    if with_nextn:
        w.add_uint32(f"{arch}.nextn_predict_layers", 1)

    def raw(name, shape, qtype):
        block, tsize = {GGMLQuantizationType.Q8_0: (32, 34), GGMLQuantizationType.Q4_K: (256, 144)}[qtype]
        # gguf-py wants a uint8 array shaped like the tensor with the last dim in bytes
        rows = int(np.prod(shape[:-1])) if len(shape) > 1 else 1
        row_bytes = shape[-1] // block * tsize
        data = rng.integers(0, 256, size=(rows, row_bytes), dtype=np.uint8)
        w.add_tensor(name, data, raw_dtype=qtype)

    w.add_tensor("token_embd.weight", rng.standard_normal((n_vocab, n_embd), dtype=np.float32).astype(np.float16))
    w.add_tensor("output_norm.weight", rng.standard_normal((n_embd,), dtype=np.float32))
    raw("output.weight", (n_vocab, n_embd), GGMLQuantizationType.Q4_K)
    total = n_blocks + (1 if with_nextn else 0)
    for i in range(total):
        w.add_tensor(f"blk.{i}.attn_norm.weight", rng.standard_normal((n_embd,), dtype=np.float32))
        raw(f"blk.{i}.ffn_up.weight", (n_ff, n_embd), GGMLQuantizationType.Q8_0)
        raw(f"blk.{i}.ffn_down.weight", (n_embd, n_ff), GGMLQuantizationType.Q4_K)
        if with_nextn and i == total - 1:
            raw(f"blk.{i}.nextn.eh_proj.weight", (n_embd, 2 * n_embd), GGMLQuantizationType.Q8_0)
            w.add_tensor(f"blk.{i}.nextn.enorm.weight", rng.standard_normal((n_embd,), dtype=np.float32))
            w.add_tensor(f"blk.{i}.nextn.hnorm.weight", rng.standard_normal((n_embd,), dtype=np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return path


@pytest.fixture
def synthetic_trunk(tmp_path):
    return make_synthetic_gguf(str(tmp_path / "trunk.gguf"), n_blocks=2, with_nextn=False, seed=1)


@pytest.fixture
def synthetic_donor(tmp_path):
    return make_synthetic_gguf(str(tmp_path / "donor.gguf"), n_blocks=2, with_nextn=True, seed=2)
