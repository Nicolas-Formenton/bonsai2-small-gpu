import hashlib
import os
import struct

import pytest

import gguf_raw as gr


def sha(path):
    return gr.sha256_file(path)


def test_align_up():
    assert gr.align_up(0, 32) == 0
    assert gr.align_up(1, 32) == 32
    assert gr.align_up(32, 32) == 32
    assert gr.align_up(33, 32) == 64


def test_tensor_nbytes_known_types():
    assert gr.tensor_nbytes([5120], 0) == 5120 * 4          # F32
    assert gr.tensor_nbytes([5120, 248320], 143) == 5120 * 248320 // 128 * 28   # PTQ1_0, Bonsai output.weight
    assert gr.tensor_nbytes([10240, 5120], 8) == 10240 * 5120 // 32 * 34        # Q8_0, donor eh_proj
    assert gr.tensor_nbytes([256], 999) is None
    with pytest.raises(ValueError):
        gr.tensor_nbytes([100], 143)


def test_parse_synthetic_matches_gguf_py(synthetic_trunk):
    from gguf import GGUFReader
    ref = GGUFReader(synthetic_trunk)
    g = gr.read_header(synthetic_trunk)
    assert g.version == 3
    assert g.alignment == ref.alignment
    assert g.data_start == ref.data_offset
    assert len(g.tensors) == len(ref.tensors)
    assert g.arch == "qwen35"
    assert g.get("qwen35.block_count") == 2
    assert g.get("tokenizer.ggml.add_bos_token") is False
    assert g.get("prism.hadamard.weight_names") == ["output.weight", "blk.0.ffn_up.weight"]
    assert g.get("qwen35.rope.dimension_sections") == [11, 11, 10, 0]
    # gguf-py adds 3 virtual fields (GGUF.version, tensor_count, kv_count)
    assert len(g.kvs) == len(ref.fields) - 3
    for t, rt in zip(g.tensors, ref.tensors):
        assert t.name == rt.name
        assert t.dims == [int(x) for x in rt.shape]
        assert t.ttype == int(rt.tensor_type)
        assert t.offset == rt.data_offset - ref.data_offset
        assert t.nbytes == rt.n_bytes
        span = g.spans[t.name]
        assert span.length >= rt.n_bytes
        assert span.length - rt.n_bytes < g.alignment
    assert gr.check_spans(g) == []


def test_header_reserializes_byte_identical(synthetic_trunk):
    g = gr.read_header(synthetic_trunk)
    with open(synthetic_trunk, "rb") as f:
        original = f.read(g.data_start)
    assert gr.serialize_header(g.kvs, g.tensors, g.alignment) == original


def test_noop_copy_reproduces_sha256(synthetic_trunk, tmp_path):
    g = gr.read_header(synthetic_trunk)
    out = str(tmp_path / "copy.gguf")
    gr.write_gguf(out, g.kvs, gr.entries_of(g), g.alignment)
    assert os.path.getsize(out) == os.path.getsize(synthetic_trunk)
    assert sha(out) == sha(synthetic_trunk)


def test_unknown_type_id_round_trips(tmp_path):
    """A tensor with the PrismML PTQ1_0 id (143) is copied as opaque bytes."""
    kvs = [gr.KV("general.architecture", gr.STRING, "qwen35"), gr.KV("qwen35.block_count", gr.UINT32, 1)]
    blob_a = bytes(range(256)) * 7  # 1792 bytes = 64 blocks of 28 bytes -> 8192 elements
    blob_b = b"\x11" * 20480
    src = tmp_path / "src.bin"
    src.write_bytes(blob_a + blob_b)
    entries = [
        (gr.TensorInfo("blk.0.ffn_up.weight", [128, 64], 143, 0), gr.TensorData(str(src), 0, len(blob_a))),
        (gr.TensorInfo("blk.0.attn_norm.weight", [5120], 0, 0), gr.TensorData(str(src), len(blob_a), len(blob_b))),
    ]
    out = str(tmp_path / "custom.gguf")
    gr.write_gguf(out, kvs, entries)
    g = gr.read_header(out)
    assert [t.name for t in g.tensors] == ["blk.0.ffn_up.weight", "blk.0.attn_norm.weight"]
    assert g.tensors[0].ttype == 143
    assert g.tensors[0].nbytes == 128 * 64 // 128 * 28 == len(blob_a)
    assert g.tensors[0].offset == 0
    assert g.tensors[1].offset == gr.align_up(len(blob_a), 32)
    assert g.spans["blk.0.ffn_up.weight"].read()[:len(blob_a)] == blob_a
    assert g.spans["blk.0.attn_norm.weight"].read() == blob_b
    assert gr.check_spans(g) == []
    # and the copy of the copy is identical
    out2 = str(tmp_path / "custom2.gguf")
    gr.write_gguf(out2, g.kvs, gr.entries_of(g), g.alignment)
    assert sha(out2) == sha(out)


def test_all_kv_value_types_round_trip(tmp_path):
    kvs = [
        gr.KV("general.architecture", gr.STRING, "qwen35"),
        gr.KV("t.u8", gr.UINT8, 200), gr.KV("t.i8", gr.INT8, -7), gr.KV("t.u16", gr.UINT16, 60000),
        gr.KV("t.i16", gr.INT16, -300), gr.KV("t.u32", gr.UINT32, 4000000000), gr.KV("t.i32", gr.INT32, -5),
        gr.KV("t.f32", gr.FLOAT32, 9.999999974752427e-07), gr.KV("t.bool", gr.BOOL, True),
        gr.KV("t.u64", gr.UINT64, 1 << 40), gr.KV("t.i64", gr.INT64, -(1 << 40)), gr.KV("t.f64", gr.FLOAT64, 1e-300),
        gr.KV("t.arr_i32", gr.ARRAY, (gr.INT32, [-1, 1, 1, -1])),
        gr.KV("t.arr_str", gr.ARRAY, (gr.STRING, ["a", "", "\u00e9\u00e8"])),
        gr.KV("t.arr_nested", gr.ARRAY, (gr.ARRAY, [(gr.UINT8, [1, 2]), (gr.UINT8, [])])),
    ]
    src = tmp_path / "src.bin"
    src.write_bytes(b"\x01" * 64)
    entries = [(gr.TensorInfo("x", [16], 0, 0), gr.TensorData(str(src), 0, 64))]
    out = str(tmp_path / "kv.gguf")
    gr.write_gguf(out, kvs, entries)
    g = gr.read_header(out)
    assert [(k.key, k.vtype, k.value) for k in g.kvs] == [(k.key, k.vtype, k.value) for k in kvs]
    with open(out, "rb") as f:
        assert gr.serialize_header(g.kvs, g.tensors, g.alignment) == f.read(g.data_start)


# --- real files, skipped where absent -------------------------------------------------

def test_donor_header_parses_and_reserializes(donor_path):
    g = gr.read_header(donor_path)
    assert g.arch == "qwen35"
    assert g.get("qwen35.block_count") == 65
    assert g.get("qwen35.nextn_predict_layers") == 1
    names = {t.name for t in g.tensors}
    for n in ("blk.64.nextn.eh_proj.weight", "blk.64.nextn.enorm.weight", "blk.64.nextn.hnorm.weight",
              "blk.64.attn_q.weight", "blk.64.ffn_down.weight", "token_embd.weight", "output.weight"):
        assert n in names
    with open(donor_path, "rb") as f:
        original = f.read(g.data_start)
    assert gr.serialize_header(g.kvs, g.tensors, g.alignment) == original
    assert gr.check_spans(g) == []


def test_bonsai_header_matches_server_log(bonsai_path):
    """Facts from the node's llama-server load log and the fork's gguf-py dump."""
    g = gr.read_header(bonsai_path)
    assert g.file_size == 5946648928
    assert g.arch == "qwen35"
    assert g.get("qwen35.block_count") == 64
    assert g.kv("qwen35.nextn_predict_layers") is None
    assert len(g.tensors) == 851
    assert len(g.kvs) == 49
    assert g.get("general.file_type") == 143
    hist = {}
    for t in g.tensors:
        hist[t.ttype] = hist.get(t.ttype, 0) + 1
    assert hist == {143: 402, 0: 353, 30: 96}
    assert "token_embd.weight" in g.get("prism.hadamard.inverse_weight_names")
    assert "output.weight" in g.get("prism.hadamard.weight_names")
    with open(bonsai_path, "rb") as f:
        original = f.read(g.data_start)
    assert gr.serialize_header(g.kvs, g.tensors, g.alignment) == original
    assert gr.check_spans(g) == []
    last = max(g.tensors, key=lambda t: t.offset)
    assert g.data_start + last.offset + last.nbytes == g.file_size
