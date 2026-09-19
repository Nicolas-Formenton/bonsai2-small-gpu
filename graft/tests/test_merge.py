import os

import pytest

import extract_head as eh
import gguf_raw as gr
import merge as mg
from conftest import make_synthetic_gguf


def sha(path):
    return gr.sha256_file(path)


def test_extract_head_selects_last_block_and_adds_embed_tokens(synthetic_donor, tmp_path):
    g = gr.read_header(synthetic_donor)
    n_all, n_nextn, idx = eh.head_block_indices(g)
    assert (n_all, n_nextn, idx) == (3, 1, [2])
    entries, notes = eh.head_entries(g)
    names = [e[0].name for e in entries]
    assert all(n.startswith("blk.2.") for n in names)
    assert "blk.2.nextn.eh_proj.weight" in names
    assert "blk.2.nextn.enorm.weight" in names
    assert "blk.2.nextn.embed_tokens.weight" in names
    assert "blk.2.nextn.shared_head_head.weight" not in names
    emb = dict((e[0].name, e) for e in entries)["blk.2.nextn.embed_tokens.weight"]
    src = g.tensor("token_embd.weight")
    assert emb[0].dims == src.dims and emb[0].ttype == src.ttype
    assert emb[1].abs_offset == g.spans["token_embd.weight"].abs_offset
    assert len(notes) == 1
    entries2, _ = eh.head_entries(g, embed_tokens=False, shared_head=True)
    names2 = [e[0].name for e in entries2]
    assert "blk.2.nextn.embed_tokens.weight" not in names2
    assert "blk.2.nextn.shared_head_head.weight" in names2


def test_extract_head_writes_valid_file(synthetic_donor, tmp_path):
    out = str(tmp_path / "head.gguf")
    assert eh.main([synthetic_donor, out]) == 0
    h = gr.read_header(out)
    assert h.arch == "qwen35"
    assert h.get("qwen35.nextn_predict_layers") == 1
    assert h.get("graft.head_blocks") == [2]
    assert gr.check_spans(h) == []
    g = gr.read_header(synthetic_donor)
    for t in h.tensors:
        src_name = "token_embd.weight" if t.name.endswith("embed_tokens.weight") else t.name
        assert h.spans[t.name].read()[:g.spans[src_name].length] == g.spans[src_name].read()


def test_merge_then_strip_round_trips(synthetic_trunk, synthetic_donor, tmp_path):
    head = str(tmp_path / "head.gguf")
    merged = str(tmp_path / "merged.gguf")
    stripped = str(tmp_path / "stripped.gguf")
    assert eh.main([synthetic_donor, head]) == 0
    assert mg.main([synthetic_trunk, head, merged]) == 0

    t = gr.read_header(synthetic_trunk)
    h = gr.read_header(head)
    m = gr.read_header(merged)
    assert len(m.tensors) == len(t.tensors) + len(h.tensors)
    assert m.get("qwen35.block_count") == t.get("qwen35.block_count") + 1 == 3
    assert m.get("qwen35.nextn_predict_layers") == 1
    assert m.get("graft.head_blocks") == [2]
    assert gr.check_spans(m) == []
    # trunk tensors keep name, dims, type, offset and bytes
    for a, b in zip(t.tensors, m.tensors):
        assert (a.name, a.dims, a.ttype, a.offset) == (b.name, b.dims, b.ttype, b.offset)
        assert t.spans[a.name].read() == m.spans[b.name].read()[:t.spans[a.name].length]
    # head tensors follow, renamed to the new block index (same here: donor blk.2 -> blk.2)
    for a, b in zip(h.tensors, m.tensors[len(t.tensors):]):
        assert a.name == b.name
        assert h.spans[a.name].read()[:h.spans[a.name].length] == m.spans[b.name].read()[:h.spans[a.name].length]
    # every other kv is unchanged and in the same order
    kept = [kv for kv in m.kvs if kv.key not in ("qwen35.nextn_predict_layers",) and not kv.key.startswith("graft.")]
    assert [(k.key, k.vtype) for k in kept] == [(k.key, k.vtype) for k in t.kvs]

    assert mg.main(["--strip", merged, stripped]) == 0
    assert os.path.getsize(stripped) == os.path.getsize(synthetic_trunk)
    assert sha(stripped) == sha(synthetic_trunk)


def test_merge_renames_head_block_to_next_free_index(tmp_path):
    trunk = make_synthetic_gguf(str(tmp_path / "trunk4.gguf"), n_blocks=4, with_nextn=False, seed=3)
    donor = make_synthetic_gguf(str(tmp_path / "donor2.gguf"), n_blocks=2, with_nextn=True, seed=4)
    head = str(tmp_path / "head.gguf")
    merged = str(tmp_path / "merged.gguf")
    stripped = str(tmp_path / "stripped.gguf")
    assert eh.main([donor, head]) == 0
    assert mg.main([trunk, head, merged]) == 0
    m = gr.read_header(merged)
    assert m.get("qwen35.block_count") == 5
    assert m.get("graft.head_blocks") == [4]
    head_names = [t.name for t in m.tensors if t.name.startswith("blk.4.")]
    assert "blk.4.nextn.eh_proj.weight" in head_names
    assert not any(t.name.startswith("blk.2.nextn") for t in m.tensors)
    assert mg.main(["--strip", merged, stripped]) == 0
    assert sha(stripped) == sha(trunk)


def test_merge_refuses_arch_mismatch_without_force(tmp_path):
    trunk = make_synthetic_gguf(str(tmp_path / "trunk.gguf"), arch="qwen35", n_blocks=2, seed=5)
    donor = make_synthetic_gguf(str(tmp_path / "donor.gguf"), arch="qwen3next", n_blocks=2, with_nextn=True, seed=6)
    head = str(tmp_path / "head.gguf")
    assert eh.main([donor, head]) == 0
    with pytest.raises(SystemExit):
        mg.plan_merge(gr.read_header(trunk), gr.read_header(head))
    kvs, entries, s = mg.plan_merge(gr.read_header(trunk), gr.read_header(head), force_arch=True)
    keys = [k.key for k in kvs]
    assert "qwen35.nextn_predict_layers" in keys
    assert "qwen3next.nextn_predict_layers" not in keys


def test_merge_refuses_double_graft(synthetic_trunk, synthetic_donor, tmp_path):
    head = str(tmp_path / "head.gguf")
    merged = str(tmp_path / "merged.gguf")
    assert eh.main([synthetic_donor, head]) == 0
    assert mg.main([synthetic_trunk, head, merged]) == 0
    with pytest.raises(SystemExit):
        mg.plan_merge(gr.read_header(merged), gr.read_header(head))


# --- real files, skipped where absent -------------------------------------------------

def test_real_donor_head_extracts(donor_path, tmp_path):
    g = gr.read_header(donor_path)
    n_all, n_nextn, idx = eh.head_block_indices(g)
    assert (n_all, n_nextn, idx) == (65, 1, [64])
    entries, notes = eh.head_entries(g)
    names = sorted(e[0].name for e in entries)
    expected = sorted([
        "blk.64.attn_k.weight", "blk.64.attn_k_norm.weight", "blk.64.attn_norm.weight", "blk.64.attn_output.weight",
        "blk.64.attn_q.weight", "blk.64.attn_q_norm.weight", "blk.64.attn_v.weight", "blk.64.ffn_down.weight",
        "blk.64.ffn_gate.weight", "blk.64.ffn_up.weight", "blk.64.nextn.eh_proj.weight", "blk.64.nextn.enorm.weight",
        "blk.64.nextn.hnorm.weight", "blk.64.nextn.shared_head_norm.weight", "blk.64.post_attention_norm.weight",
        "blk.64.nextn.embed_tokens.weight",
    ])
    assert names == expected
    by_name = {e[0].name: e[0] for e in entries}
    assert by_name["blk.64.nextn.eh_proj.weight"].dims == [10240, 5120]
    assert by_name["blk.64.nextn.embed_tokens.weight"].dims == [5120, 248320]
    # write it, it is small (about 1 GB with the embedding table)
    out = str(tmp_path / "head.gguf")
    assert eh.main([donor_path, out]) == 0
    h = gr.read_header(out)
    assert len(h.tensors) == 16
    assert gr.check_spans(h) == []
    assert h.spans["blk.64.nextn.enorm.weight"].read()[:20480] == g.spans["blk.64.nextn.enorm.weight"].read()[:20480]
