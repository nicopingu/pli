"""PLI (lot L3 revised, PLI_EXTENDED_DRAFT_LIST): the extended list adds the model head's exact rows to the reduced heads."""

import types

import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")

from tensorfold.kernels.qwen.dense.v1 import lane_qmm  # noqa: E402
from tensorfold.drafters.dflash_drafter import DFlashDrafter  # noqa: E402
from tensorfold.pli import extended_list  # noqa: E402

EXTRA = [1500, 1501, 2047, 3000, 3999]          # 3999 is already in the list (3968 to 4096): not added


def _drafter(n=4096, k=512, bits=4, group=64):
    mx.random.seed(11)
    holder = nn.Sequential(nn.Linear(k, n, bias=False))
    holder.set_dtype(mx.bfloat16)
    nn.quantize(holder, group_size=group, bits=bits)
    mx.eval(holder.parameters())
    head = holder.layers[0]
    config = types.SimpleNamespace(output_multiplier=1.0, final_logit_softcapping=None)
    model = types.SimpleNamespace(lm_head=head, config=config)
    model.compute_logits = lambda hidden: head(hidden) * config.output_multiplier
    d = object.__new__(DFlashDrafter)
    d.model = model
    d.draft_vocab = ((0, 1024), (3968, 4096))
    return d, holder


@pytest.fixture
def extra(monkeypatch):
    monkeypatch.setattr(extended_list, "FILE", "trial.txt")
    monkeypatch.setattr(extended_list, "_ids", sorted(EXTRA))


def _close(logits, full, ids):
    kept = mx.take(full, ids, axis=-1).astype(mx.float32)
    assert logits.shape == kept.shape
    assert float(mx.max(mx.abs(logits.astype(mx.float32) - kept)).item()) <= 0.02 * float(mx.max(mx.abs(kept)).item())


def test_outside_pads_without_duplicates(extra):
    spans = ((0, 1024), (3968, 4096))
    assert extended_list.outside(spans, 4096) == [1500, 1501, 2047, 3000]
    h = extended_list.outside(spans, 4096, 32)
    assert len(h) == 32 and len(set(h)) == 32 and h == sorted(h)
    assert all(not (0 <= t < 1024 or 3968 <= t < 4096) for t in h)
    assert {1500, 1501, 2047, 3000} <= set(h)


def test_off_nothing_changes(monkeypatch):
    monkeypatch.setattr(extended_list, "FILE", "")
    monkeypatch.setattr(extended_list, "_ids", None)
    d, _ = _drafter()
    hidden = (mx.random.normal((1, 4, 512)) * 0.5).astype(mx.bfloat16)
    _, ids = d.candidate_logits(hidden)
    assert ids.tolist() == list(range(1024)) + list(range(3968, 4096))


def test_head_as_views_with_extended_list(extra):
    d, _ = _drafter()                                 # head not laid out in tiles
    hidden = (mx.random.normal((1, 4, 512)) * 0.5).astype(mx.bfloat16)
    full = d.model.compute_logits(hidden)
    logits, ids = d.candidate_logits(hidden)
    assert ids.tolist() == list(range(1024)) + list(range(3968, 4096)) + [1500, 1501, 2047, 3000]
    _close(logits, full, ids)


@pytest.mark.parametrize("bits, group", [(4, 64), (4, 32)])
def test_head_in_tiles_with_extended_list(extra, bits, group):
    from tensorfold.families.qwen3_5 import tensor_units

    if not tensor_units():                     # tiled kernels: Metal 4 tensor units (M5-generation GPU)
        pytest.skip("needs Metal 4 tensor units (an M5-generation GPU)")
    try:
        d, holder = _drafter(bits=bits, group=group)
        lane_qmm.install(holder, rows=lane_qmm.MAX_ROWS)
        hidden = (mx.random.normal((1, 16, 512)) * 0.5).astype(mx.bfloat16)
        full = d.model.compute_logits(hidden)
        logits, ids = d.candidate_logits(hidden)
        mx.eval(full, logits, ids)
    except RuntimeError as exc:  # no Metal 4 tensor units on this machine
        lane_qmm.uninstall()
        pytest.skip(str(exc).splitlines()[0][:80])
    try:
        got = ids.tolist()
        assert got[:1152] == list(range(1024)) + list(range(3968, 4096))
        assert len(got) == 1152 + 32 and {1500, 1501, 2047, 3000} <= set(got[1152:])
        _close(logits, full, ids)
    finally:
        lane_qmm.uninstall()


def test_mtp_list_extended(extra, monkeypatch):
    from tensorfold.families.qwen4_exp import draft_head

    base = draft_head.draft_ids()
    assert set(EXTRA) <= set(base.tolist())
    assert len(base) % 64 == 0
