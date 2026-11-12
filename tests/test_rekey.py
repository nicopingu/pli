"""PLI 4.2.0: the release number leaves the checkpoint key when the code is in it; old checkpoints are re-keyed in place."""

from __future__ import annotations

import os

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from tensorfold.engine import prefix_snapshots as ps  # noqa: E402
from tensorfold.pli import code_key, rekey  # noqa: E402

OLD = "/models/qwen|mlx=0.32.3|kernels=k1|prefill=p|tensorfold=4.1.2|pli=abc|points=4"


class Layer:
    def __init__(self) -> None:
        self.state_a = mx.arange(12).reshape(3, 4).astype(mx.float16)
        self.history = np.array([[11, 12]], dtype=np.int64)
        self.offset = 7


def test_release_part():
    assert code_key.release_part("4.2.0", enabled=True) == ""
    assert code_key.release_part("4.2.0", enabled=False) == "|tensorfold=4.2.0"


def test_new_key_drops_the_release_and_adds_the_code():
    new = rekey.new_key(OLD, "4.1.2", "0123456789ab")
    assert new == "/models/qwen|mlx=0.32.3|kernels=k1|prefill=p|pli=abc|points=4|code=0123456789ab"
    assert rekey.new_key(new, "4.1.2", "ffffffffffff").endswith("|points=4|code=ffffffffffff")


def test_move_keeps_tensors_tokens_and_age(tmp_path):
    ps.save_snapshot(tmp_path, OLD, [1, 2, 3], [Layer()])
    ps.save_snapshot(tmp_path, "other-model", [4], [Layer()])
    [old_file] = [p for p in tmp_path.glob("*.safetensors") if ps.read_metadata(p)["model"] == OLD]
    past = 1_700_000_000
    os.utime(old_file, (past, past))
    new = rekey.new_key(OLD, "4.1.2", "0123456789ab")
    out = rekey.move([tmp_path], OLD, new)
    assert out["files"] == 1 and out["others"] == 1
    assert not old_file.exists()
    target = tmp_path / (ps.snapshot_key(new, [1, 2, 3]) + ".safetensors")
    assert target.exists() and int(target.stat().st_mtime) == past
    [(tokens, cache)] = list(ps.load_snapshots(tmp_path, new))
    assert tokens == [1, 2, 3] and cache[0].offset == 7
    assert mx.array_equal(cache[0].state_a, mx.arange(12).reshape(3, 4).astype(mx.float16))
    assert cache[0].history.tolist() == [[11, 12]]
    assert list(ps.load_snapshots(tmp_path, OLD)) == []
    assert rekey.move([tmp_path], OLD, new)["files"] == 0          # twice: nothing left to move


def test_move_back(tmp_path):
    ps.save_snapshot(tmp_path, OLD, [1, 2], [Layer()])
    new = rekey.new_key(OLD, "4.1.2", "0123456789ab")
    rekey.move([tmp_path], OLD, new)
    assert rekey.move([tmp_path], new, OLD)["files"] == 1
    [(tokens, _)] = list(ps.load_snapshots(tmp_path, OLD))
    assert tokens == [1, 2]


def test_fingerprint_leaves_out_falcon_traces_and_events():
    base = {"PLI_HYBRIDE": "1", "PLI_POINTS_BITS": "4"}
    more = dict(base, PLI_FALCON_TOKENS="/x", PLI_FALCON="/y", PLI_TRACE="1", PLI_EVENEMENTS="/z", PLI_CODE_KEY="0",
                PATH="/bin")
    assert code_key.settings_fingerprint(base) == code_key.settings_fingerprint(more)
    assert code_key.settings_fingerprint(base) != code_key.settings_fingerprint(dict(base, PLI_HYBRIDE="0"))


def test_fingerprint_is_the_old_one_without_those_settings():
    import hashlib
    from tensorfold.pli import env_names
    env = {"PLI_HYBRIDE": "1", "PLI_DISK_REPLY_END": "1", "PLI_TOOL_GRAMMAR": "1"}
    old = env_names.pli_settings(env, exclude=code_key.SETTINGS_EXCLUDED)
    assert code_key.settings_fingerprint(env) == hashlib.sha256(repr(old).encode()).hexdigest()[:12]


def test_plan_takes_the_known_fingerprints(tmp_path):
    env = {"PLI_HYBRIDE": "1"}
    fp = code_key.settings_fingerprint(env)
    ps.save_snapshot(tmp_path, "base|prefill=p|tensorfold=4.1.2|pli=aaaaaaaaaaaa|points=q4", [1], [Layer()])
    ps.save_snapshot(tmp_path, "base|prefill=p|tensorfold=4.1.2|pli=bbbbbbbbbbbb|points=q4", [2], [Layer()])
    ps.save_snapshot(tmp_path, "base|prefill=p|tensorfold=4.1.2|pli=cccccccccccc|points=q4", [3], [Layer()])
    out = rekey.plan([tmp_path], "4.1.2", ["aaaaaaaaaaaa", "bbbbbbbbbbbb"], env)
    assert sorted(out["old"]) == ["base|prefill=p|tensorfold=4.1.2|pli=aaaaaaaaaaaa|points=q4",
                                  "base|prefill=p|tensorfold=4.1.2|pli=bbbbbbbbbbbb|points=q4"]
    assert out["new"] == "base|prefill=p|pli=" + fp + "|points=q4|code=" + code_key.digest()
    assert out["others"] == 1


def test_move_in_place_when_the_header_fits(tmp_path):
    ps.save_snapshot(tmp_path, OLD, [5, 6], [Layer()])
    [f] = list(tmp_path.glob("*.safetensors"))
    inode = f.stat().st_ino
    new = rekey.new_key(OLD, "4.1.2", "0123456789ab")             # one character longer: "saved" loses a decimal
    assert rekey.move([tmp_path], OLD, new)["files"] == 1
    target = tmp_path / (ps.snapshot_key(new, [5, 6]) + ".safetensors")
    assert target.stat().st_ino == inode                          # renamed, not copied
    [(tokens, cache)] = list(ps.load_snapshots(tmp_path, new))
    assert tokens == [5, 6] and cache[0].offset == 7


def test_move_with_a_longer_key_copies(tmp_path):
    ps.save_snapshot(tmp_path, OLD, [3, 4], [Layer()])
    longer = OLD + "|" + "x" * 4000                     # does not fit in the old header: the file is copied
    assert rekey.move([tmp_path], OLD, longer)["files"] == 1
    [(tokens, cache)] = list(ps.load_snapshots(tmp_path, longer))
    assert tokens == [3, 4] and mx.array_equal(cache[0].state_a, mx.arange(12).reshape(3, 4).astype(mx.float16))
    assert not list(tmp_path.glob("*.partial.safetensors"))


def test_plan_code_changes_only_the_digest(tmp_path):
    ps.save_snapshot(tmp_path, "base|pli=abc|points=q4|code=000000000000", [1], [Layer()])
    out = rekey.plan_code([tmp_path])
    assert out["old"] == ["base|pli=abc|points=q4|code=000000000000"]
    assert out["new"] == "base|pli=abc|points=q4|code=" + code_key.digest()


def test_copy_log_is_out_of_the_fingerprint():
    base = {"PLI_HYBRIDE": "1"}
    assert code_key.settings_fingerprint(base) == code_key.settings_fingerprint(dict(base, PLI_COPY_LOG="/x.jsonl"))
