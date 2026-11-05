"""PLI: the "checkpoint" field of /health and /v1/models says which weights, draft and MTP head are served."""

import http.client
import json
import threading
from contextlib import contextmanager
from http.server import ThreadingHTTPServer
from types import SimpleNamespace

from tensorfold.server.checkpoint_info import describe, describe_weights, hub_snapshot
from tensorfold.server.http import make_handler


def weights(directory, config):
    directory.mkdir(parents=True)
    (directory / "config.json").write_text(json.dumps(config))
    return directory


def test_hub_snapshot_reads_repo_and_revision(tmp_path):
    snap = tmp_path / "hub" / "models--z-lab--Qwen3.8-27B-DFlash2" / "snapshots" / "50307d4c"
    assert hub_snapshot(snap) == ("z-lab/Qwen3.8-27B-DFlash2", "50307d4c")
    assert hub_snapshot(snap / "model.safetensors") == ("z-lab/Qwen3.8-27B-DFlash2", "50307d4c")
    assert hub_snapshot(tmp_path / "models" / "My-Model-4bit") is None


def test_local_folder_reads_type_and_quantization(tmp_path):
    folder = weights(tmp_path / "My-Model-4bit", {
        "model_type": "qwen3_5", "text_config": {"model_type": "qwen3_5_text"},
        "quantization": {"group_size": 64, "bits": 4, "mode": "affine"}})
    info = describe_weights(folder)
    assert info == {"name": "My-Model-4bit", "path": str(folder), "repo": None, "revision": None,
                    "model_type": "qwen3_5", "text_model_type": "qwen3_5_text",
                    "quantization": {"bits": 4, "group_size": 64, "mode": "affine"}}


def test_unreadable_config_gives_nulls_not_an_error(tmp_path):
    folder = tmp_path / "No-Config"
    folder.mkdir()
    (folder / "config.json").write_text("{not json")
    info = describe_weights(folder)
    assert info["name"] == "No-Config" and info["model_type"] is None and info["quantization"] is None
    assert describe_weights(tmp_path / "absent")["model_type"] is None


def test_hybrid_reports_draft_and_mtp_head(tmp_path):
    model_dir = weights(tmp_path / "My-Model-4bit", {"model_type": "qwen3_5", "quantization": {"bits": 4, "group_size": 64}})
    draft = weights(tmp_path / "hub" / "models--z-lab--Draft" / "snapshots" / "abc123", {"model_type": "qwen3"})
    head = tmp_path / "heads" / "v12" / "mtp-4bit.safetensors"
    model = SimpleNamespace(head_drafts=object(), mtp=object(), pli_mtp_head={"path": str(head), "hybrid": True})
    info = describe(model_dir, drafter=str(draft), model=model, version="9.9.9")
    assert info["name"] == "My-Model-4bit"
    assert info["draft"]["name"] == "z-lab/Draft" and info["draft"]["revision"] == "abc123"
    assert info["draft"]["model_type"] == "qwen3"
    assert info["mtp_head"] == {"name": "v12/mtp-4bit.safetensors", "path": str(head), "hybrid": True,
                                "active": True, "in_checkpoint": False}
    assert info["engine"] == {"name": "PLI", "version": "9.9.9", "base": "TensorFold 0.6.6"}


def test_mtp_head_alone_replaces_the_draft(tmp_path):
    model_dir = weights(tmp_path / "M", {"model_type": "qwen3_5"})
    draft = weights(tmp_path / "D", {"model_type": "qwen3"})
    model = SimpleNamespace(head_drafts=None, mtp=object(), pli_mtp_head={"path": "/x/t.safetensors", "hybrid": False})
    info = describe(model_dir, drafter=str(draft), model=model)
    assert info["draft"] is None
    assert info["mtp_head"]["hybrid"] is False and info["mtp_head"]["active"] is True


def test_draft_only_and_checkpoint_head(tmp_path):
    model_dir = weights(tmp_path / "M", {"model_type": "qwen3_5"})
    draft = weights(tmp_path / "D", {"model_type": "qwen3"})
    dflash = SimpleNamespace(head_drafts=object(), mtp=object())       # Qwen35Family: "mtp" is the draft
    info = describe(model_dir, drafter=str(draft), model=dflash)
    assert info["draft"]["name"] == "D" and info["mtp_head"] is None
    moe = SimpleNamespace(head_drafts=None, mtp=object())                # Qwen3.6: the checkpoint's MTP layer
    info = describe(model_dir, model=moe)
    assert info["draft"] is None and info["mtp_head"]["in_checkpoint"] is True
    assert describe(model_dir, model=SimpleNamespace(head_drafts=None, mtp=None))["mtp_head"] is None
    from tensorfold import __version__
    assert describe(None)["engine"]["version"] == __version__ and describe(None)["name"] is None


class Memory:
    def memory_snapshot(self, reset_peak):
        return {"active": 1}


@contextmanager
def serving(app):
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(app))
    worker = threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    worker.start()
    try:
        yield httpd.server_port
    finally:
        httpd.shutdown()
        httpd.server_close()
        worker.join()


def get(port, path):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=3)
    try:
        connection.request("GET", path)
        response = connection.getresponse()
        assert response.status == 200
        return json.loads(response.read())
    finally:
        connection.close()


def test_health_and_models_carry_checkpoint_and_keep_their_fields(tmp_path):
    card = describe(weights(tmp_path / "My-Model-4bit", {"model_type": "qwen3_5"}), version="1.0")
    app = SimpleNamespace(served_name="qwen-served", model_ids=["qwen-served", "alias"], max_batch_size=2,
                          prompt_memory=Memory(), checkpoint=card)
    with serving(app) as port:
        health = get(port, "/health")
        models = get(port, "/v1/models")
    assert health["model"] == "qwen-served" and health["model_ids"] == ["qwen-served", "alias"]
    assert health["status"] == "ok" and health["max_batch_size"] == 2 and health["memory"] == {"active": 1}
    assert health["checkpoint"]["name"] == "My-Model-4bit"
    assert [m["id"] for m in models["data"]] == ["qwen-served", "alias"]
    assert all(m["object"] == "model" and m["owned_by"] == "tensorfold" for m in models["data"])
    assert models["data"][0]["checkpoint"]["name"] == "My-Model-4bit"


def test_app_without_checkpoint_gives_null(tmp_path):
    app = SimpleNamespace(served_name="t", model_ids=["t"], max_batch_size=1, prompt_memory=Memory())
    with serving(app) as port:
        assert get(port, "/health")["checkpoint"] is None
        assert "checkpoint" not in get(port, "/v1/models")["data"][0]
