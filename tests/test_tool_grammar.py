"""PLI (10/11/2026): a request that offers tools gets its model's tool-call grammar (PLI_TOOL_GRAMMAR=1)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

xgr = pytest.importorskip("xgrammar")

from tensorfold.engine import grammar as g   # noqa: E402

MODEL = Path(os.environ.get("PLI_TEST_QWEN38_MODEL", "~/models/Qwen3.8-27B-MLX-4bit")).expanduser()
if not (MODEL / "tokenizer.json").is_file():
    pytest.skip("Qwen3.8 tokenizer absent (PLI_TEST_QWEN38_MODEL)", allow_module_level=True)

BRIDGE = {"type": "function", "function": {"name": "tool_call", "description": "call deferred tools", "parameters": {
    "type": "object", "properties": {"calls": {"type": "array", "items": {"type": "object", "properties": {
        "name": {"type": "string"}, "arguments": {"type": "object"}}, "required": ["name", "arguments"]}}},
    "required": ["calls"]}}}
TERMINAL = {"type": "function", "function": {"name": "terminal", "description": "run", "parameters": {
    "type": "object", "properties": {"command": {"type": "string"}, "timeout": {"type": "integer"}},
    "required": ["command"]}}}
CALL = ('<tool_call>\n<function=tool_call>\n<parameter=calls>\n'
        '[{"name": "vault_read", "arguments": {"path": "a.md"}}]\n</parameter>\n</function>\n</tool_call>')


class Owner:
    def __init__(self, tokenizer) -> None:
        self.model_dir = str(MODEL)
        self.stop_ids = tokenizer.convert_tokens_to_ids(["<|im_end|>"])


@pytest.fixture(scope="module")
def tok():
    from transformers import PreTrainedTokenizerFast
    return PreTrainedTokenizerFast(tokenizer_file=str(MODEL / "tokenizer.json"))


@pytest.fixture()
def on(monkeypatch):
    monkeypatch.setattr(g, "TOOL_GRAMMAR", True)


def _walk(constraint, tok, text: str, stop: bool = True) -> str:
    ids = tok.encode(text, add_special_tokens=False) + (list(constraint_stop(tok)) if stop else [])
    bits = np.full((1, constraint.words), -1, dtype=np.int32)
    for i, t in enumerate(ids):
        constraint.m.fill_next_token_bitmask(bits, 0)
        if not constraint.m.accept_token(t):
            return f"rejected at {i}"
    return "finished" if constraint.m.is_terminated() else "open"


def constraint_stop(tok):
    return tok.convert_tokens_to_ids(["<|im_end|>"])


def test_off_by_default_gives_nothing(tok):
    owner = Owner(tok)
    if not g.TOOL_GRAMMAR:
        assert g.request_constraint(owner, {}, None, tools=[BRIDGE]) is None


def test_format_from_config():
    assert g.tool_format(MODEL) == "qwen_3_5"
    assert g.tool_format(None) is None


def test_valid_calls_and_text_pass(tok, on):
    owner = Owner(tok)
    for text in ("Hello, here is the answer.", "Reading.\n\n" + CALL, CALL + "\n" + CALL,
                 '<tool_call>\n<function=terminal>\n<parameter=command>\nls -la\n</parameter>\n</function>\n</tool_call>'):
        c = g.request_constraint(owner, {}, None, tools=[BRIDGE, TERMINAL])
        assert _walk(c, tok, text) == "finished", text


@pytest.mark.parametrize("bad", [
    CALL.replace('"name": "vault_read", ', ""),                                         # no tool name
    CALL.replace('{"name": "vault_read", "arguments": {"path": "a.md"}}',
                 '{"arguments": {"path": "a.md"}}'),                                     # arguments first, no name
    CALL.replace('"a.md"}}', '"a.md"}'),                                                 # broken JSON
    CALL.replace("function=tool_call", "function=unknown"),                             # a tool not offered
])
def test_bad_calls_are_impossible(tok, on, bad):
    c = g.request_constraint(Owner(tok), {}, None, tools=[BRIDGE, TERMINAL])
    assert _walk(c, tok, bad).startswith("rejected")


def test_bridge_items_take_either_order(tok, on):
    spec = g.tools_spec(Owner(tok), [BRIDGE])
    items = json.loads(spec.text)["tools"][0]["function"]["parameters"]["properties"]["calls"]["items"]
    assert [list(v["properties"]) for v in items["anyOf"]] == [["name", "arguments"], ["arguments", "name"]]
    c = g.request_constraint(Owner(tok), {}, None, tools=[BRIDGE])
    swapped = CALL.replace('{"name": "vault_read", "arguments": {"path": "a.md"}}',
                           '{"arguments": {"path": "a.md"}, "name": "vault_read"}')
    assert _walk(c, tok, swapped) == "finished"


def test_other_schemas_keep_their_order(tok, on):
    spec = g.tools_spec(Owner(tok), [TERMINAL])
    assert list(json.loads(spec.text)["tools"][0]["function"]["parameters"]["properties"]) == ["command", "timeout"]


def test_response_format_wins_over_tools(tok, on):
    c = g.request_constraint(Owner(tok), {"response_format": {"type": "json_object"}}, None, tools=[BRIDGE])
    assert c.spec.kind == "json"


def test_a_tools_grammar_that_does_not_compile_leaves_the_request_free(tok, on, monkeypatch):
    owner = Owner(tok)
    grammars = g.compiler(owner, owner.model_dir, owner.stop_ids)

    def boom(text):
        raise ValueError("unsupported schema")

    monkeypatch.setattr(grammars, "_compile_tools", boom)
    weird = {"type": "function", "function": {"name": "odd", "parameters": {"type": "object"}}}
    assert g.request_constraint(owner, {}, None, tools=[weird]) is None
    assert g.request_constraint(owner, {}, None, tools=[weird]) is None        # remembered, not compiled again


def test_pack_and_follow_keep_the_tools_grammar(tok, on):
    owner = Owner(tok)
    c = g.request_constraint(owner, {}, None, tools=[BRIDGE])
    grammars = g.compiler(owner, owner.model_dir, owner.stop_ids)
    again = grammars.follow(g.pack(c))
    assert again.spec.kind == "tools" and _walk(again, tok, CALL) == "finished"
