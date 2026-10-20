"""Fake tokenizer and special tokens shared by the HTTP route tests (taken from the former CUDA tool-choice test)."""

import re

EOS, OPEN, CLOSE, THINK, END = 0, 1000, 1001, 1002, 1003
SPECIAL = {"<tool_call>": OPEN, "</tool_call>": CLOSE, "<think>": THINK, "</think>": END}
CALL = "\n<function=get_weather>\n<parameter=city>\nOslo\n</parameter>\n</function>\n"
TOOLS = [{"type": "function", "function": {"name": name, "parameters": {"type": "object", "properties": {}}}}
         for name in ("get_weather", "search")]


class Tokens:
    """Characters as their code points, the markup as single tokens."""

    def encode(self, text, **kwargs):
        ids, parts = [], re.split("(" + "|".join(re.escape(s) for s in SPECIAL) + ")", text)
        for part in parts:
            ids += [SPECIAL[part]] if part in SPECIAL else [ord(c) for c in part]
        return type("Encoding", (), {"ids": ids})()

    def decode(self, ids, **kwargs):
        names = {v: k for k, v in SPECIAL.items()}
        return "".join(names.get(i, "" if i == EOS else chr(i)) for i in ids)

    def token_to_id(self, text):
        return SPECIAL.get(text)
