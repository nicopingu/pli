"""A tool result's images through the vision checkpoints' own chat templates (no GPU, no weights).

Agents return screenshots as tool results, and every later turn of the session sends them again. Each vision family's
template renders a tool message's image parts in place, inside its tool response, so the server passes them through as
it does a user message's. The prompt then holds exactly one image marker per image, the count the image frontends
check. The test needs each checkpoint's template files in the Hugging Face cache; repos that are absent are skipped:

    hf download Vontra/GLM-5.3-Flash-MLX-4bit-MTP tokenizer_config.json chat_template.jinja
"""

from __future__ import annotations

import base64
import io

import pytest

pytest.importorskip("jinja2")
Image = pytest.importorskip("PIL.Image")


FILES = ("tokenizer_config.json", "chat_template.jinja")
CHECKPOINTS = [                    # each repo and its image frontend's marker
    ("Vontra/GLM-5.3-Flash-MLX-4bit-MTP", "<|image|>"),
    ("Vontra/Qwen3.8-27B-MLX-4bit", "<|image_pad|>"),
    ("Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP", "<|image_pad|>"),
]


def _checkpoint(repo):
    from tensorfold import hub

    try:
        found = hub.cached(repo)
    except ImportError:
        return None
    return found if found is not None and all((found / name).is_file() for name in FILES) else None


def _url(color):
    output = io.BytesIO()
    Image.new("RGB", (2, 2), color).save(output, format="PNG")
    url = "data:image/png;base64," + base64.b64encode(output.getvalue()).decode()
    return {"type": "image_url", "image_url": {"url": url}}


def _session():
    call = {"id": "call_1", "type": "function", "function": {"name": "screenshot", "arguments": "{}"}}
    return [{"role": "user", "content": [{"type": "text", "text": "Compare the page with this mockup: "}, _url("red")]},
            {"role": "assistant", "content": "", "tool_calls": [call]},
            {"role": "tool", "tool_call_id": "call_1", "content": [{"type": "text", "text": "PAGE"}, _url("blue")]},
            {"role": "user", "content": "Does it match?"}]


TOOLS = [{"type": "function", "function": {"name": "screenshot", "description": "Capture the page.",
                                           "parameters": {"type": "object", "properties": {}}}}]

