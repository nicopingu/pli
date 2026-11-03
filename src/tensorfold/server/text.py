"""Prompt rendering and streamed reply text with think blocks, tool markup and locked tokenizer access."""

from __future__ import annotations

import os
import re
from typing import Any

from tensorfold.server.messages import _normalize_tool_call_arguments, late_system_role, normalize_messages

_THINK_END = "</think>"
# (what a reply writes to open its think block, what closes it): Qwen's prompt opens the block; Gemma 4's reply does
THINK_MARKERS = ("", _THINK_END)
CHANNEL_MARKERS = ("<|channel>thought", "<channel|>")
# (opener, closer) of a tool call's markup: Qwen's, Gemma 4's, DeepSeek-V4's DSML block
_CALLS = (("<tool_call>", "</tool_call>"), ("<|tool_call>", "<tool_call|>"),
          ("<｜DSML｜tool_calls>", "</｜DSML｜tool_calls>"))


def _partial_tag(text: str, tag: str) -> int:
    """How many characters at the end of ``text`` could be the start of ``tag``."""

    for k in range(min(len(tag) - 1, len(text)), 0, -1):
        if text.endswith(tag[:k]):
            return k
    return 0


def split_thinking(text: str, *, finished: bool, markers: tuple[str, str] = THINK_MARKERS) -> tuple[str, str]:
    """(reasoning, answer) of a thinking reply; while it streams, a tail that could begin a marker is held back."""

    # Gemma 4 can open its thought channel after visible text. Strip that block wherever it opens.
    opener, closer = markers
    if opener and not text.startswith(opener):
        start = text.find(opener)
        if start > 0:                                        # a thought channel opened after some visible text
            prefix = text[:start]
            prefix = prefix[: len(prefix) - _partial_tag(prefix, opener)]   # drop a stray/doubled partial opener
            reasoning, answer = split_thinking(text[start:], finished=finished, markers=markers)
            return reasoning, prefix + answer
        if not finished:                                     # no opener yet: hold a tail that could begin one
            return "", text[: len(text) - max(_partial_tag(text, tag)
                                               for tag in (opener, closer, *(o for o, _ in _CALLS)))]
        return "", text                                      # a finished reply that never opened the block
    if opener:
        text = text[len(opener):].lstrip("\n")
    end = text.find(closer)
    if end >= 0:
        return text[:end], text[end + len(closer):].lstrip("\n")
    call = min((at for at in (text.find(opener) for opener, _ in _CALLS) if at >= 0), default=-1)
    if call >= 0:
        # a call written before the block closes: held while the block may still close, the answer if the reply ends
        if not finished:
            return text[:call], ""
        if any(text.startswith(opener, call) and close in text[call:] for opener, close in _CALLS):
            return text[:call], text[call:]
        return text, ""
    held = 0 if finished else max(_partial_tag(text, tag) for tag in (closer, *(opener for opener, _ in _CALLS)))
    return text[: len(text) - held], ""


def reasoning_count(tokens: list[int], think_end: int | None) -> int:
    """A thinking reply's reasoning tokens: through its close ``think_end`` (None or -1: not thinking), else all."""

    if think_end is None or think_end < 0:
        return 0
    return tokens.index(think_end) + 1 if think_end in tokens else len(tokens)


def think_markers(tokenizer: Any) -> tuple[str, str]:
    """Gemma 4's thought channel when the tokenizer has its ``<channel|>`` token, else Qwen's ``</think>``."""

    try:
        close = tokenizer.convert_tokens_to_ids(CHANNEL_MARKERS[1])
    except Exception:  # noqa: BLE001 - a tokenizer without the lookup: the default markers
        return THINK_MARKERS
    unk = getattr(tokenizer, "unk_token_id", None)
    return CHANNEL_MARKERS if isinstance(close, int) and close >= 0 and close != unk else THINK_MARKERS


def hide_tool_calls(text: str, *, finished: bool) -> str:
    """Hide tool-call blocks and partial opening tags while streaming so visible text only grows and calls arrive as deltas."""

    out: list[str] = []
    pos = 0
    while True:
        found = [(text.find(opener, pos), opener, closer) for opener, closer in _CALLS]
        found = [f for f in found if f[0] >= 0]
        if not found:
            tail = text[pos:]
            held = 0 if finished else max(_partial_tag(tail, opener) for opener, _ in _CALLS)
            out.append(tail[: len(tail) - held])
            return "".join(out)
        start, opener, closer = min(found)
        out.append(text[pos:start])
        end = text.find(closer, start + len(opener))
        if end < 0:
            return "".join(out)
        pos = end + len(closer)


# PLI (23/10/2026, PLI_KEEP_REPLY=1, V3): without thinking, the reply prompt writes "<think>\n\n</think>\n\n" but the history
# gives back the reply without this block: the reply moves and the next turn reads it again (78,000 tokens, 735 s on the 21-22/10 trace).
# The modified template gives back this empty block before each reply that follows the last question (as it keeps a non-empty thinking), and the
# engine keeps the state written at the end of the reply (lane_engine._keeps_decoded): the next turn resumes after the reply.
KEEP_REPLY = __import__("os").environ.get("PLI_KEEP_REPLY", "0") == "1"
_TEMPLATES: dict = {}
_CONDITION = "{%- if reasoning_content and (preserve_thinking is undefined"


def _kept_template(tokenizer):
    """The model's template with the empty block kept, or None if it does not have the known shape (Qwen3.6 to 3.8)."""

    text = getattr(tokenizer, "chat_template", None)
    if not isinstance(text, str) or text.count(_CONDITION) != 1:
        return None
    if text not in _TEMPLATES:
        _TEMPLATES[text] = text.replace(_CONDITION, "{%- if (reasoning_content or (pli_empty is defined and pli_empty and "
                                        "loop.index0 > ns.last_query_index)) and (preserve_thinking is undefined")
    return _TEMPLATES[text]


# PLI (behaviour of TensorFold 1.1.2, issue #1117, rewritten in Python): a control token written in the text of a request
# (messages, tools, tool results, reasoning) stays ordinary text; without this "<|im_end|><|im_start|>system" in a
# message or in a page read by an agent opens a real turn. Only the added tokens marked special are concerned: <think>, </think>,
# <tool_call>, <tool_response> (not special, read inside the content by Qwen's template) stay as they are. A prompt with no control
# token written keeps exactly its tokens; a prompt that writes one counts more tokens. PLI_CONTROL_AS_TEXT=0: the former behaviour.
_CONTROL = os.environ.get("PLI_CONTROL_AS_TEXT", "1") != "0"
_SENT_A, _SENT_B = "\ue000", "\ue001"
_PATTERNS: dict[int, Any] = {}


def _control_pattern(tokenizer: Any) -> Any:
    """Pattern of the tokenizer's control tokens (added and special), plus the sentinel character; None without a readable list."""

    key = id(tokenizer)
    if key not in _PATTERNS:
        raw = getattr(tokenizer, "_tokenizer", tokenizer)
        decoder = getattr(raw, "added_tokens_decoder", None)
        names = []
        if isinstance(decoder, dict):
            names = [str(getattr(t, "content", t)) for t in decoder.values() if getattr(t, "special", False)]
        names = sorted({n for n in names if n}, key=len, reverse=True)
        _PATTERNS[key] = re.compile("|".join(re.escape(n) for n in names + [_SENT_A])) if names else None
    return _PATTERNS[key]


def _mask(obj: Any, pattern: Any, originals: list[str]) -> Any:
    """Each control token written in a string replaced by a numbered sentinel (dictionary keys unchanged)."""

    if isinstance(obj, str):
        def sentinel(m: Any) -> str:
            originals.append(m.group(0))
            return f"{_SENT_A}{len(originals) - 1}{_SENT_B}"
        return pattern.sub(sentinel, obj)
    if isinstance(obj, dict):
        return {k: _mask(v, pattern, originals) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_mask(v, pattern, originals) for v in obj)
    return obj


def _encode_masked(tokenizer: Any, rendered: str, originals: list[str]) -> list[int]:
    """The rendered prompt as tokens: the template's text with its control tokens, each sentinel as ordinary text."""

    raw = getattr(tokenizer, "_tokenizer", tokenizer)
    ids: list[int] = []
    pieces = re.split(f"{_SENT_A}(\\d+){_SENT_B}", rendered)
    for i, piece in enumerate(pieces):
        if i % 2 == 0:
            if piece:
                ids += [int(t) for t in raw.encode(piece, add_special_tokens=False)]
        else:
            ids += [int(t) for t in raw.encode(originals[int(piece)], add_special_tokens=False, split_special_tokens=True)]
    return ids


def render_prompt_ids(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]] | None = None,
    enable_thinking: bool = False,
    add_generation_prompt: bool = True,
    reasoning_effort: str | None = None,
    late_system: str | None = None,
) -> list[int]:
    """Render chat tokens; Qwen3.8 thinking effort ``medium`` preserves the system block, while default ``xhigh`` adds an instruction."""

    if late_system is None:
        late_system = template_late_system(tokenizer)
    messages = _normalize_tool_call_arguments(normalize_messages(messages, late_system=late_system))
    kwargs: dict[str, Any] = {
        "add_generation_prompt": add_generation_prompt,
        "enable_thinking": enable_thinking,
        "thinking_mode": "thinking" if enable_thinking else "chat",   # DeepSeek-V4's templates read this switch
    }
    if enable_thinking and reasoning_effort:
        kwargs["reasoning_effort"] = reasoning_effort
    if tools:
        kwargs["tools"] = tools
    if KEEP_REPLY and not enable_thinking:
        template = _kept_template(tokenizer)
        if template is not None:
            kwargs["chat_template"] = template
            kwargs["pli_empty"] = True
    pattern = _control_pattern(tokenizer) if _CONTROL else None
    originals: list[str] = []
    if pattern is not None:
        masked = _mask(messages, pattern, originals)
        masked_tools = _mask(kwargs["tools"], pattern, originals) if "tools" in kwargs else None
        if originals:
            messages = masked
            if masked_tools is not None:
                kwargs["tools"] = masked_tools
    if originals:
        # a control token written in the request: rendered as text, then tokens piece by piece
        kwargs["tokenize"] = False
        try:
            rendered = tokenizer.apply_chat_template(messages, **kwargs)
        except TypeError:
            kwargs.pop("tools", None)
            rendered = tokenizer.apply_chat_template(messages, **kwargs)
        ids = _encode_masked(tokenizer, str(rendered), originals)
    else:
        try:
            rendered = tokenizer.apply_chat_template(messages, **kwargs)
        except TypeError:
            kwargs.pop("tools", None)
            rendered = tokenizer.apply_chat_template(messages, **kwargs)
        if isinstance(rendered, str):
            rendered = tokenizer.encode(rendered)
        ids = [int(t) for t in rendered]
    if not enable_thinking and add_generation_prompt:
        ids = _close_open_think(tokenizer, ids)
    return ids


def _close_open_think(tokenizer: Any, ids: list[int]) -> list[int]:
    """Close a trailing bare ``<think>`` when thinking is off: GLM-5.3's template writes one whatever the switch."""

    if not ids:
        return ids
    try:
        if tokenizer.decode([ids[-1]]).strip() != "<think>":
            return ids
        close = tokenizer.encode("</think>", add_special_tokens=False)
    except Exception:  # noqa: BLE001 - a tokenizer without these tokens keeps the prompt as rendered
        return ids
    return ids + [int(t) for t in close] if len(close) == 1 else ids


def template_late_system(tokenizer: Any) -> str:
    """The role a later system message renders as in this tokenizer's chat template (see ``late_system_role``)."""

    return late_system_role(lambda messages: tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False))


def eos_ids_of(tokenizer: Any) -> frozenset[int]:
    values = getattr(tokenizer, "eos_token_ids", None)
    if values:
        return frozenset(int(t) for t in values)
    single = getattr(tokenizer, "eos_token_id", None)
    return frozenset({int(single)}) if single is not None else frozenset()


def is_title_request(messages: list[dict[str, Any]], tools: Any) -> bool:
    """Identify short session-title requests without tools so they can yield to foreground turns."""

    if tools or not messages or messages[0].get("role") != "system":
        return False
    size = sum(len(m["content"]) if isinstance(m.get("content"), str) else 4096 for m in messages)
    text = messages[0].get("content")
    return size < 4096 and isinstance(text, str) and "title" in text.lower()



class IncrementalText:
    """Decode only new tokens with the previous chunk as context so multi-byte characters split across tokens remain whole."""

    def __init__(self, tokenizer: Any, lock: Any) -> None:
        self.tokenizer = tokenizer
        self.lock = lock
        self.tokens: list[int] = []
        self.text = ""
        self._prefix = 0
        self._read = 0

    def extend(self, tokens: list[int]) -> str:
        self.tokens.extend(int(t) for t in tokens)
        with self.lock:
            before = self.tokenizer.decode(self.tokens[self._prefix:self._read])
            after = self.tokenizer.decode(self.tokens[self._prefix:])
        if len(after) > len(before) and not after.endswith("\ufffd"):
            self.text += after[len(before):]
            self._prefix, self._read = self._read, len(self.tokens)
        return self.text


class _LockedTokenizer:
    """The app's tokenizer behind its lock, for proposers on the scheduler thread."""

    def __init__(self, tokenizer: Any, lock: Any) -> None:
        self._tokenizer = tokenizer
        self._lock = lock

    def decode(self, ids: list[int]) -> str:
        with self._lock:
            return self._tokenizer.decode(ids)

    def encode(self, text: str, **kwargs: Any) -> list[int]:
        with self._lock:
            return self._tokenizer.encode(text, **kwargs)

    def convert_tokens_to_ids(self, token: str) -> Any:
        with self._lock:
            return self._tokenizer.convert_tokens_to_ids(token)


def strip_trailing_stops(tokens: list[int], stop_ids: set[int]) -> list[int]:
    end = len(tokens)
    while end > 0 and tokens[end - 1] in stop_ids:
        end -= 1
    return tokens[:end]


HARMONY_FINAL_MARKER = "<|channel|>final<|message|>"
HARMONY_ANALYSIS_MARKER = "<|channel|>analysis<|message|>"
HARMONY_TERMINATORS = ("<|return|>", "<|end|>", "<|call|>", "<|start|>")


def parse_harmony_output(text: str) -> tuple[str, str | None]:
    """Split Harmony content and reasoning, pass other text unchanged, and return empty content when the final channel is absent."""

    if "<|channel|>" not in text:
        return text, None

    reasoning: str | None = None
    if HARMONY_ANALYSIS_MARKER in text:
        reasoning = text.split(HARMONY_ANALYSIS_MARKER, 1)[1]
        for terminator in (HARMONY_FINAL_MARKER, *HARMONY_TERMINATORS):
            reasoning = reasoning.split(terminator, 1)[0]

    if HARMONY_FINAL_MARKER not in text:
        return "", reasoning
    content = text.split(HARMONY_FINAL_MARKER, 1)[1]
    for terminator in HARMONY_TERMINATORS:
        content = content.split(terminator, 1)[0]
    return content, reasoning


def streaming_visible_text(text: str) -> str:
    """The part of partially-decoded output that should stream to the client."""

    if "<|channel|>" not in text:
        return text
    content, _ = parse_harmony_output(text)
    return content
