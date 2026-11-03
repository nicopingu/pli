"""PLI (29/10/2026): a system message in mid-conversation (Claude Code sends some) goes through the Anthropic route;
it is given the role the chat template accepts (late_system), as on the OpenAI route."""

import inspect

from tensorfold.server import anthropic
from tensorfold.server.anthropic_translate import translate
from tensorfold.server.messages import normalize_messages


def _body():
    return {"model": "m", "max_tokens": 16, "system": "You are an agent.",
            "messages": [{"role": "user", "content": "Hello"},
                         {"role": "system", "content": [{"type": "text", "text": "Reminder: stay in the folder."}]},
                         {"role": "assistant", "content": "Understood."},
                         {"role": "user", "content": "Next"}]}


def test_mid_conversation_system_message_is_translated():
    chat = translate(_body())
    assert [m["role"] for m in chat["messages"]] == ["system", "user", "system", "assistant", "user"]


def test_mid_conversation_system_message_given_as_user_when_the_template_does_not_keep_it():
    out = normalize_messages(translate(_body())["messages"], late_system="user")
    assert [m["role"] for m in out] == ["system", "user", "user", "assistant", "user"]
    assert "Reminder" in str(out[2]["content"])


def test_the_route_no_longer_refuses():
    assert "does not support mid-conversation system messages" not in inspect.getsource(anthropic.post)
