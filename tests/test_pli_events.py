"""PLI 3.6.0: event log (pli/events.py)."""

import importlib
import json
import os
import tempfile
import types
import unittest


def load(path: str, max_mb: str | None = None):
    os.environ["PLI_EVENTS"] = path
    if max_mb is None:
        os.environ.pop("PLI_EVENTS_MAX_MB", None)
    else:
        os.environ["PLI_EVENTS_MAX_MB"] = max_mb
    from tensorfold.pli import events
    return importlib.reload(events)


class EventLog(unittest.TestCase):
    def tearDown(self) -> None:
        os.environ.pop("PLI_EVENTS", None)
        os.environ.pop("PLI_EVENTS_MAX_MB", None)
        from tensorfold.pli import events
        importlib.reload(events)

    def test_off_writes_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            ev = load("")
            self.assertFalse(ev.active())
            ev.ev("done", conv="x")
            self.assertEqual(os.listdir(d), [])

    def test_same_conversation_over_several_turns(self) -> None:
        ev = load("/dev/null")
        turn1 = [{"role": "system", "content": "tool"}, {"role": "user", "content": "sort the list"}]
        turn2 = turn1 + [{"role": "assistant", "content": "here"}, {"role": "user", "content": "and in PHP?"}]
        other = [{"role": "system", "content": "tool"}, {"role": "user", "content": "other topic"}]
        self.assertEqual(ev.conversation(turn1), ev.conversation(turn2))
        self.assertNotEqual(ev.conversation(turn1), ev.conversation(other))
        self.assertEqual(len(ev.conversation(turn1)), 10)
        # content in parts (Anthropic format or images): same fingerprint as the joined text
        parts = [{"role": "system", "content": "tool"}, {"role": "user", "content": [{"type": "text", "text": "sort "}, {"type": "text", "text": "the list"}]}]
        self.assertEqual(ev.conversation(turn1), ev.conversation(parts))

    def test_line_written_and_attached_by_tokens(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "ev.jsonl")
            ev = load(path)
            a = types.SimpleNamespace(job_id="req-a", prompt_ids=list(range(100)))
            b = types.SimpleNamespace(job_id="req-b", prompt_ids=list(range(50)) + [7] * 50)
            ev.bind(a, "conva")
            ev.bind(b, "convb")
            self.assertEqual(a.pli_conv, "conva")
            self.assertEqual(ev.conv_of_tokens(list(range(90))), "conva")
            self.assertEqual(ev.conv_of_tokens(list(range(100)) + [1, 2, 3]), "conva")   # prompt + reply
            self.assertEqual(ev.conv_of_tokens(list(range(40))), "shared")               # common start
            self.assertEqual(ev.conv_of_tokens([9, 9, 9]), "")
            ev.ev("done", a, output=12, tok_s=41.234567, wait=None)
            ev.ev("spill", conv=ev.conv_of_tokens(list(range(90))), tokens=90)
            lines = [json.loads(line) for line in open(path, encoding="utf-8")]
            self.assertEqual(lines[0]["a"], "done")
            self.assertEqual(lines[0]["req"], "req-a")
            self.assertEqual(lines[0]["conv"], "conva")
            self.assertEqual(lines[0]["tok_s"], 41.235)
            self.assertNotIn("wait", lines[0])
            self.assertEqual(lines[1]["conv"], "conva")
            self.assertNotIn("req", lines[1])

    def test_rotation_and_swallowed_error(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "ev.jsonl")
            ev = load(path, "1")
            with open(path, "w") as f:
                f.write("x" * (1024 * 1024 + 10))
            ev.ev("done", conv="c")
            self.assertTrue(os.path.exists(path + ".1"))
            self.assertEqual(len(open(path).read().splitlines()), 1)
            ev = load(os.path.join(d, "missing", "ev.jsonl"))
            ev.ev("done", conv="c")          # missing folder: nothing comes out


if __name__ == "__main__":
    unittest.main()
