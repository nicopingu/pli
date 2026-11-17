"""PLI (16/11/2026): background tasks (pli/tasks/__init__.py, tasks_prune.py, tasks_drafter.py), without a model."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from pli import tasks as T
from pli.tasks import drafter as D
from pli.tasks import prune as P


class Clock:
    def __init__(self, t: float = 1000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


class Counting(T.Task):
    name = "count"

    def __init__(self, tasks, context, *, need=0, fail=False):
        super().__init__(tasks, context)
        self.need, self.fail, self.ran = need, fail, 0

    def due(self, now):
        return True

    def need_bytes(self):
        return self.need

    def step(self, engine):
        if self.fail:
            raise RuntimeError("boom")
        self.ran += 1


def make(tmp_path, **kw):
    clock = Clock()
    tasks = T.Tasks(calm_s=30, log_path=str(tmp_path / "log.jsonl"), state_dir=str(tmp_path / "state"), clock=clock,
                    **kw)
    return tasks, clock


def test_tick_waits_for_calm(tmp_path):
    tasks, clock = make(tmp_path)
    task = Counting(tasks, {})
    tasks.tasks.append(task)
    assert not tasks.tick(None, last_activity=clock.t - 10)
    assert task.ran == 0
    assert tasks.tick(None, last_activity=clock.t - 31)
    assert task.ran == 1


def test_tick_waits_while_busy_and_for_memory(tmp_path):
    busy = {"v": True}
    tasks, clock = make(tmp_path, busy=lambda: busy["v"], memory_ok=lambda need, frac: need < 100)
    big = Counting(tasks, {}, need=1000)
    tasks.tasks.append(big)
    assert not tasks.tick(None, clock.t - 60)            # a generation child works
    busy["v"] = False
    assert not tasks.tick(None, clock.t - 60)            # does not fit
    assert tasks.waits_memory == 1
    small = Counting(tasks, {}, need=10)
    tasks.tasks.append(small)
    assert tasks.tick(None, clock.t - 60)
    assert small.ran == 1 and big.ran == 0


def test_errors_switch_a_task_off(tmp_path):
    tasks, clock = make(tmp_path, max_errors=3)
    bad = Counting(tasks, {}, fail=True)
    tasks.tasks.append(bad)
    for _ in range(3):
        tasks.tick(None, clock.t - 60)
    assert bad.off
    assert not tasks.tick(None, clock.t - 60)
    events = [json.loads(line)["event"] for line in (tmp_path / "log.jsonl").read_text().splitlines()]
    assert events.count("error") == 3 and events[-1] == "off"


def test_tasks_take_turns(tmp_path):
    tasks, clock = make(tmp_path)
    a, b = Counting(tasks, {}), Counting(tasks, {})
    tasks.tasks += [a, b]
    for _ in range(4):
        tasks.tick(None, clock.t - 60)
    assert (a.ran, b.ran) == (2, 2)


def test_from_env_off_by_default(monkeypatch):
    monkeypatch.delenv("PLI_TASKS", raising=False)
    assert T.from_env({}) is None


def test_from_env_skips_a_task_that_cannot_start(monkeypatch, tmp_path):
    monkeypatch.setenv("PLI_TASKS", "drafter,unknown")
    assert T.from_env({"drafter": None}) is None         # no drafter: nothing to run, the server starts


# -- prune --------------------------------------------------------------------------------------------------------
def touch(path: Path, size: int, age_s: float) -> None:
    path.write_bytes(b"x" * size)
    t = time.time() - age_s
    os.utime(path, (t, t))


def test_prune_plan_rules():
    now = 10_000_000.0
    day = 86400
    files = [
        (now - 60, 50, Path("fresh.safetensors")),              # rule 4: kept, counted
        (now - 8 * day, 10, Path("old.safetensors")),           # rule 1
        (now - 2 * day, 10, Path("x.partial.safetensors")),     # rule 3
        (now - 3600, 10, Path("y.partial.safetensors")),        # partial, recent: kept
        (now - 7200, 30, Path("a.safetensors")),
        (now - 9000, 30, Path("b.safetensors")),                # past the cap (50 + 30 + 30 > 100)
    ]
    out = {p.name: r for p, _, r in P.plan(files, now, days=7, cap_bytes=100)}
    assert out == {"old.safetensors": "unused", "x.partial.safetensors": "partial", "b.safetensors": "cap"}


def test_prune_steps_delete_and_log(tmp_path, monkeypatch):
    folder = tmp_path / "session-snapshots"
    folder.mkdir()
    touch(folder / "keep.safetensors", 10, 3600)
    for i in range(40):
        touch(folder / f"old{i}.safetensors", 10, 9 * 86400)
    tasks, clock = make(tmp_path)
    prune = P.Prune(tasks, {"session_dir": str(folder)})
    tasks.tasks.append(prune)
    assert prune.due(time.time())
    tasks.tick(None, clock.t - 60)                     # the listing
    assert len(prune.pending) == 40
    tasks.tick(None, clock.t - 60)                     # 32 deleted
    tasks.tick(None, clock.t - 60)                     # 8 more, the pass ends
    assert sorted(p.name for p in folder.iterdir()) == ["keep.safetensors"]
    assert not prune.due(time.time())                  # next pass in an hour
    last = [json.loads(line) for line in (tmp_path / "log.jsonl").read_text().splitlines()][-1]
    assert last["event"] == "pass" and last["deleted_unused"] == 40 and last["freed_bytes"] == 400
    again = P.Prune(tasks, {"session_dir": str(folder)})   # a restart reads the last pass back
    assert not again.due(time.time())


# -- drafter -------------------------------------------------------------------------------------------------------
class FakeDrafter:
    def __init__(self, path):
        self.path = path


def drafter_dir(root: Path, name: str) -> str:
    d = root / name
    d.mkdir()
    (d / "config.json").write_text("{}")
    return str(d.resolve())


def drafter_task(tmp_path, monkeypatch, drafter, *, fail=()):
    loads = []

    def reload(obj, path):
        if path in fail:
            raise ValueError("another shape")
        loads.append(path)
        obj.path = path
        return {}

    want = tmp_path / "a-servir.txt"
    monkeypatch.setenv("PLI_TASKS_DRAFTER_FILE", str(want))
    monkeypatch.setenv("PLI_TASKS_DRAFTER_MIN_DAYS", "3")
    monkeypatch.setenv("PLI_TASKS_DRAFTER_MIN_REQUESTS", "2")
    tasks, clock = make(tmp_path)
    task = D.DrafterSwap(tasks, {"drafter": drafter, "reload_drafter": reload})
    tasks.tasks.append(task)
    return task, tasks, clock, loads, want


def test_drafter_first_change_then_watch(tmp_path, monkeypatch):
    zlab, night1, night2 = (drafter_dir(tmp_path, n) for n in ("zlab", "n1", "n2"))
    drafter = FakeDrafter(zlab)
    task, tasks, clock, loads, want = drafter_task(tmp_path, monkeypatch, drafter)
    assert not task.due(time.time())                    # no file yet
    want.write_text(night1 + "\n")
    assert task.due(time.time())
    tasks.tick(None, clock.t - 60)
    assert loads == [night1] and task.state["served"] == night1 and task.state["previous"] == zlab
    os.utime(want, None)
    want.write_text(night2 + "\n")
    assert not task._change_due(time.time())            # the new one is watched 3 days first
    assert task._change_due(time.time() + 3.1 * 86400)


def test_drafter_guard_goes_back_and_refuses(tmp_path, monkeypatch):
    zlab, night1 = drafter_dir(tmp_path, "zlab"), drafter_dir(tmp_path, "n1")
    drafter = FakeDrafter(zlab)
    task, tasks, clock, loads, want = drafter_task(tmp_path, monkeypatch, drafter)
    for _ in range(3):                                  # z-lab: 50 % of the tokens from drafts
        task.observe(prompt=1000, generation=100, accepted=50, drafted=80)
    want.write_text(night1)
    tasks.tick(None, clock.t - 60)
    for _ in range(3):                                  # night1: 40 %
        task.observe(prompt=1000, generation=100, accepted=40, drafted=80)
    task.state["since"] -= 3.5 * 86400
    assert task._guard_due(time.time())
    task.step(None)
    assert drafter.path == zlab and loads == [night1, zlab]
    assert night1 in task.state["refused"] and task.state["served"] == zlab
    assert not task._change_due(time.time() + 10 * 86400)   # the file still names night1: ignored
    guard = [e for e in map(json.loads, (tmp_path / "log.jsonl").read_text().splitlines()) if e["event"] == "guard"][-1]
    assert guard["verdict"] == "back" and guard["ratio"] == 0.8


def test_drafter_guard_keeps_a_better_one(tmp_path, monkeypatch):
    zlab, night1 = drafter_dir(tmp_path, "zlab"), drafter_dir(tmp_path, "n1")
    drafter = FakeDrafter(zlab)
    task, tasks, clock, loads, want = drafter_task(tmp_path, monkeypatch, drafter)
    for _ in range(3):
        task.observe(prompt=20000, generation=100, accepted=50, drafted=80)
    want.write_text(night1)
    tasks.tick(None, clock.t - 60)
    for _ in range(3):
        task.observe(prompt=20000, generation=100, accepted=52, drafted=80)
    task.state["since"] -= 3.5 * 86400
    task.step(None)
    assert drafter.path == night1 and task.state["judged"] and not task.state["refused"]


def test_drafter_of_another_shape_is_refused(tmp_path, monkeypatch):
    zlab, other = drafter_dir(tmp_path, "zlab"), drafter_dir(tmp_path, "other")
    drafter = FakeDrafter(zlab)
    task, tasks, clock, loads, want = drafter_task(tmp_path, monkeypatch, drafter, fail=(other,))
    want.write_text(other)
    tasks.tick(None, clock.t - 60)
    assert drafter.path == zlab and other in task.state["refused"]
    assert not task.due(time.time())


def test_drafter_restored_after_a_restart(tmp_path, monkeypatch):
    zlab, night1 = drafter_dir(tmp_path, "zlab"), drafter_dir(tmp_path, "n1")
    drafter = FakeDrafter(zlab)
    task, tasks, clock, loads, want = drafter_task(tmp_path, monkeypatch, drafter)
    want.write_text(night1)
    tasks.tick(None, clock.t - 60)
    task.observe(prompt=100, generation=10, accepted=5)
    task.save()
    since = task.state["since"]
    # the server restarts on its configured drafter (z-lab): night1 comes back, its watch goes on
    restarted = FakeDrafter(zlab)
    task2, tasks2, clock2, loads2, _ = drafter_task(tmp_path, monkeypatch, restarted)
    assert task2.restore == night1
    task2.observe(prompt=100, generation=10, accepted=5)  # not counted: z-lab is the one running
    tasks2.tick(None, clock2.t - 60)
    assert restarted.path == night1 and loads2 == [night1]
    assert task2.state["since"] == since and task2.state["stats"]["lt4k"][0] == 1
    assert task2.state["previous"] == zlab


def test_drafter_day_line(tmp_path, monkeypatch):
    zlab = drafter_dir(tmp_path, "zlab")
    task, tasks, clock, loads, want = drafter_task(tmp_path, monkeypatch, FakeDrafter(zlab))
    task.observe(prompt=70000, generation=200, accepted=150)
    task.state["day"] = "2026-11-15"
    assert task.due(time.time())
    task.step(None)
    day = [e for e in map(json.loads, (tmp_path / "log.jsonl").read_text().splitlines()) if e["event"] == "day"][-1]
    assert day["sizes"]["ge64k"] == {"requests": 1, "share": 0.75}
    assert task.state["day_stats"] == {}


def test_compare_needs_enough_requests():
    new = {"lt4k": [1, 100, 10, 0]}
    old = {"lt4k": [100, 100, 50, 0]}
    assert D.compare(new, old, min_requests=50)["ratio"] is None


# -- in the server ------------------------------------------------------------------------------------------------
def test_steps_run_only_when_the_engine_is_idle():
    from tests.test_lane_server import SlowFakeEngine, expected_reply, make_app

    app = make_app(lanes=2, engine_factory=SlowFakeEngine)
    try:
        seen = []

        class Watch(T.Task):
            name = "watch"

            def due(self, now):
                return True

            def step(self, engine):
                sched = app.scheduler
                seen.append((engine.active_count, len(sched._fills), sched._queue.empty()))

        tasks = T.Tasks(calm_s=0.05)
        tasks.tasks.append(Watch(tasks, {}))
        app.scheduler.tasks = tasks
        deadline = time.perf_counter() + 5
        while not seen and time.perf_counter() < deadline:
            time.sleep(0.01)
        assert seen, "no step in a quiet engine"
        turn = [{"role": "user", "content": "a question while the tasks run"}]
        reply = app.chat(turn, max_tokens=24)
        assert reply["content"] == expected_reply(app, turn, 24)[1]
        assert all(active == 0 and fills == 0 and empty for active, fills, empty in seen)
        assert app.scheduler.task_steps == len(seen) or app.scheduler.task_steps >= len(seen) - 1
        context = app._task_context(app.engine.model)
        assert context["drafter"] is None and context["reload_drafter"] is None
    finally:
        app.close()


# -- falcon (the pure parts; the cycle itself runs on the real 27B) ---------------------------------------------------
def test_falcon_window_and_gains():
    from pli.tasks import falcon as F

    assert F.in_window(1, (1, 7)) and F.in_window(6, (1, 7)) and not F.in_window(7, (1, 7))
    assert F.in_window(23, (22, 3)) and F.in_window(2, (22, 3)) and not F.in_window(12, (22, 3))
    g, blocks = F.gains({"zlab4": {"tout": {"acceptes_glouton": 3.0, "blocs": 900}, "prose": {"acceptes_glouton": 2.0}},
                         "ajuste4": {"tout": {"acceptes_glouton": 3.3}, "prose": {"acceptes_glouton": 1.9}}})
    assert blocks == 900 and round(g["tout"], 4) == 0.1 and round(g["prose"], 4) == -0.05


def test_falcon_answers_and_anchors():
    import random

    from pli.tasks import falcon as F

    im_end = 9
    jetons = [1, 2, 3, 4, 5, 9, 7, 8, 1, 2, 3, 9]
    assert F.coupe_reponses([[1, 12, "prose"]], jetons, im_end) == [(1, 6, "prose")]
    assert F.nouvelles_reponses([[1, 12, "prose"], [6, 12, "outil"]], jetons, im_end, 7) == [(6, 12, "outil")]
    lots = F.ancrages_suivis([(1, 12, "prose")], 10, random.Random(0), 8)
    assert sorted(len(x) for x in lots) == [2, 8] and all(a < 11 for lot in lots for a, _, _ in lot)
    rows = [{"entete": 2, "jetons": list(range(30)), "ecrits": [[5, 20, "prose"], [25, 29, "outil"]]}]
    cut = F.preparer_sessions(rows, 20, im_end)
    assert len(cut[0]["jetons"]) == 22 and cut[0]["ecrits"] == [[5, 20, "prose"]]


def test_prune_cap_follows_the_disk(tmp_path, monkeypatch):
    folder = tmp_path / "snap"
    folder.mkdir()
    for i in range(4):
        touch(folder / f"f{i}.safetensors", 1000, 3600 + i)
    monkeypatch.setenv("PLI_TASKS_PRUNE_CAP_PERCENT", "50")
    monkeypatch.delenv("PLI_TASKS_PRUNE_CAP_GB", raising=False)

    class Disk:
        f_bavail, f_frsize = 2000, 1                  # 2,000 bytes free + 4,000 in the folder: cap 3,000 at 50 %

    monkeypatch.setattr(P.os, "statvfs", lambda path: Disk)
    tasks, clock = make(tmp_path)
    prune = P.Prune(tasks, {"session_dir": str(folder)})
    prune._list()
    assert prune.pass_stats["cap_bytes"] == 3000 and [p.name for p, _, _ in prune.pending] == ["f3.safetensors"]
