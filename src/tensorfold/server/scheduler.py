"""The engine thread: requests queued as jobs, admitted into the engine's lanes, stepped round by round."""

from __future__ import annotations

from dataclasses import dataclass, field
import heapq
import itertools
import os
from pathlib import Path
import queue
import threading
import time
import traceback
from typing import Any, Callable

from tensorfold.engine.lane_engine import LaneStream
from tensorfold.server.cancellation import Cancellation, RequestCancelled
from tensorfold.server.checkpoints import CheckpointStore
from tensorfold.server.errors import RequestError, RoundError
from tensorfold.server.live import ChunkRate, Meter
from tensorfold.server.prompt_fill import Filling, PromptFill
from tensorfold.server.stream_gate import StreamGate
from tensorfold.pli import events as _ev

# PLI (idea from Kapitsa #206 and #243): heartbeat between two requests. Without GPU work, Metal releases the wired weights
# about 2 s after the last computation and the next pass waits for them to come back (measured, 14 GiB: 256 ms instead
# of 29; with a small computation every 500 ms: 52 to 58 ms). PLI_HEARTBEAT_MS = period (0 = off), PLI_HEARTBEAT_WINDOW_S = duration
# after the last activity, PLI_HEARTBEAT_ANE=1 = a Neural Engine keep-awake call at the same rate. No reply computation touched.
_HEARTBEAT_S = float(os.environ.get("PLI_HEARTBEAT_MS", "0") or 0) / 1e3
_HEARTBEAT_WINDOW_S = float(os.environ.get("PLI_HEARTBEAT_WINDOW_S", "600") or 600)
_HEARTBEAT_ANE = os.environ.get("PLI_HEARTBEAT_ANE", "0") == "1"

@dataclass
class ChatJob:
    job_id: str
    prompt_ids: list[int]
    max_tokens: int
    temperature: float
    history_len: int = 0
    # Snapshot system-and-tools blocks for reuse across sessions; history boundaries already include session-specific text.
    shared_prefix_lens: tuple[int, ...] = ()
    # the request's proposer (suffix copies, a drafter model, tool-call structure); None: the engine's default
    proposer: Any = None
    # False ("draft": false): one token a round and no drafts, the serial reference output is checked against
    drafts: bool = True
    # exact_sampling.Sampling for this request (None = greedy)
    sampling: Any = None
    # thinking budget (0: none) and the tokens that close a think block ("\n</think>\n\n"; LaneStream)
    think_budget: int = 0
    think_close: tuple[int, ...] = ()
    think_end: int = -1
    # Background jobs yield to foreground arrivals, and callers restart preempted jobs from the beginning.
    background: bool = False
    preempted: bool = False
    submitted_at: float = field(default_factory=time.perf_counter)
    started_at: float = 0.0
    prefilled_at: float = 0.0
    finished_at: float = 0.0
    cached_tokens: int = 0
    stream: LaneStream | None = None
    error: BaseException | None = None
    chunks: "queue.Queue[list[int] | None]" = field(default_factory=queue.Queue)
    done: threading.Event = field(default_factory=threading.Event)
    cancellation: Cancellation = field(default_factory=Cancellation)
    ignore_eos: bool = False
    stop_check: Callable[[list[int]], bool] | None = None
    call_gate: Any = None                   # tool_choice "required": the answer opens a tool call (LaneStream)
    constraint: Any = None                  # response_format's grammar (engine.grammar.Constraint), or None
    vision: Any = None
    # a decision: prefill ends at these labels' last-row logits and joins no round (empty: a chat)
    label_ids: tuple[int, ...] = ()
    scored: tuple[list[float], float] | None = None
    pli_conv: str = ""                      # PLI 3.6.0: fingerprint of the conversation (pli/events.py)


class _JobQueue(queue.PriorityQueue):
    """Jobs in arrival order, background jobs after every other job."""

    def __init__(self) -> None:
        super().__init__()
        self._order = itertools.count()

    def put(self, job: ChatJob, block: bool = True, timeout: float | None = None) -> None:
        super().put((1 if job.background else 0, next(self._order), job), block, timeout)

    def get(self, block: bool = True, timeout: float | None = None) -> ChatJob:
        return super().get(block, timeout)[2]

    def foreground_waiting(self) -> bool:
        return self.peek_foreground() is not None

    def peek_foreground(self) -> ChatJob | None:
        with self.mutex:
            return self.queue[0][2] if self.queue and self.queue[0][0] == 0 else None

    def small(self, limit: int) -> ChatJob | None:
        """PLI: the first foreground request of at most ``limit`` tokens in the queue, without removing it (None: none)."""

        with self.mutex:
            for prio, _, job in sorted(self.queue, key=lambda e: e[:2]):
                if prio == 0 and len(job.prompt_ids) <= limit and not job.cancellation.cancelled:
                    return job
        return None

    def remove_job(self, job: ChatJob) -> bool:
        """PLI: remove ``job`` from the queue (true if it was there)."""

        with self.mutex:
            before = len(self.queue)
            self.queue[:] = [entry for entry in self.queue if entry[2] is not job]
            heapq.heapify(self.queue)
            return len(self.queue) < before

    def remove(self, cancellation: Cancellation) -> list[ChatJob]:
        with self.mutex:
            removed = [entry[2] for entry in self.queue if entry[2].cancellation is cancellation]
            self.queue[:] = [entry for entry in self.queue if entry[2].cancellation is not cancellation]
            heapq.heapify(self.queue)
            return removed


class Scheduler(PromptFill):
    """Owns the engine on one thread: admits jobs, steps rounds, delivers tokens."""

    def __init__(
        self,
        engine: Any,
        *,
        lanes: int,
        eos_ids: frozenset[int],
        checkpoints: CheckpointStore | None = None,
        proposer_factory: Callable[[], Any] | None = None,
        idle_wait: float = 0.02,
        snapshot_dir: Any = None,
        session_dir: Any = None,
        model_id: str = "",
        admission: Any = None,
        prompt_memory: Any = None,
        decode_share: float = 0.25,
    ) -> None:
        if lanes < 1:
            raise ValueError("lanes must be positive")
        if decode_share < 0:
            raise ValueError("decode_share must be 0 or more")
        # ``engine.memory.Admission``: a job starts beside live streams only while the projected memory fits
        self.admission = admission
        self.snapshot_dir = snapshot_dir
        self.session_dir = session_dir
        self.model_id = model_id
        self.disk_blocks: Any = None
        self.session_blocks: Any = None
        if model_id and (snapshot_dir is not None or session_dir is not None):
            from tensorfold.engine.prefix_snapshots import DiskBlocks, remove_stale_partials

            for directory in (snapshot_dir, session_dir):
                freed = 0 if directory is None else remove_stale_partials(Path(directory))
                if freed:
                    print(f"[tensorfold] removed {freed / 1024**3:.2f} GiB of unfinished snapshot writes in "
                          f"{directory}", flush=True)
            if snapshot_dir is not None:
                self.disk_blocks = DiskBlocks(Path(snapshot_dir), model_id)
            if session_dir is not None:
                self.session_blocks = DiskBlocks(Path(session_dir), model_id)
        self.engine = engine
        self.lanes = int(lanes)
        self.eos_ids = eos_ids
        self.checkpoints = checkpoints
        self.prompt_memory = prompt_memory
        self.proposer_factory = proposer_factory
        self.idle_wait = float(idle_wait)
        self._active_seen = time.monotonic()
        self._beat_at = 0.0
        self.heartbeats = 0
        self.slow_round_ms = 1000.0
        self._held: ChatJob | None = None
        self._queue = _JobQueue()
        self._engine_calls: queue.Queue[tuple[Callable[[Any], Any], queue.Queue[Any]]] = queue.Queue()
        self.preemptions = 0
        self._jobs: dict[str, ChatJob] = {}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="tensorfold-engine", daemon=True)
        self.rounds = 0
        self.failed_rounds = 0
        self.completed = 0
        self.cancelled = 0
        self.starts = 0
        self._starting: ChatJob | None = None
        self._fills: list[Filling] = []         # open prompts, oldest first, each filling a chunk a step
        # while prompts fill, rounds get decode_share of each chunk's time, at most fill_rounds between two chunks
        self.decode_share = float(decode_share)
        # PLI: adjustable cap (PLI_ROUNDS_BETWEEN_PREFILLS). At 32, it cancels any decode_share beyond 0.25 (measured on 09/10).
        self.fill_rounds = max(1, int(os.environ.get("PLI_ROUNDS_BETWEEN_PREFILLS", "32")))
        self.fill_guard = 8             # a prompt passed over this many chunks takes the next one (no starvation)
        self._credit, self._rounds_left = 0.0, 0
        self._released_at = 0            # ``starts`` when MLX's freed buffers were last handed back
        # Evaluate and save cache arrays on the scheduler thread that owns their streams during shutdown.
        self.on_stop: Callable[[], Any] | None = None
        # streams take memory as they grow: the newest waits, or ends, when the next round's growth can't fit
        self.gate = (StreamGate(prompt_memory, admission.memory.per_token, admission.memory.round_bytes,
                                admission.budget, lanes=self.lanes)
                     if admission is not None and prompt_memory is not None and self.lanes > 1 else None)
        # PLI (item 2 of plan 2.2, 11/10): PLI_ADMIT_BY_WORK=1 makes admission count the prefill work of the PromptMemory
        # guard (measured on a real chunk) instead of the prefill_bytes line. Off by default: original admission.
        # PLI_ADMIT_SMALL=N: only for a request of at most N tokens ("the small ones go first", 11/10).
        small_max = int(os.environ.get("PLI_ADMIT_SMALL", "0") or 0)
        self._pli_small_max = max(0, small_max)  # > 0: a small request also goes ahead of a request held at admission
        if ((os.environ.get("PLI_ADMIT_BY_WORK", "0") == "1" or small_max > 0) and admission is not None
                and prompt_memory is not None and hasattr(prompt_memory, "_work") and hasattr(admission, "work_bytes")):
            admission.work_bytes, admission.small_max = prompt_memory._work, max(0, small_max)
            print("[pli] admission: prefill work counted by PromptMemory"
                  + (f" for requests of at most {small_max} tokens (PLI_ADMIT_SMALL)" if small_max > 0
                     else " (PLI_ADMIT_BY_WORK=1)"), flush=True)
        self.stall_s = 120.0            # no round, start or finish while requests wait: dump stacks
        self.stall_prefill_s = 900.0    # the same while one prefill runs
        self._watchdog = threading.Thread(target=self._watch, name="tensorfold-watchdog", daemon=True)
        self.decoded, self.prefilled = Meter(), ChunkRate()       # the live line's decode and prefill tok/s

    # -- lifecycle ------------------------------------------------------------
    def start(self) -> None:
        if not self._thread.is_alive():
            self._thread.start()
        if not self._watchdog.is_alive():
            self._watchdog.start()

    def _watch(self) -> None:
        """Dump every thread stack once when waiting requests stall, including prefill chunks; SIGUSR1 also dumps stacks on demand."""

        import faulthandler
        import sys

        mark: tuple[int, int, int, int, bool] | None = None
        since = time.perf_counter()
        dumped = False
        while not self._stop.wait(1.0):
            waiting = self._queue.qsize() + (self._held is not None) + len(self._jobs) + (self._starting is not None)
            starting = self._starting is not None
            now_mark = (self.rounds, self.starts, self.completed, getattr(self.engine, "prefill_chunks", 0), starting)
            now = time.perf_counter()
            if now_mark != mark or not waiting:
                mark, since, dumped = now_mark, now, False
                continue
            limit = self.stall_prefill_s if starting else self.stall_s
            if now - since > limit and not dumped:
                dumped = True
                print(f"[tensorfold] stalled {now - since:.0f}s: queued={self._queue.qsize()} "
                      f"held={self._held is not None} jobs={len(self._jobs)} active={self.engine.active_count} "
                      f"starting={starting}; every thread's stack follows", flush=True)
                faulthandler.dump_traceback(file=sys.stderr, all_threads=True)

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=timeout)

    def submit(self, job: ChatJob) -> None:
        if self._stop.is_set():
            raise RuntimeError("the scheduler is closed")
        self._queue.put(job)

    def on_engine(self, fn: Callable[[Any], Any], timeout: float = 600.0) -> Any:
        """Run ``fn`` on the engine thread once no stream is live and no prompt is open."""

        if self._stop.is_set():
            raise RuntimeError("the scheduler is closed")
        done: queue.Queue[Any] = queue.Queue(1)
        self._engine_calls.put((fn, done))
        try:
            result = done.get(timeout=timeout)
        except queue.Empty as exc:
            raise TimeoutError("the engine did not score the prompt in time") from exc
        if isinstance(result, BaseException):
            raise result
        return result

    def _run_engine_call(self) -> bool:
        try:
            fn, done = self._engine_calls.get_nowait()
        except queue.Empty:
            return False
        try:
            done.put(fn(self.engine))
        except Exception as exc:  # noqa: BLE001 - the waiter raises this on its own thread
            exc.__traceback__ = exc.__cause__ = exc.__context__ = None
            done.put(exc)
        return True

    def _fail_engine_calls(self) -> None:
        while True:
            try:
                _, done = self._engine_calls.get_nowait()
            except queue.Empty:
                return
            done.put(RuntimeError("the scheduler is closed"))

    def cancel(self, cancellation: Cancellation) -> None:
        cancellation.cancel()
        for job in self._queue.remove(cancellation):
            self._finish_cancelled(job)

    def _finish_cancelled(self, job: ChatJob) -> None:
        if job.done.is_set():
            return
        job.error = RequestCancelled("request cancelled")
        job.proposer = None
        self.cancelled += 1
        _ev.ev("cancelled", job)
        self._finish(job)

    def _discard_job(self, job: ChatJob) -> None:
        self._jobs.pop(job.job_id, None)
        if job.stream is not None:
            self.engine.discard_stream(job.stream)
            job.stream.history_checkpoints = []
            job.stream.proposer = None
        self._finish_cancelled(job)

    def _cancel_active(self) -> None:
        if self._held is not None and self._held.cancellation.cancelled:
            self._finish_cancelled(self._held)
            self._held = None
        for filling in [f for f in self._fills if f.job.cancellation.cancelled]:
            self._fill(filling, abort=RequestCancelled("request cancelled"))      # stops between chunks, progress kept
        for job in list(self._jobs.values()):
            if job.cancellation.cancelled:
                self._discard_job(job)

    @property
    def active(self) -> int:
        return len(self._jobs)

    @property
    def waiting(self) -> int:
        """Requests not started yet: queued, or held until they fit."""

        return self._queue.qsize() + (self._held is not None)

    @property
    def filling(self) -> list[Any]:
        """The jobs whose prompts are prefilling."""

        return [f.job for f in self._fills]

    @staticmethod
    def finish_job(job: ChatJob, reason: str = "stop") -> None:
        """Ask the engine to stop a stream at the next round (safe from any thread)."""

        stream = job.stream
        if stream is not None and not stream.finished:
            stream.finish_reason = reason
            stream.finished = True
            stream.finished_at = time.perf_counter()

    # -- the loop -------------------------------------------------------------
    def _run(self) -> None:
        try:
            self._loop()
        finally:
            self._fail_engine_calls()
            for filling in list(self._fills):
                self._fill(filling, abort=RequestCancelled("server stopping"))
            if self.on_stop is not None:
                try:
                    self.on_stop()
                except Exception as exc:  # noqa: BLE001 - the shutdown must finish
                    print(f"[tensorfold] shutdown hook failed: {type(exc).__name__}: {exc}", flush=True)

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._cancel_active()
            self._preempt_background()
            self._admit()
            self._starting = self._fills[0].job if self._fills else None
            self._retire_externally_finished()
            if self._chunk_due():
                self._fill()
                continue
            if self.engine.active_count == 0:
                if self._held is None and not self._fills:
                    if self._run_engine_call():
                        continue
                    self._release_idle()
                    try:
                        self._held = self._queue.get(timeout=self.idle_wait)
                    except queue.Empty:
                        if _HEARTBEAT_S > 0:
                            self._heartbeat()
                        continue
                continue  # _admit starts it
            self._active_seen = time.monotonic()
            if not self._fills:
                self._credit, self._rounds_left = 0.0, 0     # no prompt waits: rounds owe nothing
            if self.gate is not None:
                self._gate_round()
            started = self.clock()
            try:
                landed = self.engine.step()
            except Exception as exc:  # noqa: BLE001 - one bad round must not kill the server
                self.failed_rounds += 1
                traceback.print_exception(exc)
                error_type, message = type(exc).__name__, str(exc)
                exc.__traceback__ = exc.__cause__ = exc.__context__ = None
                for job in list(self._jobs.values()):
                    job.error = RoundError(error_type, message)
                    if job.stream is not None:
                        self.engine.discard_stream(job.stream)
                        job.stream.finish_reason = "error"
                        job.stream.history_checkpoints = []
                        job.stream.proposer = None
                    job.proposer = None
                    self._finish(job)
                self._jobs.clear()
                self.engine.reset()
                job = landed = tokens = None
                continue
            self.rounds += 1
            self._spend_round(self.clock() - started)
            self.decoded.add(sum(len(t) for t in landed.values()))
            self._cancel_active()
            if self.engine.round_stats:
                last = self.engine.round_stats[-1]
                if last.total_ms > self.slow_round_ms:
                    print(f"[tensorfold] slow round {last.total_ms:.0f} ms streams={last.streams} "
                          f"width={last.width} rows={last.rows} forward={last.forward_ms:.0f}", flush=True)
                if len(self.engine.round_stats) > 4096:
                    del self.engine.round_stats[:2048]
            for stream_id, tokens in landed.items():
                job = self._jobs.get(stream_id)
                if job is None:
                    continue
                if tokens:
                    job.chunks.put(list(tokens))
                if job.stream is not None and job.stream.finished:
                    del self._jobs[stream_id]
                    self._retire(job)
            job = landed = tokens = None

    def _heartbeat(self) -> None:
        """Between two requests (engine idle, nothing queued): a small GPU computation every PLI_HEARTBEAT_MS, during
        PLI_HEARTBEAT_WINDOW_S after the last activity; in the engine thread (the only one that holds MLX's GPU stream)."""

        now = time.monotonic()
        if now - self._beat_at < _HEARTBEAT_S or now - self._active_seen > _HEARTBEAT_WINDOW_S:
            return
        self._beat_at = now
        try:
            import mlx.core as mx

            mx.eval(mx.ones((8, 8)) + 1)
            if _HEARTBEAT_ANE:
                from tensorfold.pli.ane import ffn as _ffn

                _ffn.heartbeat()
            self.heartbeats += 1
            if self.heartbeats == 1:
                print(f"[PLI] heartbeat: every {_HEARTBEAT_S * 1e3:.0f} ms for {_HEARTBEAT_WINDOW_S:.0f} s after the last activity"
                      + (", Neural Engine included" if _HEARTBEAT_ANE else ""), flush=True)
        except Exception:  # noqa: BLE001 - the heartbeat never makes the server fail
            pass

    def _release_idle(self) -> None:
        """Once no stream is left and nothing waits, hand MLX's freed buffers back: no round can want them now."""

        if self.prompt_memory is not None and self._released_at != self.starts and self._queue.empty():
            self._released_at = self.starts
            getattr(self.engine, "release_rounds", lambda: None)()
            self.prompt_memory.release_freed()

    def _preempt_background(self) -> None:
        """Release background work until the next foreground request has both a lane and enough memory."""

        waiting = self._held
        if waiting is None or waiting.background:
            waiting = self._queue.peek_foreground()
        if waiting is None or waiting.cancellation.cancelled:
            return
        for filling in [f for f in reversed(self._fills) if f.job.background]:      # the newest first
            if self._room() and self._fits(waiting):
                return
            self._preempt_filling(filling)
        for job in self._jobs.values():
            if self._room() and self._fits(waiting):
                break
            if job.background and not job.preempted and job.stream is not None and not job.stream.finished:
                job.preempted = True
                self.preemptions += 1
                self.engine.discard_stream(job.stream)
                job.stream.finish_reason = "preempted"
                job.stream.history_checkpoints = []

    def _retire_externally_finished(self) -> None:
        for stream_id, job in list(self._jobs.items()):
            if job.stream is not None and job.stream.finished:
                del self._jobs[stream_id]
                self._retire(job)

    def _room(self) -> bool:
        """Whether a lane is free: live streams and open prompts each hold one."""

        return self.engine.active_count + len(self._fills) < self.lanes

    def _admit(self) -> None:
        """Admit fitting jobs while lanes are free; each opens a prompt that fills beside the others, a chunk a step."""

        while self._room():
            job = self._held
            self._held = None
            if job is not None and job.background and self._queue.foreground_waiting():
                self._queue.put(job)          # a request that arrived since goes first
                job = None
            if job is None:
                try:
                    job = self._queue.get_nowait()
                except queue.Empty:
                    return
            if not self._fits(job):
                if not getattr(job, "_pli_ev_held", False):        # PLI 3.6.0: event log
                    job._pli_ev_held = True
                    _ev.ev("waiting", job, prompt=len(job.prompt_ids), prefills=len(self._fills),
                           decodes=self.engine.active_count)
                if os.environ.get("PLI_LOG_TIMES", "") == "1" and not getattr(job, "_pli_held", False):
                    # PLI (11/10): log of a request held at admission, once per request
                    job._pli_held = True
                    memory = self.prompt_memory
                    fits = None if memory is None else memory.would_fit(len(job.prompt_ids), self._reserved(int(job.max_tokens)))
                    print(f"[pli] {time.strftime('%H:%M:%S')} request of {len(job.prompt_ids)} tokens held at admission: "
                          f"{self.engine.active_count} decode(s) running, {len(self._fills)} prefill(s) open "
                          f"(left {[f.left for f in self._fills]}), memory enough for it alone beside: {fits}", flush=True)
                    adm = self.admission
                    if adm is not None:
                        gib = 2 ** 30
                        live = [(n, min(len(j.prompt_ids) + int(j.max_tokens), n + self._reserved(int(j.max_tokens))))
                                for j in self._jobs.values() if j.stream is not None and not j.stream.finished
                                for n in [j.stream.context_len]]
                        live += [(len(f.job.prompt_ids) - f.left, len(f.job.prompt_ids) + self._reserved(int(f.job.max_tokens)))
                                 for f in self._fills]
                        widest = max([len(job.prompt_ids), *(len(f.job.prompt_ids) for f in self._fills)])
                        long = len(job.prompt_ids) + self._reserved(int(job.max_tokens))
                        grow = sum(max(0, m - n) for n, m in live) * adm.memory.per_token
                        print(f"[pli] projection (GiB): used {adm.used() / gib:.2f} + growth of the others {grow / gib:.2f} "
                              f"+ the request {adm.memory.stream_bytes(long) / gib:.2f} + work {max(adm.round_bytes(len(live) + 1), adm.prefill_work_bytes(widest, len(job.prompt_ids))) / gib:.2f} "
                              f"(prefill of {widest} tokens {adm.prefill_work_bytes(widest, len(job.prompt_ids)) / gib:.2f}, original {adm.memory.prefill_bytes(widest) / gib:.2f}, of the request alone {adm.prefill_work_bytes(len(job.prompt_ids), len(job.prompt_ids)) / gib:.2f}) "
                              f"= {adm.projected(widest, long, live, len(job.prompt_ids)) / gib:.2f} for a budget of {adm.budget / gib:.2f}", flush=True)
                self._held = job              # waits until a live stream finishes and frees its memory
                # PLI (item 2 of plan 2.2): the held request no longer holds the small requests that arrived behind it.
                small = self._queue.small(self._pli_small_max) if self._pli_small_max > 0 else None
                if small is None or not self._fits(small) or not self._queue.remove_job(small):
                    return
                if os.environ.get("PLI_LOG_TIMES", "") == "1":
                    print(f"[pli] {time.strftime('%H:%M:%S')} request of {len(small.prompt_ids)} tokens admitted ahead of the held "
                          f"request of {len(job.prompt_ids)} tokens", flush=True)
                _ev.ev("jumps_ahead", small, ahead_of=job.job_id, prompt=len(small.prompt_ids))
                job = small
            if job.cancellation.cancelled:
                self._finish_cancelled(job)
                continue
            if self.decode_share <= 0:
                self._start_job(job)                  # decode_share 0: each prompt whole before any round (0.3.6.2)
                continue
            self._open_job(job)
            if self._chunk_due():
                self._fill()                     # no round is owed: a chunk now, as before

    def _fits(self, job: ChatJob) -> bool:
        """Alone, prompt admission checks memory; beside streams or open prompts, both projections must fit."""

        if self.engine.active_count == 0 and not self._fills:
            return True
        reply = self._reserved(int(job.max_tokens))
        memory = self.prompt_memory
        if memory is not None and not memory.would_fit(len(job.prompt_ids), reply):
            return False
        if self.admission is None:
            return True
        live = [(n, min(len(j.prompt_ids) + int(j.max_tokens), n + self._reserved(int(j.max_tokens))))
                for j in self._jobs.values() if j.stream is not None and not j.stream.finished
                for n in [j.stream.context_len]]
        # an open prompt grows from the rows it holds to its prompt and reply horizon; its chunks' workspace counts too
        live += [(len(f.job.prompt_ids) - f.left, len(f.job.prompt_ids) + self._reserved(int(f.job.max_tokens)))
                 for f in self._fills]
        widest = max([len(job.prompt_ids), *(len(f.job.prompt_ids) for f in self._fills)])
        if getattr(self.admission, "work_bytes", None) is None:
            return self.admission.admits(widest, len(job.prompt_ids) + reply, live)       # original admission
        return self.admission.admits(widest, len(job.prompt_ids) + reply, live, request=len(job.prompt_ids))

    def _reserved(self, reply: int) -> int:
        """The reply tokens a stream holds memory for ahead of its length: the gate's horizon when it guards rounds."""

        return reply if self.gate is None else min(reply, self.gate.horizon)

    def _gate_round(self) -> None:
        """Hold back the newest streams whose growth can't fit, or end the newest when even the oldest can't grow."""

        live = sorted((j for j in self._jobs.values() if j.stream is not None and not j.stream.finished),
                      key=lambda j: j.started_at)
        plan = self.gate.plan([(j.stream.stream_id, j.stream.context_len, len(j.prompt_ids) + int(j.max_tokens))
                               for j in live])
        if set(plan.paused) != self.engine.paused:
            print(f"[tensorfold] memory: {len(plan.paused)} of {len(live)} streams wait for room (newest first)",
                  flush=True)
        self.engine.paused = set(plan.paused)
        for job in (j for j in live if j.stream.stream_id in plan.ended):
            job.error = RequestError(
                f"This server ran out of memory with {len(live)} streams decoding, so the newest (this request, "
                f"after {len(job.stream.emitted)} tokens) was stopped for the older ones to finish. Retry it, "
                "shorten the prompt or max_tokens, or start the server with a smaller --parallel.")
            print(f"[tensorfold] memory: ended {job.job_id}, the newest of {len(live)} streams", flush=True)
            self.engine.discard_stream(job.stream)
            job.stream.finish_reason = "error"
            job.stream.history_checkpoints = []
            del self._jobs[job.stream.stream_id]
            self._finish(job)
            if self.prompt_memory is not None:
                self.prompt_memory.release_freed()      # its arrays are free now: the next plan sees the room

    def _read_disk_block(self, prompt: list[int], usable: Any = None) -> None:
        """Put the longest stored prefix ``usable`` accepts (system blocks, saved conversations) in the store."""

        if self.checkpoints is None or (self.disk_blocks is None and self.session_blocks is None):
            return
        try:
            have = self.checkpoints.longest(prompt, usable)
            found: Any = None
            for blocks, pinned in ((self.disk_blocks, True), (self.session_blocks, False)):
                hit = None if blocks is None else blocks.best(prompt, have if found is None else len(found[1]), usable)
                if hit is not None:
                    found = (hit[0], hit[1], blocks, pinned)
            if found is None:
                return
            from tensorfold.engine.prefix_snapshots import load_snapshot

            started = time.perf_counter()
            if self.prompt_memory is not None and not self.prompt_memory.allow_load(found[0].stat().st_size):
                print(f"[tensorfold] left the stored prefix of {len(found[1])} tokens on disk: memory cannot "
                      "hold a copy beside what runs; this prompt re-prefills it", flush=True)
                _ev.ev("stays_on_disk", conv=_ev.conv_of_tokens(prompt), tokens=len(found[1]))
                return
            loaded = load_snapshot(found[0], self.model_id)
            if loaded is None:
                return
            tokens, cache = loaded
            self.checkpoints.insert(tokens, cache, last_prompt=tokens, pinned=found[3])
            found[2].touch(tokens)
            _ev.ev("disk_read", conv=_ev.conv_of_tokens(prompt), tokens=len(tokens),
                   kind="system_block" if found[3] else "conversation", duration=time.perf_counter() - started)
            print(f"[tensorfold] read {'system-block' if found[3] else 'conversation'} snapshot tokens={len(tokens)} "
                  f"from disk in {time.perf_counter() - started:.2f}s", flush=True)
        except Exception as exc:  # noqa: BLE001 - a bad file costs a prefill, never the request
            print(f"[tensorfold] snapshot read failed: {type(exc).__name__}: {exc}", flush=True)

    def _keep_checkpoints(self, job: ChatJob, shared_at: set[int]) -> None:
        """Store the prefixes the job's prefill kept (system blocks pinned and saved to disk), once."""

        stream = job.stream
        if stream is None or job.vision is not None:
            return
        kept, stream.history_checkpoints = stream.history_checkpoints, []
        for tokens, snapshot in kept if self.checkpoints is not None else ():
            shared = len(tokens) in shared_at
            self.checkpoints.insert(tokens, snapshot, last_prompt=job.prompt_ids, pinned=shared)
            if self.snapshot_dir is not None and shared:
                self._persist(tokens, snapshot)

    def _persist(self, tokens: list[int], cache: list[Any]) -> None:
        """Write a system-block snapshot to disk once, so a restart does not lose it."""

        from tensorfold.engine.prefix_snapshots import save_snapshot

        try:
            started = time.perf_counter()
            path = save_snapshot(self.snapshot_dir, self.model_id, tokens, cache)
            if path is not None:
                _ev.ev("system_block_saved", conv=_ev.conv_of_tokens(tokens), tokens=len(tokens),
                       duration=time.perf_counter() - started)
                print(f"[tensorfold] saved system-block snapshot tokens={len(tokens)} "
                      f"in {time.perf_counter() - started:.1f}s", flush=True)
        except Exception as exc:  # noqa: BLE001 - a full disk must not fail the request
            print(f"[tensorfold] snapshot save failed: {type(exc).__name__}: {exc}", flush=True)

    def _retire(self, job: ChatJob) -> None:
        if job.cancellation.cancelled:
            self._discard_job(job)
            return
        stream = job.stream
        if stream is not None and getattr(stream, "error", None) is not None and job.error is None:
            job.error = stream.error                  # its grammar failed: this request alone answers with the error
        retained = self.engine.finished_caches.pop(job.job_id, None)
        if retained is not None and self.checkpoints is not None and job.vision is None:
            tokens, cache = retained
            if len(tokens) > len(job.prompt_ids):
                self.checkpoints.insert(tokens, cache, last_prompt=job.prompt_ids)
        if stream is not None and stream in self.engine.streams:
            self.engine.streams.remove(stream)
        self.completed += 1
        self._finish(job)

    @staticmethod
    def _finish(job: ChatJob) -> None:
        job.finished_at = time.perf_counter()
        job.chunks.put(None)
        job.done.set()
