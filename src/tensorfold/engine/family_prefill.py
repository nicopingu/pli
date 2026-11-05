"""The family rounds' prefill: prompt chunks, checkpoints at chunk starts, the first token and first drafts."""

from __future__ import annotations

from functools import partial
import time
from typing import Any, Iterator, Sequence

from tensorfold.engine.family_common import cache_arrays, drop_spares
from tensorfold.engine.prefill_plan import PromptChunks
_KEEP_REPLY = __import__("os").environ.get("PLI_KEEP_REPLY", "0") == "1"   # PLI (23/10/2026): a state at the end of a reply resumes outside chunk starts
from tensorfold.pli import trace_layers as _tc
from tensorfold.pli.falcon import capture as _falcon


def drain(steps: Iterator[Any]) -> Any:
    """Run a generator of prefill steps to its end and return its value."""

    while True:
        try:
            next(steps)
        except StopIteration as done:
            return done.value



def _pli_pieces(begin: int, end: int):
    """PLI: cut [begin, end) into passes of PLI_CHUNK_TOKENS tokens (None if off or if the chunk is already short)."""
    import os
    try:
        step = int(os.environ.get("PLI_CHUNK_TOKENS", "0"))
    except ValueError:
        step = 0
    if step < 64 or end - begin <= step + step // 2:
        return None
    bounds = list(range(begin, end, step))
    if end - bounds[-1] < step // 2 and len(bounds) > 1:
        bounds.pop()                                   # a remainder too short joins the previous pass
    return [(a, b) for a, b in zip(bounds, bounds[1:] + [end])]


def _pli_layers() -> int:
    """PLI: number of layers per step of the interleaved split (PLI_CHUNK_LAYERS; 0 = off)."""
    import os
    try:
        return max(0, int(os.environ.get("PLI_CHUNK_LAYERS", "0")))
    except ValueError:
        return 0



def _mtp_in_window(chunk_end: int, prompt_end: int) -> bool:
    """PLI (24/10/2026, PLI_MTP_WINDOW=N): the MTP head reads only the chunks that end within the last N
    tokens of the prompt; before, neither its layer nor the last layer of the 27B is computed. 0 or absent: the whole prompt.
    The drafts see less context; the text does not change (exact verification)."""

    import os

    try:
        n = int(os.environ.get("PLI_MTP_WINDOW", "0") or 0)
    except ValueError:
        n = 0
    return n <= 0 or chunk_end > prompt_end - n


def _head_reads(model: Any, chunk_end: int, prompt_end: int) -> bool:
    """The window applies only to a head that reads the final state (MTP); DFlash2 alone always reads the whole prompt."""

    return not getattr(model, "draft_reads_hidden", True) or _mtp_in_window(chunk_end, prompt_end)

class FamilyPrefill:
    """Prefill for ``FamilyRounds``."""

    prefill_tokens = 0          # prompt tokens fed, for the server's live line
    prefill_pass = 8            # plan chunks one forward may take while a prompt fills alone (1: a chunk a forward)
    pass_cache = 16 * 1024**3   # MLX's cache of freed buffers during a pass, where the memory budget has room for it

    def _family_feed(self, tokens: Sequence[int], cache: list[Any], chunks: Sequence[tuple[int, int]],
                     prompt_data: Any = None) -> Any:
        """Absorb the ``chunks`` ([begin, end) of ``tokens``); the last hidden state [1, 1, D], draft head fed too."""

        return drain(self._family_feed_steps(tokens, cache, chunks, prompt_data))

    def _family_feed_steps(self, tokens: Sequence[int], cache: list[Any], chunks: Sequence[tuple[int, int]],
                           prompt_data: Any = None, wide: bool = False,
                           widths: list[int] | None = None, raised: list[bool] | None = None,
                           whole: list[Any] | None = None, merge_at: int | None = None,
                           stream_id: Any = None) -> Iterator[None]:
        """``_family_feed`` as steps: yields between forwards; chunks, absorbs and eval stay whole."""

        import mlx.core as mx

        from tensorfold.pli import merge_tail

        last = None
        feed = getattr(self.model, "prefill", None) or self.model.hidden
        passes = wide and prompt_data is None and getattr(self.model, "prompt_pass", True)
        together = getattr(self.model, "hidden_pass", None) if passes else None
        reach = max(1, int(self.prefill_pass)) if together is not None else 1
        self._fed_rows = 0
        chunks = list(chunks)
        # PLI (PLI_MERGE_TAIL): the last chunk, short, starts at the checkpoint: it is read with the previous one
        if merge_at is not None and len(chunks) >= 2 and chunks[-1][0] == merge_at:
            chunks[-2:] = [(chunks[-2][0], chunks[-1][1])]
        ahead = getattr(self.model, "prefetch_prompt", None)
        if ahead is not None and chunks:
            ahead(tokens, chunks[0][0], chunks[min(reach, len(chunks)) - 1][1])
        n = 0
        while n < len(chunks):
            if n:
                yield                                         # between forwards: the scheduler may run decode rounds
            width = self._pass_width(chunks, n, cache) if together is not None else 1
            span = chunks[n:n + width]
            begin, end = span[0][0], span[-1][1]
            n += width
            if widths is not None:
                widths.append(width)
            if ahead is not None and n < len(chunks):
                ahead(tokens, chunks[n][0], chunks[min(n + reach, len(chunks)) - 1][1])   # read while this computes
            # PLI (item "chunk"): while another request decodes, a chunk of the plan is read in several passes
            # of PLI_CHUNK_TOKENS tokens at most, the engine yielding between two; the chunk grid
            # (resumes, kept checkpoints) does not change. Off (0): original behaviour.
            pieces = _pli_pieces(begin, end) if (width == 1 and prompt_data is None
                                                 and getattr(self, "active_count", 0) > 0) else None
            if pieces is not None:
                if whole is not None:
                    whole[0] = None
                for k, (pa, pb) in enumerate(pieces):
                    if k:
                        yield
                    rows = [int(t) for t in tokens[pa:pb]]
                    if self.prefill_guard is not None:
                        self.prefill_guard.before_chunk(cache, len(rows))
                    hidden = feed(mx.array([rows], dtype=mx.uint32), cache)
                    self._fed_rows = len(rows)
                    self.prefill_tokens += len(rows)
                    last = hidden[:, -1:, :]
                    drafting = getattr(self.model, "mtp", None) is not None
                    reads = drafting and _head_reads(self.model, pb, len(tokens))
                    absorb = self.model.absorb_draft_context if reads else getattr(self.model, "absorb_draft_taps", None)
                    if drafting and absorb is not None:
                        nxt = [int(t) for t in tokens[pa + 1:pb + 1]]
                        if nxt:
                            absorb(hidden[:, :len(nxt)], mx.array(nxt, dtype=mx.uint32), cache, start=0)
                            if reads and stream_id is not None and _falcon.active():     # PLI: FALCON, prompt rows read
                                _falcon.context(stream_id, pa, hidden[0, :len(nxt)], nxt, len(tokens))
                    final = pb == end and n == len(chunks)
                    reads_last = final or (reads and getattr(self.model, "draft_reads_hidden", True))
                    mx.eval(*((last,) if reads_last else ()), *cache_arrays(cache))
                    if self.prefill_guard is not None:
                        self.prefill_guard.after_chunk(cache, len(rows))
                self.prefill_chunks += width
                if raised is not None:
                    raised.append(False)
                if whole is not None:
                    whole[0] = end
                continue
            rows = [int(t) for t in tokens[begin:end]]
            if self.prefill_guard is not None:
                self.prefill_guard.before_chunk(cache, len(rows))
            if merge_at is not None and begin < merge_at < end:
                merge_tail.arm(cache, merge_at - begin)   # PLI_MERGE_TAIL: DeltaNet state captured at the checkpoint
            if whole is not None:
                whole[0] = None                        # a forward in flight: the cache holds no prompt prefix whole
            inputs = mx.array([rows], dtype=mx.uint32)
            sizes = [b - a for a, b in span]
            kept = self._raise_pass_cache(cache, sizes) if width > 1 else None
            if raised is not None:
                raised.append(kept is not None)
            try:
                if prompt_data is not None:
                    hidden = self.model.prefill_vision(inputs, cache, prompt_data, begin, end)
                elif width > 1:
                    hidden = together(inputs, cache, sizes)
                else:
                    # PLI (interleaved split): while another request decodes, the prefill of the whole chunk
                    # stops every PLI_CHUNK_LAYERS layers and yields; same computations, same order.
                    steps = getattr(self.model, "prefill_steps", None)
                    if feed is self.model.hidden:
                        steps = None
                    step = _pli_layers() if (steps is not None and getattr(self.model, "interleave_ok", False)
                                             and getattr(self, "active_count", 0) > 0) else 0
                    # PLI (split prefill): share of the chunk's rows whose MLP is done by the CPU; 0 = off.
                    share = getattr(self.model, "prefill_cpu_part", None) if steps is not None else None
                    if step or (share is not None and share()):
                        walk = steps(inputs, cache, step)
                        while True:
                            try:
                                next(walk)
                            except StopIteration as finished:
                                hidden = finished.value
                                break
                            yield
                    else:
                        hidden = feed(inputs, cache)
                _tc_t = time.perf_counter() if _tc.measuring() else None   # PLI_TRACE_LAYERS: end of the chunk (draft, evaluation)
                self._fed_rows = len(rows)
                self.prefill_chunks += width
                self.prefill_tokens += len(rows)
                last = hidden[:, -1:, :]
                drafting = getattr(self.model, "mtp", None) is not None
                taps = getattr(self.model, "absorb_draft_taps", None)
                reads = drafting and _head_reads(self.model, end, len(tokens))
                if drafting and (reads or taps is not None):
                    for a, b in span:                         # chunk by chunk, as one chunk a forward feeds the head
                        nxt = [int(t) for t in tokens[a + 1:b + 1]]
                        read = _head_reads(self.model, b, len(tokens))
                        absorb = self.model.absorb_draft_context if read else taps
                        if nxt and absorb is not None:
                            at = a - begin
                            absorb(hidden[:, at:at + len(nxt)], mx.array(nxt, dtype=mx.uint32), cache, start=at)
                            if read and stream_id is not None and _falcon.active():     # PLI: FALCON, prompt rows read
                                _falcon.context(stream_id, a, hidden[0, at:at + len(nxt)], nxt, len(tokens))
                # an earlier chunk is read only through its caches (and a taps head's taps): MLX skips its last layer
                reads_last = n == len(chunks) or (reads and getattr(self.model, "draft_reads_hidden", True))
                mx.eval(*((last,) if reads_last else ()), *cache_arrays(cache))
                if _tc_t is not None:
                    _tc.add("end", (time.perf_counter() - _tc_t) * 1e3)
            finally:
                if kept is not None:
                    mx.set_cache_limit(kept)
            if whole is not None:
                whole[0] = end
            if self.prefill_guard is not None:
                self.prefill_guard.after_chunk(cache, len(rows))
        return last

    def _raise_pass_cache(self, cache: list[Any], sizes: list[int]) -> int | None:
        """The pass-cache limit raised for one pass where the budget has room, or None when unchanged."""

        import mlx.core as mx

        old = int(mx.set_cache_limit(int(self.pass_cache)))
        guard = self.prefill_guard
        if old >= self.pass_cache or (guard is not None and not guard.pass_room(cache, sizes, self.pass_cache - old)):
            mx.set_cache_limit(old)
            return None
        return old

    def _pass_width(self, chunks: list[tuple[int, int]], n: int, cache: list[Any]) -> int:
        """How many chunks a forward takes: several only while no stream waits, and as many as fit."""

        guard = self.prefill_guard
        if guard is None and getattr(self, "active_count", 0):      # in process: live streams' rounds wait on it
            return 1
        small = int(getattr(self.model, "fused_rows", 16))
        sizes: list[int] = []
        for a, b in chunks[n:n + max(1, int(self.prefill_pass))]:
            if b - a <= small:
                break
            sizes.append(b - a)
        if len(sizes) < 2:
            return 1
        return max(1, min(len(sizes), guard.pass_width(cache, sizes))) if guard is not None else len(sizes)

    def _family_start(self, cache: list[Any] | None, cached_tokens: int, chunks: Any) -> tuple[list[Any], int]:
        """The working cache and where its prefill starts: a stored state only at one of the prompt's chunk starts."""

        if cache is None or (int(cached_tokens) not in chunks
                             and not (_KEEP_REPLY and 0 < int(cached_tokens) < chunks.length)):
            return self.model.make_cache(), 0
        adopt = getattr(self.model, "adopt_cache", None)
        work = adopt(cache) if adopt is not None else cache
        # PLI (23/10/2026): the kept choice of tiles (pli_choice) and the anchors of the reduced prefill (pli_anchors, pli_request_start,
        # pli_prefill_start) live on each layer's cache; resumed from a checkpoint, they made a request depend on the previous one.
        for layer in work:
            for name in ("pli_choice", "pli_anchors", "pli_request_start", "pli_prefill_start"):
                if getattr(layer, name, None) is not None:
                    setattr(layer, name, None)
            if getattr(layer, "pli_kidx", None) is not None:    # PLI (A1): LSA index keys of the resumed positions only
                from tensorfold.pli import a1_lsa as _a1

                _a1.resume(layer, int(cached_tokens))
        return work, int(cached_tokens)

    def _family_prefill(self, stream: Any, *, cache: list[Any] | None, cached_tokens: int,
                        checkpoints_at: Sequence[int]) -> list[Any]:
        return drain(self._family_prefill_steps(stream, cache=cache, cached_tokens=cached_tokens,
                                                checkpoints_at=checkpoints_at))

    def _family_prefill_steps(self, stream: Any, *, cache: list[Any] | None, cached_tokens: int,
                              checkpoints_at: Sequence[int]) -> Iterator[None]:
        """The prompt's prefill, yielding only between its chunks; the last chunk and the first token go together."""

        prompt = stream.prompt_ids
        if not prompt:
            raise ValueError(f"{stream.stream_id}: empty prompt")
        prepared = getattr(stream, "prompt_data", None)
        # an image prompt never resumes, so it needs no cut at message starts: its chunks are the step grid alone
        chunks = self.prompt_chunks(prompt) if prepared is None else PromptChunks(
            None, len(prompt), step=getattr(self.prefill_plan, "step", None) or self.prefill_step)
        work, start = self._family_start(cache, cached_tokens, chunks)
        for i_layer, layer in enumerate(work):   # PLI (23/10/2026, PLI_REDUCED_ANCHOR): reduced prefill applies only after the prompt
            try:
                if getattr(layer, "pli_layer", None) is None:        # PLI (05/11/2026): fresh cache (make_cache): layer number
                    layer.pli_layer = i_layer                        # (read by the reduced prefill and A1; it was missing at the first prefill)
                layer.pli_request_start = len(prompt)
                layer.pli_prefill_start = int(start)          # PLI (S3b, 02/11/2026): bound of the anchors when reading the prompt
                layer.pli_anchors = {}
            except Exception:
                pass
        if prepared is not None:
            if start or cache is not None:
                raise ValueError("image prompts require a fresh cache")
            prepared = self.model.encode_vision(prepared, work)
            checkpoints_at = ()
        cached_tokens = start
        if _falcon.active():
            _falcon.begin(stream.stream_id, start, prompt)          # PLI: FALCON, link to the previous request
        whole: list[Any] = [start]         # this prompt's own progress: other prompts' forwards run between its own
        stream.history_checkpoints = []
        stream.prefill_widths = []
        stream.prefill_raised = []
        from tensorfold.pli import merge_tail

        bounds = sorted({chunks.floor(int(b)) for b in checkpoints_at})
        merged = None
        if merge_tail.active() and prepared is None and bounds and start < bounds[-1] < len(prompt) \
                and len(prompt) - bounds[-1] <= merge_tail.TAIL:
            # PLI_MERGE_TAIL: the last checkpoint is read with the chunk before it, if this chunk exists
            # in the grid and the two fit in an ordinary chunk (2,048 rows: Neural Engine included)
            before = max(bounds[-2], start) if len(bounds) >= 2 else start
            grid_end = chunks.between(before, len(prompt))
            if (len(grid_end) >= 2 and grid_end[-1][0] == bounds[-1]
                    and grid_end[-1][1] - grid_end[-2][0] <= int(getattr(chunks, "step", 2048))):
                merged = bounds.pop()
        try:
            fed = False
            for boundary in bounds:
                if not start < boundary < len(prompt):
                    continue
                if fed:
                    yield
                yield from self._family_feed_steps(prompt, work, chunks.between(start, boundary), wide=True,
                                                   widths=stream.prefill_widths, raised=stream.prefill_raised,
                                                   whole=whole, stream_id=stream.stream_id)
                fed = True
                if self.prefill_guard is None or self.prefill_guard.allow_checkpoint(work):
                    from tensorfold.pli import kv_sharing as _kvp       # PLI: KV layers by length

                    stream.history_checkpoints.append((list(prompt[:boundary]),
                                                       _kvp.view(work, self.copy_single_cache) if _kvp.ACTIVE
                                                       else drop_spares(self.copy_single_cache(work))))
                else:
                    self.prefill_guard.refuse(boundary, work)        # logged where it refuses (issue #155)
                start = boundary
            if fed:
                yield
            hidden = yield from self._family_feed_steps(prompt, work, chunks.between(start, len(prompt)), prepared,
                                                        wide=True, widths=stream.prefill_widths,
                                                        raised=stream.prefill_raised, whole=whole, merge_at=merged,
                                                        stream_id=stream.stream_id)
            if merged is not None:
                captures = merge_tail.captured(work)
                if captures and (self.prefill_guard is None or self.prefill_guard.allow_checkpoint(work)):
                    copy = self.copy_single_cache(work)
                    if merge_tail.snapshot(copy, captures, len(prompt), merged):
                        from tensorfold.pli import kv_sharing as _kvp   # PLI

                        if _kvp.ACTIVE:
                            copy = _kvp.lighten(copy, work)
                        stream.history_checkpoints.append((list(prompt[:merged]), drop_spares(copy)))
                    else:
                        print(f"[PLI] merge_tail: checkpoint at {merged} not kept (cache not brought back)", flush=True)
                elif not captures:
                    print(f"[PLI] merge_tail: checkpoint at {merged} not kept (no state captured)", flush=True)
        except BaseException:
            if merged is not None:
                merge_tail.captured(work)
            at = whole[0]                              # stopped between chunks: keep the progress, a taken prefix too
            kept = [len(tokens) for tokens, _ in stream.history_checkpoints]
            if prepared is None and at is not None and at in chunks and at not in kept:
                from tensorfold.pli import kv_sharing as _kvp           # PLI

                stream.history_checkpoints.append((list(prompt[:at]), _kvp.view(work, self.copy_single_cache) if _kvp.ACTIVE
                                                   else drop_spares(self.copy_single_cache(work))))
            raise
        if getattr(stream, "label_ids", ()):                 # a decision: the last row, then no round
            self._family_score(stream, hidden, cached_tokens)
            return work
        first = self._family_first(stream, work, hidden, cached_tokens, self._fed_rows - 1)
        self._family_commit_first(stream, int(first.item()) if hasattr(first, "item") else int(first))
        return work

    def _family_first(self, stream: Any, work: list[Any], hidden: Any, cached_tokens: int, row: int) -> Any:
        """Draw or force the first token before the draft head or the next forward reads it."""

        import mlx.core as mx

        prompt_len = len(stream.prompt_ids)
        stream.emitted = []
        stream.pending = []
        stream.cache_len = prompt_len
        stream.cached_tokens = int(cached_tokens)
        stream.started_at = time.perf_counter()
        logits = self.model.head(hidden)
        if stream.constraint is not None:                # the first token under the reply's grammar
            logits = stream.constraint.mask(logits)
        token = self._draw(logits, stream.sampling, [prompt_len])
        forced = self._forced_next(stream, token)
        if forced is not None:
            token = mx.array([forced], dtype=mx.uint32)
        if self.family_mtp and stream.drafts:
            if _falcon.active():                 # PLI: FALCON, the prompt's last row, with its target (1st token)
                _falcon.context(stream.stream_id, prompt_len - 1, hidden.reshape(1, -1), token.reshape(-1),
                                 prompt_len + 1, logits.reshape(1, -1))
            # the head reads the prompt's last position and the first token, and drafts the one after it
            firsts = self.model.speculate(work, token, prompt_len - 1, stream.sampling, start=row)
            # the first drafts settle at the next round, so the first token goes out without the draft forward
            self._next[stream.stream_id] = partial(self.model.settle, work, 1, firsts.reshape(-1)[:1],
                                                   prompt_len + 1, stream.sampling, self._depth(stream))
        elif self.pipelined and stream.constraint is None:      # a grammar reads each token before the next
            self._queue_next(stream, work, token)
        return token

    @staticmethod
    def _family_commit_first(stream: Any, first: int) -> None:
        stream.commit([first])
        stream.pending = [first]

    def _queue_next(self, stream: Any, cache: list[Any], token: Any) -> None:
        """Feed ``token`` (a GPU array, not read yet) and queue the draw of the one after it."""

        import mlx.core as mx

        hidden = self.model.hidden(token.reshape(1, 1), cache)
        stream.cache_len += 1
        nxt = self._draw(self.model.head(hidden), stream.sampling, [stream.cache_len])
        mx.async_eval(nxt)
        self._inflight[stream.stream_id] = nxt

    def _family_prefill_prefix(self, prompt_ids: Sequence[int], *, cache: list[Any] | None,
                               cached_tokens: int) -> list[Any]:
        if not prompt_ids:
            raise ValueError("empty prefix")
        chunks = self.prompt_chunks(prompt_ids)
        work, start = self._family_start(cache, cached_tokens, chunks)
        self._family_feed(prompt_ids, work, chunks.between(start, len(prompt_ids)))
        return drop_spares(work)

    def _family_score(self, stream: Any, hidden: Any, cached_tokens: int) -> None:
        """Read the decision's last row, draw nothing, and keep the stream out of the rounds."""

        prompt_len = len(stream.prompt_ids)
        stream.emitted = []
        stream.pending = []
        stream.cache_len = prompt_len
        stream.cached_tokens = int(cached_tokens)
        stream.started_at = time.perf_counter()
        stream.scored = self._label_logits(hidden, stream.label_ids)
        stream.finished = True
        stream.finish_reason = "decision"

    def _label_logits(self, hidden: Any, label_ids: Sequence[int]) -> tuple[list[float], float]:
        """Last-row logits of ``label_ids`` and the full-vocabulary logsumexp."""

        import math

        import mlx.core as mx

        logits = self.model.head(hidden)
        row = logits.reshape(-1, logits.shape[-1])[-1].astype(mx.float32)
        picked = row[mx.array([int(token) for token in label_ids], dtype=mx.int32)]
        peak = mx.max(row)
        logsumexp = peak + mx.log(mx.sum(mx.exp(row - peak)))
        mx.eval(picked, logsumexp)
        values = [float(item) for item in picked.tolist()]
        total = float(logsumexp.item())
        if not math.isfinite(total) or any(not math.isfinite(value) for value in values):
            raise ValueError("label scoring produced a non-finite logit")
        return values, total

    def score_labels(self, prompt_ids: Sequence[int], label_ids: Sequence[int]) -> tuple[list[float], float]:
        """Last-position logits of ``label_ids`` and the full-vocabulary logsumexp. No token is sampled."""

        prompt = [int(token) for token in prompt_ids]
        labels = [int(token) for token in label_ids]
        if not prompt:
            raise ValueError("empty prompt")
        if not labels:
            raise ValueError("empty labels")
        chunks = self.prompt_chunks(prompt)
        work, start = self._family_start(None, 0, chunks)
        try:
            hidden = self._family_feed(prompt, work, chunks.between(start, len(prompt)))
            return self._label_logits(hidden, labels)
        finally:
            del work

    def _family_add_stream(self, stream: Any, *, cache: list[Any] | None, cached_tokens: int,
                           checkpoints_at: Sequence[int]) -> Iterator[None]:
        """Prefill a stream a chunk a step; after the last chunk it takes part in the rounds."""

        work = yield from self._family_prefill_steps(stream, cache=cache, cached_tokens=cached_tokens,
                                                     checkpoints_at=checkpoints_at)
        self.streams.append(stream)
        if stream.finished:
            stream.finished_at = time.perf_counter()
            self._release_stream_state(stream.stream_id)
            return
        self._live.append((stream, work))
