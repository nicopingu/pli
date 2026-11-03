"""Share Qwen3.8 forwards across streams while preserving each stream's serial bits, sampling rule and accepted path."""

from __future__ import annotations

import os

import time
from typing import Any, Sequence

from tensorfold.families.qwen3_5.dflash_head import DFlashHead, DraftSlot


def _identity(x: Any) -> Any:
    return x


def _tokens(window: Any) -> list[int]:
    return [int(t) for t in (window.reshape(-1).tolist() if hasattr(window, "tolist") else window)]


def _chain(rows: int) -> list[int]:
    return [-1, *range(rows - 1)]


class Qwen35Family:
    """Qwen3.8 dense on the lane kernels, as a family model."""

    lane_family = True
    draft_reads_hidden = False      # the DFlash head drafts from the tapped layers, never the last hidden state
    speculate_early = False            # DFlash2 reads the kept rows' taps after a round is read
    batch_rows = 128                   # a shared forward's rows (the lane matmul's limit; the costs decide how many)
    max_streams = 64
    draft_prior = (0.78, 0.74, 0.71, 0.73, 0.71, 0.75, 0.78, 0.75, 0.75, 0.75, 0.75, 0.7, 0.7, 0.7, 0.7)

    def __init__(self, model: Any, *, drafter: Any = None, nodes: int = 15, widest: int = 32,
                 rows: bool = False, first_copy_rows: int | None = None) -> None:
        if os.environ.get("PLI_TREE_NODES"):      # PLI (D2): nodes at most of a draft tree (15 by default)
            nodes = int(os.environ["PLI_TREE_NODES"])
        self.inner = model
        language_model = getattr(model, "language_model", model)
        self.core = language_model.model
        args = getattr(language_model, "args", None)
        tied = args is not None and getattr(args, "tie_word_embeddings", False)
        self.lm_head = self.core.embed_tokens.as_linear if tied else language_model.lm_head
        self.rows = bool(rows)         # the decoder without tensor units (``row_forward``, installed)
        trees = True
        if self.rows:
            from tensorfold.kernels.qwen.dense.v1 import row_forward

            trees = row_forward.ROW_ATTENTION or row_forward.TREES_ATTM     # PLI (D2): trees through attm
            self.batch_rows, self.max_streams = 32, 32
        self.head_drafts = DFlashHead(drafter, nodes, _calibration(), chains=not trees) if drafter is not None else None
        if self.head_drafts is not None and self.head_drafts.v1:
            # DFlash (v1) gives no per-draft chances: the engine sizes its chains from per-depth acceptance and
            # measured round times (``DraftDepth._depth``), not from the forward's cost alone
            self.draft_probabilities = None
        self.mtp = drafter
        self.drafts = int(nodes) if drafter is not None else 0
        self.mtp_step_ms = 0.0         # a lattice costs the same whatever the tree's size
        self._last: dict[int, tuple[Any, int, int, int]] = {}   # Cache id -> last forward record, row count, start and first row within the shared forward.
        self._shared: list[tuple[Any, int, int]] = []         # the last hidden_rows: (record, rows, start) a stream
        self.exact_width, self.window_costs = self.check_windows(int(widest), int(self.batch_rows))
        # a copy's first window; each copy that lands whole doubles the next, up to exact_width rows
        self.first_copy_rows = min(int(first_copy_rows or widest), self.exact_width)

    def make_cache(self) -> list[Any]:
        caches = list(self.inner.make_cache())
        if self.head_drafts is not None:
            caches.append(self.head_drafts.slot())
        return caches

    def adopt_cache(self, cache: list[Any]) -> list[Any]:
        if self.head_drafts is not None and not (cache and isinstance(cache[-1], DraftSlot)):
            cache.append(self.head_drafts.slot())
        return cache

    @staticmethod
    def _layers(cache: list[Any]) -> list[Any]:
        return cache[:-1] if cache and isinstance(cache[-1], DraftSlot) else cache

    @staticmethod
    def _number_layers(layers: list[Any]) -> None:
        """PLI: each layer cache carries its number (read by the attention's reduced prefill, exact_attention)."""

        if layers and getattr(layers[-1], "pli_layer", None) is None:
            for i, item in enumerate(layers):
                try:
                    item.pli_layer = i
                except Exception:
                    pass

    @staticmethod
    def _position(layers: list[Any]) -> int:
        return next(int(item.offset) for item in layers if hasattr(item, "keys") and hasattr(item, "offset"))

    def hidden(self, inputs: Any, cache: list[Any], parents: Sequence[int] | None = None) -> Any:
        """Return hidden states [1, R, D]; chains advance all rows, while trees wait for keep_rows to commit the accepted path."""

        tokens, layers = _tokens(inputs), self._layers(cache)
        self._number_layers(layers)
        start, rows = self._position(layers), len(tokens)
        tree = list(parents) if parents is not None else _chain(rows)
        if self.rows:
            from tensorfold.kernels.qwen.dense.v1 import row_forward

            hidden, records = row_forward.hidden_rows(self.core, [tokens], [layers], starts=[start], parents=[tree])
            record = records[0]
        else:
            from tensorfold.kernels.qwen.dense.v1 import lane_tree

            hidden, record = lane_tree.tree_forward(self.core, _identity, tokens, tree, layers, start)
        if parents is None:
            self._commit(layers, record, list(range(rows)), rows, start)
        self._last = {id(cache): (record, rows, start, 0)}
        return hidden

    def prefill(self, inputs: Any, cache: list[Any]) -> Any:
        """A prompt chunk through MLX's forward: a fresh prompt and a resumed one get the same chunks and bits."""

        import mlx.core as mx

        self._unpack()
        tokens, layers = _tokens(inputs), self._layers(cache)
        self._last = {id(cache): (None, len(tokens), self._position(layers), 0)}
        from tensorfold.pli import trace_layers as tc   # PLI (PLI_TRACE_LAYERS): trace under the chunk, evolution mode

        if tc.ACTIVE:
            tc.begin(len(tokens), self._position(layers))
        with tc.prefill():
            return self.core(mx.array([tokens], dtype=mx.uint32), cache=layers)

    _unpack_done = False

    def _unpack(self) -> None:
        """PLI (PLI_PREFILL_UNPACK): weights unpacked before the product on the large prefill calls; set up once."""

        if self._unpack_done:
            return
        self._unpack_done = True
        from tensorfold.pli import unpack

        if unpack.active():
            unpack.install(getattr(self.core, "pipeline_layers", None) or getattr(self.core, "layers", None))
            from tensorfold.pli.ane import ffn as ane_ffn   # PLI SP1 (PLI_ANE_FFN): FFN shared with the Neural Engine in prefill

            ane_ffn.install(getattr(self.core, "pipeline_layers", None) or getattr(self.core, "layers", None))
            head = getattr(self, "head_drafts", None)
            unpack.install_draft(getattr(head, "drafter", None) if head is not None else None)   # PLI_PREFILL_UNPACK_NN
        from tensorfold.pli import trace_layers as tc   # PLI (PLI_TRACE_LAYERS): set up after the Neural Engine split

        tc.install(self.core)
        from tensorfold.pli import merge_tail   # PLI (PLI_MERGE_TAIL): final chunk read with the previous one

        merge_tail.install(getattr(self.core, "pipeline_layers", None) or getattr(self.core, "layers", None))

    # PLI: interleaved split of the prefill on the dense model (27B), off until it is measured on the 27B
    interleave_ok = __import__("os").environ.get("PLI_INTERLEAVE_DENSE", "0") == "1"
    prefill_cpu_ok = True      # PLI: dense MLP, the prefill may give rows to the CPU (PLI_PREFILL_CPU)
    _prefill_cpu: Any = None

    def prefill_cpu_part(self) -> float:
        """PLI (split prefill): share of a chunk's rows given to the CPU; 0 = off or model not handled."""

        from tensorfold.families.qwen3_5 import prefill_cpu

        p = prefill_cpu.part() if self.prefill_cpu_ok else 0.0
        if not p:
            return 0.0
        if self._prefill_cpu is None:
            self._prefill_cpu = prefill_cpu.PrefillCPU()
        layers = getattr(self.core, "pipeline_layers", None) or []
        return p if layers and self._prefill_cpu.weights(layers[0]) is not None else 0.0

    def _draft_captures(self) -> Any:
        """The list of layer states captured for the draft (one slot per captured layer), or None."""

        draft = getattr(self.head_drafts, "drafter", None) if getattr(self, "head_drafts", None) is not None else None
        return getattr(getattr(draft, "target", None), "_hidden_states", None)

    def prefill_steps(self, inputs: Any, cache: list[Any], step: int) -> Any:
        """PLI (interleaved split): ``prefill`` in steps of ``step`` layers; yields between two steps.

        Same operations, same order as MLX's pass: only the moment of evaluation changes. Final value: the
        hidden states, like ``prefill``. If the model does not have the expected shape, a single step per ``prefill``.
        """

        import sys

        import mlx.core as mx

        from tensorfold.engine.family_common import cache_arrays

        self._unpack()
        core = self.core
        mod = sys.modules.get(type(core).__module__)
        fa_mask = getattr(mod, "create_attention_mask", None)
        ssm_mask = getattr(mod, "create_ssm_mask", None)
        stack = getattr(core, "pipeline_layers", None)
        part = self.prefill_cpu_part()
        if ((step < 1 and not part) or stack is None or fa_mask is None or ssm_mask is None
                or getattr(core, "pipeline_size", 1) != 1 or getattr(core, "pipeline_rank", 0) != 0
                or not hasattr(core, "fa_idx") or not hasattr(core, "ssm_idx")):
            return self.prefill(inputs, cache)
        tokens, layers = _tokens(inputs), self._layers(cache)
        if len(layers) != len(stack):
            return self.prefill(inputs, cache)
        rows, start = len(tokens), self._position(layers)
        from tensorfold.pli import trace_layers as tc   # PLI (PLI_TRACE_LAYERS): trace under the chunk, evolution mode

        if tc.ACTIVE:
            tc.begin(rows, start)
        with tc.prefill():
            h = core.embed_tokens(mx.array([tokens], dtype=mx.uint32))
        fa = fa_mask(h, layers[core.fa_idx]) if core.fa_idx is not None else None
        ssm = ssm_mask(h, layers[core.ssm_idx]) if core.ssm_idx is not None else None
        total = len(stack)
        if part:
            from tensorfold.families.qwen3_5 import prefill_cpu

            lc = self._prefill_cpu.cpu_rows(stack, h, part) if prefill_cpu.mode() == "shifted" else 0
            if lc:
                h = yield from self._prefill_cpu.shifted_pass(stack, layers, h, fa_mask, ssm_mask, lc, step,
                                                              self._draft_captures())
                self._last = {id(cache): (None, rows, start, 0)}
                return core.norm(h)
        for k, (layer, c) in enumerate(zip(stack, layers)):
            if part:
                h = self._prefill_cpu.layer(layer, h, ssm if layer.is_linear else fa, c, part)
            else:
                with tc.prefill():
                    h = layer(h, mask=ssm if layer.is_linear else fa, cache=c)
            done = k + 1
            if step > 0 and done % step == 0 and done < total:
                # The layer states the draft reads (DFlash2) are a single list for the whole model: an interleaved
                # decode round rewrites it. Those of this prefill are kept and put back at the resume (measured on the 27B).
                states = self._draft_captures()
                kept = list(states) if states else None
                mx.eval(h, *cache_arrays(layers[done - step:done]), *[a for a in (kept or []) if a is not None])
                yield
                if kept is not None:
                    states[:] = kept
        self._last = {id(cache): (None, rows, start, 0)}       # set at the end: interleaved decodes overwrite it
        with tc.prefill():
            return core.norm(h)

    def encode_vision(self, prepared: Any, cache: list[Any]) -> Any:
        encoded = self.vision.encode(prepared)
        for item in self._layers(cache):
            if hasattr(item, "keys"):
                item.vision_rope_delta = int(encoded.rope_delta)
        return encoded

    def prefill_vision(self, inputs: Any, cache: list[Any], encoded: Any, begin: int, end: int) -> Any:
        from tensorfold.vision.rotary import vision_positions

        layers = self._layers(cache)
        self._last = {id(cache): (None, end - begin, self._position(layers), 0)}
        with vision_positions(encoded.position_ids[:, :, begin:end]):
            return self.core(inputs, cache=layers, input_embeddings=encoded.inputs_embeds[:, begin:end])

    def _commit(self, layers: list[Any], record: Any, path: list[int], rows: int, start: int) -> None:
        from tensorfold.kernels.qwen.dense.v1 import exact_attention as _ea      # PLI (item D): anchored choices of the validated nodes
        if _ea.ANCHOR:
            _ea.commit_anchors(layers, path)
        if self.rows:
            from tensorfold.kernels.qwen.dense.v1 import row_forward

            row_forward.commit(layers, record, path, rows, start)
        else:
            from tensorfold.kernels.qwen.dense.v1 import lane_tree

            lane_tree.commit_tree(layers, record, path, rows, start)

    def keep_rows(self, cache: list[Any], rows: int, keep: Any) -> None:
        """Commit a prefix or accepted path from the last forward record, rolling back even a chain previously committed whole."""

        record, width, start, _ = self._last.pop(id(cache))
        path = list(range(keep)) if isinstance(keep, int) else [int(r) for r in keep]
        self._commit(self._layers(cache), record, path, width, start)

    def hidden_rows(self, windows: Sequence[Any], caches: Sequence[list[Any]],
                    parents: Sequence[Sequence[int]] | None = None) -> Any:
        """Return hidden states [1, N, D] grouped by stream; chains advance caches, while trees wait for keep_rows_streams."""

        tokens = [_tokens(w) for w in windows]
        layers = [self._layers(c) for c in caches]
        starts = [self._position(c) for c in layers]
        trees = [list(p) for p in parents] if parents is not None else [_chain(len(t)) for t in tokens]
        if self.rows:
            from tensorfold.kernels.qwen.dense.v1 import row_forward

            hidden, records = row_forward.hidden_rows(self.core, tokens, layers, starts=starts, parents=trees)
        else:
            from tensorfold.kernels.qwen.dense.v1 import lane_multi

            hidden, records, _ = lane_multi.multi_tree_forward(self.core, _identity, tokens, trees, layers, starts)
        lengths = [len(t) for t in tokens]
        self._shared = list(zip(records, lengths, starts))
        firsts = [sum(lengths[:i]) for i in range(len(lengths))]
        self._last = {id(c): (r, n, st, f) for c, r, n, st, f in zip(caches, records, lengths, starts, firsts)}
        if parents is None:
            self._commit_streams(layers, records, [list(range(n)) for n in lengths], lengths, starts)
        return hidden

    def keep_rows_streams(self, caches: Sequence[list[Any]], lengths: Sequence[int], keeps: Sequence[Any]) -> None:
        shared, self._shared = self._shared, []
        paths = [list(range(k)) if isinstance(k, int) else [int(r) for r in k] for k in keeps]
        self._commit_streams([self._layers(c) for c in caches], [r for r, _, _ in shared], paths,
                             [n for _, n, _ in shared], [start for _, _, start in shared])
        self._last.clear()

    def _commit_streams(self, layers: list[Any], records: list[Any], paths: list[list[int]], widths: list[int],
                        starts: list[int]) -> None:
        if self.rows:
            for cache, record, path, width, start in zip(layers, records, paths, widths, starts):
                self._commit(cache, record, path, width, start)
            return
        from tensorfold.kernels.qwen.dense.v1 import lane_multi

        lane_multi.commit_streams(layers, records, paths, widths, starts)   # Commit all streams in one launch per layer.

    def head(self, hidden: Any) -> Any:
        if self.rows:
            from tensorfold.kernels.qwen.dense.v1 import row_matmul

            return row_matmul.logits(self.lm_head, hidden)          # the row-exact matmul, as the decoder's steps
        return self.lm_head(hidden)

    def sample(self, logits: Any, sampling: Any, positions: Sequence[int]) -> Any:
        """The keyed rule as the one-request path draws it (``exact_sampling``): a stream equals its run alone."""

        import mlx.core as mx

        from tensorfold.engine.exact_sampling import sample_rows, top_candidates

        logits = logits.reshape(-1, logits.shape[-1])
        if sampling is None:
            return mx.argmax(logits, axis=-1).astype(mx.uint32)
        return sample_rows(logits, [int(p) for p in positions], sampling, top=top_candidates(logits, sampling))

    def sample_streams(self, logits: Sequence[Any], samplings: Sequence[Any], positions: Sequence[Sequence[int]]
                       ) -> list[Any]:
        """Read sampled candidates and greedy argmax results together for every stream in the shared round."""

        import mlx.core as mx

        from tensorfold.engine.exact_sampling import sample_rows, top_candidates

        rows = [x.reshape(-1, x.shape[-1]) for x in logits]
        tops = [None if s is None else top_candidates(r, s) for r, s in zip(rows, samplings)]
        greedy = [mx.argmax(r, axis=-1).astype(mx.uint32) if s is None else None for r, s in zip(rows, samplings)]
        mx.eval(*[a for t in tops if t is not None for a in t], *[g for g in greedy if g is not None])
        return [g if s is None else sample_rows(r, [int(p) for p in at], s, top=t)
                for r, s, at, t, g in zip(rows, samplings, positions, tops, greedy)]

    def absorb_draft_context(self, hidden: Any, next_tokens: Any, cache: list[Any], start: int = 0) -> None:
        _, _, position, first = self._last[id(cache)]
        self.head_drafts.absorb(cache, position + start, int(hidden.shape[1]), first + start)

    def speculate(self, cache: list[Any], tokens: Any, position: int, sampling: Any, start: int = 0,
                  last_only: bool = False, rows: Sequence[int] | None = None) -> Any:
        """Read the kept rows (``rows``, or ``start`` ..) of the last forward; ``settle`` drafts."""

        import mlx.core as mx

        follow = _tokens(tokens)
        kept = list(rows) if rows is not None else list(range(start, start + len(follow)))
        self.head_drafts.read(cache, kept, follow, sampling)
        return mx.array([0], dtype=mx.uint32)

    def settle(self, cache: list[Any], keep: int, first: Any, position: int, sampling: Any, count: int) -> Any:
        return self.head_drafts.tree(cache, position, sampling, count) if count > 0 else []

    def unspeculate(self, cache: list[Any]) -> None:
        pass

    def draft_streams(self, caches: Sequence[list[Any]], follows: Sequence[Sequence[int]], rows: Sequence[Sequence[int]],
                      positions: Sequence[int], samplings: Sequence[Any], depths: Sequence[int]) -> list[Any]:
        return self.head_drafts.draft_streams(caches, follows, rows, positions, samplings, depths)

    def draft_probabilities(self, cache: list[Any]) -> list[float] | None:
        """The chance each of the stream's queued drafts lands (``engine.allocate`` splits a round's rows by them)."""

        return self.head_drafts.probabilities(cache) if self.head_drafts is not None else None

    def check_windows(self, widest: int, timed: int) -> tuple[int, dict[int, float]]:
        """Find the widest bit-exact chain and per-width forward costs, sampling tile boundaries and interpolating other widths."""

        import mlx.core as mx

        from tensorfold.engine.lane_engine import LaneEngine
        from tensorfold.engine.family_common import cache_arrays

        timed = max(int(timed), int(widest))
        vocab = int(self.core.embed_tokens["weight"].shape[0])   # ids past the table read whatever memory follows it
        prompt, tokens = [t % vocab for t in range(1000, 1064)], [t % vocab for t in range(2000, 2000 + timed)]
        base = self.make_cache()
        self.hidden(mx.array([prompt]), base)
        mx.eval(*cache_arrays(self._layers(base)))

        def primed() -> list[Any]:
            cache = LaneEngine.copy_single_cache(base)
            mx.eval(*cache_arrays(self._layers(cache)))
            return cache

        step_cache, steps = primed(), []
        for t in tokens[:widest]:
            steps.append(self.head(self.hidden(mx.array([[t]]), step_cache))[0, -1])
        mx.eval(*steps)
        exact, costs = 1, {}
        # the row tiles step every 8 rows without tensor units (9, 17, 25, 33), and at 17, 33 and 65 with them
        steps_at = ((1, 2, 4, 8, 9, 12, 16, 17, 24, 25, 32, 33, 48, 64, 65, 96) if self.rows
                    else (1, 2, 4, 8, 12, 16, 17, 24, 32, 33, 48, 64, 65, 96))
        for width in sorted({w for w in steps_at if w <= timed} | {widest, timed}):
            best = float("inf")
            for rep in range(3):
                cache = primed()
                began = time.perf_counter()
                logits = self.head(self.hidden(mx.array([tokens[:width]]), cache))
                mx.eval(logits)
                best = min(best, (time.perf_counter() - began) * 1e3)
                if rep == 0 and 1 < width <= widest and exact == max([1, *(w for w in costs if w < width)]):
                    if all(bool(mx.array_equal(logits[0, i], steps[i]).item()) for i in range(width)):
                        exact = width
            costs[width] = best
        return exact, fill_widths(costs, timed)


def _calibration() -> dict[str, Any]:
    """Load DFlash2 node calibration for this target."""

    from pathlib import Path

    from tensorfold.drafters import calibration

    path = Path(__file__).with_name("dflash2_calibration.json")
    return calibration.load(path) if path.exists() else {}


def fill_widths(costs: dict[int, float], widest: int) -> dict[int, float]:
    """Interpolate costs between timed widths, using the nearest timed width outside their range."""

    timed = sorted(costs)
    out = dict(costs)
    for width in range(1, widest + 1):
        if width in out:
            continue
        below, above = [w for w in timed if w < width], [w for w in timed if w > width]
        if below and above:
            a, b = below[-1], above[0]
            out[width] = costs[a] + (costs[b] - costs[a]) * (width - a) / (b - a)
        else:
            out[width] = costs[above[0] if above else below[-1]]
    return dict(sorted(out.items()))


__all__ = ["Qwen35Family", "fill_widths"]
