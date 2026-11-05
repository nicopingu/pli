"""Stored prompt prefixes, their in-memory caches, and conversations saved at shutdown."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import threading
import time
from typing import Any, Callable
from tensorfold.pli import events as _ev

# PLI (28/10/2026, PLI_CHECKPOINT_LOG=1, off by default): each checkpoint kept, evicted or refused writes an event
# (PLI_EVENTS log), to find why an agent conversation re-reads everything.
_LOG = __import__("os").environ.get("PLI_CHECKPOINT_LOG", "0") == "1"


def _point(action: str, entry: "CheckpointEntry", **extra: Any) -> None:
    if not _LOG:
        return
    try:
        reply_end = entry.tokens != entry.last_prompt[:len(entry.tokens)]
        _ev.ev(action, conv=_ev.conv_of_tokens(entry.tokens), tokens=len(entry.tokens), gib=round(entry.nbytes / 1024**3, 3),
               kind="reply_end" if reply_end else ("block" if entry.pinned else "prompt"), **extra)
    except Exception:  # noqa: BLE001 - a log never breaks a request
        pass


def longest_common_prefix(a: list[int], b: list[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def extends(longer: list[int], shorter: list[int]) -> bool:
    """Whether ``longer`` continues ``shorter`` (its last token checked first: other conversations fail at once)."""

    n = len(shorter)
    return len(longer) > n > 0 and longer[n - 1] == shorter[-1] and longer[:n] == shorter


# PLI (rule of TensorFold PR #1291 for issue #1262, rewritten in Python): when the prompt forks from the previous one inside
# its last message (the same long document sent again with another question), the state at the fork point is kept: the next
# request of the same kind starts from there instead of reading the document again. Only past PLI_FORK_MIN tokens read since the
# resume (4,096); 0: off. Same text (a resume gives what a full prefill gives).
_FORK_MIN = int(os.environ.get("PLI_FORK_MIN", "4096") or 0)
_SPILL_PROMPT = os.environ.get("PLI_SPILL_PROMPT", "1") != "0"
# PLI (03/11/2026, public suite pli-bench-v1, agent session that does not keep the measured question):
# PLI_LAST_MESSAGE_CHECKPOINT=N also keeps the state at the start of the prompt's last message (the token that opens the
# reply, such as <|im_start|>, looked for backwards in the history) when at least N tokens are read there since the
# resume. The next request that keeps the history but not this last message (a question replaced, a reply thrown away,
# a sub-agent on the same history) starts from there instead of from the previous request's checkpoint. 0 (default):
# off. Same text (a resume gives what a full prefill gives).
_LAST_MESSAGE_MIN = int(os.environ.get("PLI_LAST_MESSAGE_CHECKPOINT", "0") or 0)


def last_message_start(history_len: int, prompt: list[int]) -> int:
    """Where the last message of the prompt starts: the last token before ``history_len`` equal to the one that opens the reply
    (``prompt[history_len]``, the turn marker of templates such as ChatML); 0 when there is none."""

    h = int(history_len)
    if not 0 < h < len(prompt):
        return 0
    marker = prompt[h]
    for i in range(h - 1, 0, -1):
        if prompt[i] == marker:
            return i
    return 0


def choose_checkpoints(
    history_len: int, cached: int, last_prompt: list[int] | None, prompt: list[int]
) -> list[int]:
    """Keep history and stable-prefix boundaries outside the reused prefix and before the prompt end for differently rendered next turns."""

    candidates = {int(history_len)}
    if last_prompt:
        stable = longest_common_prefix(last_prompt, prompt)
        if 0 < stable < int(history_len) and stable >= int(history_len) // 2:
            candidates.add(stable)
        elif _FORK_MIN > 0 and int(history_len) < stable < len(prompt) and stable - int(cached) >= _FORK_MIN:
            candidates.add(stable)
    if _LAST_MESSAGE_MIN > 0:
        start = last_message_start(history_len, prompt)
        if start - int(cached) >= _LAST_MESSAGE_MIN:
            candidates.add(start)
    return sorted(at for at in candidates if int(cached) < at < len(prompt))


@dataclass
class CheckpointEntry:
    tokens: list[int]
    cache: list[Any]
    last_prompt: list[int]
    nbytes: int = 0
    # A system block (loaded from disk, or saved to it): outside the slot count.
    pinned: bool = False
    born: int = 0              # the length of the prompt that stored it: a later turn's is longer


def save_conversations(store: "CheckpointStore", directory: Path, model_id: str, *, keep: int = 2,
                       limit_bytes: int = 10 * 1024**3) -> int:
    """Save up to ``keep`` conversation snapshots within the byte limit; system blocks use their own directory."""

    from tensorfold.engine.prefix_snapshots import save_snapshot
    from tensorfold.pli import checkpoint_bits as _cb

    with store._lock:
        entries = [entry for entry in store._entries if not entry.pinned]   # most recently used first
    # prompt-side entries before reply ends (a re-rendered reply never matches one), then the longest first
    def reply_end(entry: CheckpointEntry) -> bool:
        return entry.tokens != entry.last_prompt[:len(entry.tokens)]

    entries.sort(key=lambda entry: (reply_end(entry), -len(entry.tokens)))
    saved = total = 0
    for entry in entries:
        if saved >= keep or total + entry.nbytes > limit_bytes:
            break
        started = time.perf_counter()
        try:
            save_snapshot(directory, model_id, entry.tokens, _cb.for_disk(entry.cache), keep=keep)
        except Exception as exc:  # noqa: BLE001 - a full disk must not hang the shutdown
            print(f"[lanes] conversation save failed: {type(exc).__name__}: {exc}", flush=True)
            break
        saved += 1
        total += entry.nbytes
        _ev.ev("checkpoint", conv=_ev.conv_of_tokens(entry.tokens), tokens=len(entry.tokens),
               gib=entry.nbytes / 1024**3, duration=time.perf_counter() - started)
        print(f"[lanes] saved conversation checkpoint tokens={len(entry.tokens)} "
              f"({entry.nbytes / 1024**3:.1f} GiB) in {time.perf_counter() - started:.1f}s", flush=True)
    return saved


def spill_conversation(entry: CheckpointEntry, directory: Path, model_id: str, *, limit_bytes: int) -> bool:
    """Write an evicted conversation where ``_read_disk_block`` finds it, this model's files kept under ``limit_bytes``."""

    from tensorfold.engine.prefix_snapshots import save_snapshot
    from tensorfold.pli import checkpoint_bits as _cb

    if entry.nbytes > limit_bytes:
        return False
    started = time.perf_counter()
    try:
        # PLI (KVD5): with PLI_CHECKPOINT_BITS on, the file carries reduced K and V
        save_snapshot(directory, model_id, entry.tokens, _cb.for_disk(entry.cache), keep=1 << 30)   # pruned by bytes below
    except Exception as exc:  # noqa: BLE001 - a full disk costs a later prefill, never the request
        print(f"[tensorfold] conversation spill failed: {type(exc).__name__}: {exc}", flush=True)
        return False
    prune_conversations(directory, model_id, limit_bytes)
    _ev.ev("spill", conv=_ev.conv_of_tokens(entry.tokens), tokens=len(entry.tokens),
           gib=entry.nbytes / 1024**3, duration=time.perf_counter() - started)
    print(f"[tensorfold] spilled conversation tokens={len(entry.tokens)} ({entry.nbytes / 1024**3:.1f} GiB) "
          f"in {time.perf_counter() - started:.2f}s", flush=True)
    return True


def prune_conversations(directory: Path, model_id: str, limit_bytes: int) -> None:
    """Delete this model's oldest saved conversations until the rest fit ``limit_bytes``."""

    from tensorfold.engine.prefix_snapshots import read_metadata

    ours = []
    for path in sorted(directory.glob("*.safetensors"), key=lambda p: p.stat().st_mtime, reverse=True):
        if path.name.endswith(".partial.safetensors"):
            continue
        try:
            same = str(read_metadata(path).get("model", "")).split("|")[0] == model_id.split("|")[0]
        except Exception:  # noqa: BLE001 - an unreadable file is left alone
            continue
        if same:
            ours.append(path)
    total = 0
    for path in ours:
        total += path.stat().st_size
        if total > limit_bytes:
            path.unlink(missing_ok=True)


class CheckpointStore:
    """LRU caches require strict-prefix hits because GDN state cannot truncate; pinned system blocks bypass slot limits and outlast conversations."""

    def __init__(
        self,
        slots: int,
        copier: Callable[[list[Any]], list[Any]],
        *,
        budget_bytes: int | None = None,
        sizer: Callable[[list[Any]], int] | None = None,
        pinned_slots: int = 3,
        on_evict: Callable[[CheckpointEntry], Any] | None = None,
    ) -> None:
        if slots < 1:
            raise ValueError("slots must be positive")
        if budget_bytes is not None and budget_bytes < 1:
            raise ValueError("budget_bytes must be positive when given")
        self.slots = int(slots)
        self.pinned_slots = int(pinned_slots)
        self.copier = copier
        self.budget_bytes = None if budget_bytes is None else int(budget_bytes)
        self.sizer = sizer
        self._entries: list[CheckpointEntry] = []
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        # set when a memory controller evicts on demand: the newest entry may then exceed the byte budget
        self.admit_oversize = False
        # each evicted conversation no remaining entry extends, outside the lock on the thread that owns the arrays
        self.on_evict = on_evict
        self.spilled = 0
        self.refused = 0                   # a prefix memory or the budget refused: counted, never silent (issue #155)
        self.reduced = 0                   # PLI (KVD5): checkpoints reduced to 8 or 4 bits

    def _evicted(self, gone: list[CheckpointEntry]) -> None:
        if self.on_evict is None:
            return
        with self._lock:
            # PLI (29/10/2026): a reply end does not count as a continuation. An agent that rewrites its tool calls (Hermes Agent:
            # sorted keys, changed ids) never resumes it; the evicted prompt checkpoint did not go to disk because the reply end
            # extended it, and the next turn read everything again (night of 28 to 29/10: 15 turns of 30,000 to 80,000 tokens, prompt
            # checkpoint evicted by another session). PLI_SPILL_PROMPT=0: as before.
            remaining = [entry.tokens for entry in self._entries
                         if not (_SPILL_PROMPT and entry.tokens != entry.last_prompt[:len(entry.tokens)])]
        for entry in gone:
            # an older checkpoint of a conversation that moved on continues from the newer entry: no write
            if entry.pinned or any(extends(t, entry.tokens) for t in remaining):
                continue
            try:
                if self.on_evict(entry) is not False:
                    self.spilled += 1
            except Exception as exc:  # noqa: BLE001 - a failed spill costs a later prefill, never the request
                print(f"[tensorfold] eviction hook failed: {type(exc).__name__}: {exc}", flush=True)

    @property
    def nbytes(self) -> int:
        return sum(entry.nbytes for entry in self._entries)

    def _recount(self, entries: list[CheckpointEntry]) -> None:
        """PLI (29/10/2026): the checkpoints' bytes, each KV layer counted once in the store.

        Under K5, a length checkpoint (``KVLen``) and the real checkpoint at the reply end hold the same KV layer of the
        conversation; each checkpoint's count counted it whole, so twice (log of the evening of 28/10: prompt checkpoint at
        59,269 tokens counted 3.78 GiB, like the reply end that extends it, then evicted for the 8 GiB budget).
        The layer is counted by the first checkpoint that holds it (the most recent), at 0 in the next ones; the other layers
        (DeltaNet, MTP head, draft) belong to each checkpoint and are counted as before. Without K5: as before."""

        from tensorfold.pli import kv_sharing as _kvp

        if not _kvp.ACTIVE or self.sizer is None:
            return
        seen: set[int] = set()
        for e in entries:
            total = 0
            for item in e.cache:
                if isinstance(item, _kvp.KVLen) or _kvp.is_kv(item):
                    owner = _kvp.owner(item)
                    if id(owner) in seen:
                        continue
                    seen.add(id(owner))
                    total += int(self.sizer([owner]))
                else:
                    total += int(self.sizer([item]))
            e.nbytes = total

    def _best(self, prompt: list[int], usable: Any) -> CheckpointEntry | None:
        best: CheckpointEntry | None = None
        for entry in self._entries:
            tokens = entry.tokens
            if usable is not None and not usable(len(tokens)):
                continue
            if 0 < len(tokens) < len(prompt) and prompt[: len(tokens)] == tokens:
                if best is None or len(tokens) > len(best.tokens):
                    best = entry
        return best

    def peek(self, prompt: list[int], usable: Any = None) -> CheckpointEntry | None:
        """The entry ``match`` would use, without counting a hit or copying."""

        with self._lock:
            return self._best(prompt, usable)

    def match(self, prompt: list[int], usable: Any = None, *,
              take: bool = False) -> tuple[int, list[Any], list[int]] | None:
        """Return the longest usable strict-prefix hit with a copied cache, or remove and transfer the stored arrays with ``take``."""

        with self._lock:
            best = self._best(prompt, usable)
            if best is None:
                self.misses += 1
                return None
            self.hits += 1
            self._entries.remove(best)
            previous = list(best.last_prompt)
            from tensorfold.pli import kv_sharing as _kvp
            if _kvp.ACTIVE:
                # PLI: free KV layers given without a copy; the checkpoint keeps lengths
                n = len(best.tokens)
                live = _kvp.resume(best.cache, n, self.copier, self._entries, take=take)
                if take:
                    return n, live, previous
                best.cache = _kvp.as_lengths(best.cache, live, n)
                self._entries.insert(0, best)
                self._recount(self._entries)
                best.last_prompt = list(prompt)
                return n, live, previous
            from tensorfold.pli import checkpoint_bits as _cb
            if _cb.has_reduced(best.cache):          # PLI (KVD5): reduced layers restored to 16 bits for the request
                if take:
                    return len(best.tokens), _cb.restore_cache(best.cache), previous
                self._entries.insert(0, best)
                best.last_prompt = list(prompt)
                return len(best.tokens), _cb.restore_cache(best.cache, self.copier), previous
            if take:
                return len(best.tokens), best.cache, previous
            self._entries.insert(0, best)
            best.last_prompt = list(prompt)
            return len(best.tokens), self.copier(best.cache), previous

    def nearest_prompt(self, prompt: list[int]) -> list[int] | None:
        """PLI (PLI_FORK_MIN): with no checkpoint prefixing the prompt, the kept prompt that shares the longest start with it
        (the same document sent again with another question); None if none shares PLI_FORK_MIN tokens. Not counted as a hit or miss."""

        if _FORK_MIN <= 0 or not prompt:
            return None
        with self._lock:
            best, best_n = None, 0
            for entry in self._entries:
                last = entry.last_prompt
                if not last or last[0] != prompt[0]:
                    continue
                lo, hi = 0, min(len(last), len(prompt))
                while lo < hi:                       # longest common start, by comparing slices
                    mid = (lo + hi + 1) // 2
                    if last[:mid] == prompt[:mid]:
                        lo = mid
                    else:
                        hi = mid - 1
                if lo > best_n:
                    best, best_n = last, lo
            return list(best) if best is not None and best_n >= _FORK_MIN else None

    def longest(self, prompt: list[int], usable: Any = None) -> int:
        """Length of the longest strict-prefix entry ``usable`` accepts (0 if none); not counted as a hit or miss."""

        with self._lock:
            best = self._best(prompt, usable)
            return len(best.tokens) if best is not None else 0

    def refuse(self, tokens: list[int], cache: list[Any], nbytes: int, reason: str) -> None:
        """A refused prefix: spilled to disk where a later turn can re-read it, else counted and logged (#155)."""

        entry = CheckpointEntry(list(tokens), cache, list(tokens), nbytes)
        if self.on_evict is not None:
            try:
                if self.on_evict(entry) is not False:
                    self.spilled += 1
                    return
            except Exception as exc:  # noqa: BLE001 - a bad file costs a refill, never the request
                print(f"[tensorfold] refused snapshot spillover failed: {type(exc).__name__}: {exc}", flush=True)
        self.refused += 1
        _point("checkpoint_refused", entry, reason=reason)
        print(f"[tensorfold] kept nothing at {len(tokens)} tokens ({reason}): a turn reusing this prefix "
              "re-prefills it", flush=True)

    def insert(self, tokens: list[int], cache: list[Any], *, last_prompt: list[int],
               pinned: bool = False) -> None:
        if not tokens:
            return
        nbytes = int(self.sizer(cache)) if self.sizer is not None else 0
        oversize = self.budget_bytes is not None and nbytes > self.budget_bytes
        if oversize and not self.admit_oversize:
            self.refuse(tokens, cache, nbytes, f"its {nbytes} B copy passes the {self.budget_bytes} B budget")
            return
        with self._lock:
            replaced = [entry for entry in self._entries if entry.tokens == list(tokens)]
            kept = [entry for entry in self._entries if entry.tokens != list(tokens)]
            pinned = pinned or any(entry.pinned for entry in replaced)
            entry = CheckpointEntry(list(tokens), cache, list(last_prompt), nbytes, pinned, len(last_prompt))
            entries = [entry, *kept]
            self._recount(entries)                  # PLI: KV layers counted once
            for extra in [e for e in entries if e.pinned][self.pinned_slots:]:
                extra.pinned = False
            # an oversized newest entry displaces conversations but keeps the pinned system blocks
            limit = self.budget_bytes
            if oversize:
                limit = nbytes + sum(e.nbytes for e in entries[1:] if e.pinned)
            # Never evict the new entry; evict conversations (``_victim``'s order) before system blocks.
            gone: list[CheckpointEntry] = []
            while True:
                over_slots = sum(1 for e in entries if not e.pinned) > self.slots
                over_budget = (limit is not None and len(entries) > 1
                               and sum(e.nbytes for e in entries) > limit)
                if not (over_slots or over_budget):
                    break
                # PLI (KVD5): for the budget, reduce the oldest checkpoint that has a real copy before evicting
                if over_budget and not over_slots and self._reduce_one(entries, keep=entry):
                    continue
                unpinned = [i for i in range(1, len(entries)) if not entries[i].pinned]
                if unpinned:
                    gone.append(entries.pop(self._victim(entries, unpinned)))
                    self._recount(entries)
                    _point("checkpoint_evicted", gone[-1], reason="slots" if over_slots else "budget",
                           by=len(entry.tokens), budget_gib=round((limit or 0) / 1024**3, 2))
                elif over_budget:
                    entries.pop()
                    self._recount(entries)
                else:
                    break
                self.evictions += 1
            from tensorfold.pli import kv_sharing as _kvp
            if _kvp.ACTIVE and gone:                 # PLI: the length checkpoints follow their real layer
                follow = _kvp.evict(entries, gone, keep=entry)
                for g in follow:
                    _point("checkpoint_evicted", g, reason="k5_layer_gone")
                gone += follow
                self._recount(entries)
            self._entries = entries
            _point("checkpoint_kept", entry, slots=len(entries), gib_total=round(sum(e.nbytes for e in entries) / 1024**3, 2))
        self._evicted(gone)

    def __len__(self) -> int:
        return len(self._entries)

    @staticmethod
    def _victim(entries: list[CheckpointEntry], candidates: list[int]) -> int:
        """Of ``candidates`` (oldest last): one a later turn's checkpoint continues, oldest first, else the oldest."""

        for i in reversed(candidates):
            entry = entries[i]
            if any(other.born > entry.born and extends(other.tokens, entry.tokens) for other in entries):
                return i
        return candidates[-1]

    def _reduce_one(self, entries: list[CheckpointEntry], keep: CheckpointEntry | None = None) -> bool:
        """PLI (KVD5, PLI_CHECKPOINT_BITS): reduces the KV layers of the oldest ordinary checkpoint that has a real copy of
        them (a free layer that no other checkpoint holds by length); counts again; False if there is none. Off: False."""

        from tensorfold.pli import checkpoint_bits as _cb
        from tensorfold.pli import kv_sharing as _kvs

        if _cb.BITS == 0:
            return False
        held = {id(item.owner) for e in entries for item in e.cache if isinstance(item, _kvs.KVLen)}
        for e in reversed(entries):
            if e is keep or e.pinned:
                continue
            if not any(_cb.reducible(item) for item in e.cache):
                continue
            if any(_cb.reducible(item) and (getattr(item, "pli_live", 0) or id(item) in held) for item in e.cache):
                continue
            before = e.nbytes
            e.cache = _cb.reduce_cache(e.cache)
            if self.sizer is not None:
                e.nbytes = int(self.sizer(e.cache))
            self._recount(entries)
            self.reduced += 1
            _point("checkpoint_reduced", e, before_gib=round(before / 1024**3, 3), bits=_cb.BITS)
            return True
        return False

    def evict_one(self, keep: CheckpointEntry | None = None) -> bool:
        """Free an ordinary prefix (``_victim``'s pick), then a pinned one when memory needs it; never ``keep``."""

        with self._lock:
            if self._reduce_one(self._entries, keep=keep):      # PLI (KVD5): reducing frees memory before evicting
                return True
            candidates = [i for i, entry in enumerate(self._entries) if entry is not keep]
            if not candidates:
                return False
            ordinary = [i for i in candidates if not self._entries[i].pinned]
            gone = self._entries.pop(self._victim(self._entries, ordinary) if ordinary else candidates[-1])
            self.evictions += 1
            _point("checkpoint_evicted", gone, reason="memory")
            evicted = [gone]
            from tensorfold.pli import kv_sharing as _kvp
            if _kvp.ACTIVE:                          # PLI
                follow = _kvp.evict(self._entries, [gone], keep=keep)
                for g in follow:
                    _point("checkpoint_evicted", g, reason="k5_layer_gone")
                evicted += follow
                self._recount(self._entries)
        self._evicted(evicted)
        return True
