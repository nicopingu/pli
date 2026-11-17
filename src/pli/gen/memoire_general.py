"""PLI 5: the general model frees memory before an image or video child process loads (behind PLI_GEN=1).

``liberer_pour_gen(octets)``: on the engine thread (``Scheduler.on_engine``, engine at rest: no live stream, no
open prompt, so no checkpoint being written), empties the MLX buffer cache, then evicts conversation checkpoints
from oldest to newest, those that can be found on disk or that the next checkpoint continues (the others remain),
waits for writes to finish, measures what is actually freed, and lowers the MLX memory limit, the MLX cache limit and
the prompt admission budget by that amount so the general model does not reclaim it.
``plafonner_pour_gen()`` (PLI 5.6.3): when a child starts without anything to free, empties the buffer cache and caps
the MLX memory limit and the prompt budget at what the general model holds now (plus the margin), so that it does not
grow into the child's memory while the child computes; its new conversations push older checkpoints to disk instead.
``rendre_apres_gen()``: restores these limits (evicted checkpoints are re-read from disk on demand).
``limiter_parallele(n)``: during a video (key video_parallele_general), the general model admits only n generations at a
time; None restores the original count. Scheduler._room re-reads ``lanes`` each turn (server/scheduler.py:497-500):
excess requests wait in the scheduler queue, none are refused, those already started continue.
This module does not import MLX: the execution layer (``mx``) and the engine thread are given to it.
"""

from __future__ import annotations

import os
import threading
from typing import Any, Callable

GO = 1024**3


class MemoireGeneral:
    def __init__(self, *, store: Any, runtime: Any, on_engine: Callable[..., Any],
                 liberer_tampons: Callable[[], None] | None = None,
                 attendre_ecritures: Callable[[float], bool] | None = None,
                 prompt_memory: Any = None,
                 peut_ecrire: Callable[[Any], bool] | None = None,
                 en_cours: Callable[[Any], bool] | None = None,
                 marge_octets: int = 2 * GO, delai_moteur_s: float = 5.0, delai_ecriture_s: float = 120.0,
                 cache_pendant_gen_octets: int = 2 * GO, scheduler: Any = None) -> None:
        self.scheduler = scheduler
        self._lanes_origine: int | None = None
        self.store = store
        self.runtime = runtime
        self.on_engine = on_engine
        self.liberer_tampons = liberer_tampons
        self.attendre_ecritures = attendre_ecritures
        self.prompt_memory = prompt_memory
        self.peut_ecrire = peut_ecrire
        self.en_cours = en_cours or (lambda entree: any(getattr(item, "pli_live", 0) for item in entree.cache))
        self.marge_octets = int(marge_octets)
        # MLX buffer cache left to the general model during a child: 0 made it give back and ask again for each buffer
        # (bench gen5-bench2 of 11/13/2026: first token 1k 5.17 s -> 6.9 to 7.1 s, write -11%)
        self.cache_pendant_gen_octets = max(0, int(cache_pendant_gen_octets))
        self.delai_moteur_s = float(delai_moteur_s)
        self.delai_ecriture_s = float(delai_ecriture_s)
        self._verrou = threading.Lock()
        self._avant: dict[str, int] | None = None      # original limits, present while the general model is lowered
        self.rendu_total = 0                           # bytes kept for child processes since the first lowering
        self.evinces = 0

    # --- API interne ---------------------------------------------------------------------------------------------
    def liberer_pour_gen(self, octets: int) -> int:
        """Bytes actually freed (0 if the engine is not free in time: nothing is then touched)."""

        octets = int(octets)
        if octets <= 0:
            return 0
        abandon = threading.Event()

        def sur_moteur(_engine: Any) -> int:
            if abandon.is_set():                       # the caller gave up: the queued call does nothing
                return 0
            with self._verrou:
                return self._liberer(octets)

        try:
            return int(self.on_engine(sur_moteur, timeout=self.delai_moteur_s))
        except TimeoutError:
            abandon.set()
            return 0

    def rendre_apres_gen(self) -> None:
        with self._verrou:
            avant, self._avant = self._avant, None
            self.rendu_total = 0
            if avant is None:
                return
            self.runtime.set_memory_limit(avant["limite"])
            self.runtime.set_cache_limit(avant["cache"])
            if self.prompt_memory is not None:
                self.prompt_memory.budget += avant["budget_retire"]

    def plafonner_pour_gen(self) -> bool:
        """Caps the general model at its current use while a child runs (True: capped now or already lowered)."""

        abandon = threading.Event()

        def sur_moteur(_engine: Any) -> bool:
            if abandon.is_set():
                return False
            with self._verrou:
                if self._avant is not None:          # already lowered by a release: those limits stay
                    return True
                self._vider()
                self._plafonner()
                return True

        try:
            return bool(self.on_engine(sur_moteur, timeout=self.delai_moteur_s))
        except TimeoutError:
            abandon.set()
            return False

    def abaisse(self) -> bool:
        return self._avant is not None

    def limiter_parallele(self, n: int | None) -> None:
        if self.scheduler is None:
            return
        if self._lanes_origine is None:
            self._lanes_origine = int(self.scheduler.lanes)
        # an integer assignment: the engine thread reads it on the next turn, without a lock
        self.scheduler.lanes = self._lanes_origine if n is None else max(1, min(int(n), self._lanes_origine))

    # --- on the engine thread ------------------------------------------------------------------------------------
    def _utilise(self) -> int:
        return int(self.runtime.get_active_memory()) + int(self.runtime.get_cache_memory())

    def _vider(self) -> None:
        if self.liberer_tampons is not None:
            self.liberer_tampons()
        self.runtime.clear_cache()

    def _liberer(self, octets: int) -> int:
        avant = self._utilise()
        self._vider()
        rendu = avant - self._utilise()
        while rendu < octets:
            estime = rendu
            evince = False
            while estime < octets:
                entree = None if self.store is None else self.store.evict_oldest(can_write=self.peut_ecrire, in_use=self.en_cours)
                if entree is None:
                    break
                evince = True
                self.evinces += 1
                estime += int(entree.nbytes)
            if not evince:
                break
            if self.attendre_ecritures is not None:
                self.attendre_ecritures(self.delai_ecriture_s)     # written arrays remain in memory until then
            self._vider()
            rendu = avant - self._utilise()
        rendu = max(0, rendu)
        if rendu > 0:
            self._abaisser(rendu)
        return rendu

    def _plafonner(self) -> None:
        ancien_cache = int(self.runtime.set_cache_limit(self.cache_pendant_gen_octets))
        if ancien_cache < self.cache_pendant_gen_octets:
            self.runtime.set_cache_limit(ancien_cache)
        self._cache_garde = min(ancien_cache, self.cache_pendant_gen_octets)
        limite = max(self._utilise() + self.marge_octets + self._cache_garde, 1)
        ancienne = int(self.runtime.set_memory_limit(limite))
        if ancienne < limite:                        # never above the original setting
            self.runtime.set_memory_limit(ancienne)
            limite = ancienne
        retire = 0
        if self.prompt_memory is not None:
            retire = max(0, min(int(self.prompt_memory.budget), int(self.prompt_memory.budget) - limite))
            self.prompt_memory.budget -= retire
        self._avant = {"limite": ancienne, "cache": ancien_cache, "budget_retire": retire}
        self.rendu_total = 0

    def _abaisser(self, rendu: int) -> None:
        premier = self._avant is None
        total = rendu if premier else self.rendu_total + rendu
        if premier:
            ancien_cache = int(self.runtime.set_cache_limit(self.cache_pendant_gen_octets))
            if ancien_cache < self.cache_pendant_gen_octets:      # never go above the original setting
                self.runtime.set_cache_limit(ancien_cache)
            self._cache_garde = min(ancien_cache, self.cache_pendant_gen_octets)
        limite = max(self._utilise() + self.marge_octets + getattr(self, "_cache_garde", 0), 1)
        if premier:
            ancienne = self.runtime.set_memory_limit(limite)
            self._avant = {"limite": int(ancienne), "cache": int(ancien_cache), "budget_retire": 0}
        cible = min(self._avant["limite"], max(limite, self._avant["limite"] - total))
        self.runtime.set_memory_limit(cible)
        self.rendu_total = total
        if self.prompt_memory is not None:
            retire = min(rendu, int(self.prompt_memory.budget))
            self.prompt_memory.budget -= retire
            self._avant["budget_retire"] += retire


def brancher(superviseur: Any, app: Any) -> MemoireGeneral:
    """Links the supervisor to the general model of ``app`` via two callbacks; MLX is imported only here."""

    import mlx.core as mx

    prompt_memory = getattr(app, "prompt_memory", None)
    runtime = getattr(prompt_memory, "runtime", None) or mx
    scheduler = app.scheduler
    writer = getattr(app, "spill_writer", None)
    spill_octets = int(getattr(app, "spill_bytes", 0) or 0)

    def tampons() -> None:
        getattr(scheduler.engine, "release_rounds", lambda: None)()
        if prompt_memory is not None:
            prompt_memory.release_freed()

    memoire = MemoireGeneral(
        store=getattr(app, "checkpoints", None), runtime=runtime, on_engine=scheduler.on_engine, liberer_tampons=tampons,
        attendre_ecritures=writer.drain if writer is not None else None, prompt_memory=prompt_memory,
        peut_ecrire=lambda entree: int(entree.nbytes) <= spill_octets,
        marge_octets=int(float(os.environ.get("PLI_GEN_MARGE_GO", "2")) * GO),
        cache_pendant_gen_octets=int(float(os.environ.get("PLI_GEN_CACHE_GO", "2")) * GO), scheduler=scheduler,
        # PLI 5.6.3: up to 60 s for the engine to be at rest between two requests (5 s gave up during short text turns)
        delai_moteur_s=float(os.environ.get("PLI_GEN_ENGINE_WAIT_S", "60")))
    superviseur.liberer = memoire.liberer_pour_gen
    superviseur.rendre = memoire.rendre_apres_gen
    superviseur.plafonner = memoire.plafonner_pour_gen
    superviseur.parallele = memoire.limiter_parallele
    return memoire
