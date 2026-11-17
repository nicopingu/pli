"""The general model frees memory before a heavy child process (PLI 5, behind PLI_GEN=1): fake engine, fake disk, no model."""

from __future__ import annotations

import json
import socket
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

import pli
from pli.gen import memoire_general
from pli.gen.memoire_general import MemoireGeneral
from pli.gen.superviseur import GO, Famille, Superviseur
from pli.server.checkpoints import CheckpointStore

SRC = str(Path(pli.__file__).resolve().parent.parent)
JSON = {"Content-Type": "application/json"}
U = 4 * GO


class Couche:
    def __init__(self, nbytes, live=0):
        self.nbytes, self.pli_live = nbytes, live


class FauxMLX:
    """active: bytes of live arrays; cache: free buffers that clear_cache returns."""

    def __init__(self, active=0, cache=0):
        self.active, self.cache = active, cache
        self.limite, self.cache_limite = 100 * GO, 8 * GO
        self.appels = []

    def get_active_memory(self):
        return self.active

    def get_cache_memory(self):
        return self.cache

    def clear_cache(self):
        self.appels.append("clear_cache")
        self.cache = 0

    def set_memory_limit(self, n):
        ancien, self.limite = self.limite, n
        return ancien

    def set_cache_limit(self, n):
        ancien, self.cache_limite = self.cache_limite, n
        return ancien


class FauxStore(CheckpointStore):
    """A real checkpoint store; an eviction returns its bytes to the fake MLX once the write is done."""

    def __init__(self, mlx, **kw):
        super().__init__(10, copier=lambda c: c, sizer=lambda cache: sum(i.nbytes for i in cache), **kw)
        self.mlx = mlx
        self.ecrits = []
        self.on_evict = self._ecrire

    def _ecrire(self, entree):
        self.ecrits.append(entree.tokens[0])
        return True

    def evict_oldest(self, **kw):
        entree = super().evict_oldest(**kw)
        if entree is not None:
            self.mlx.active -= entree.nbytes
        return entree


def conversation(store, mlx, n, octets=U, live=0, pinned=False):
    mlx.active += octets
    store.insert([n, n, n], [Couche(octets, live)], last_prompt=[n, n, n, 9], pinned=pinned)


def memoire(mlx, store, *, bus=None, **kw):
    kw.setdefault("on_engine", lambda fn, timeout: fn(None))
    kw.setdefault("peut_ecrire", lambda entree: True)
    return MemoireGeneral(store=store, runtime=mlx, **kw)


def monde(n=3, **kw):
    mlx = FauxMLX(active=15 * GO)                              # the weights
    store = FauxStore(mlx, **kw)
    for i in range(1, n + 1):                                  # 1 is the oldest
        conversation(store, mlx, i)
    return mlx, store


def test_les_points_sont_evinces_du_plus_ancien_au_plus_recent_et_ecrits():
    mlx, store = monde(3)
    m = memoire(mlx, store)
    assert m.liberer_pour_gen(2 * U - 1) == 2 * U
    assert store.ecrits == [1, 2]                              # oldest first, written before being lost
    assert [e.tokens[0] for e in store._entries] == [3]


def test_le_cache_de_tampons_est_vide_en_premier_et_peut_suffire():
    mlx, store = monde(2)
    mlx.cache = 3 * GO
    m = memoire(mlx, store)
    assert m.liberer_pour_gen(2 * GO) == 3 * GO
    assert mlx.appels[0] == "clear_cache" and store.ecrits == [] and len(store) == 2


def test_un_point_en_cours_n_est_pas_touche():
    mlx, store = monde(0)
    conversation(store, mlx, 1, live=1)                        # the oldest is read by a request
    conversation(store, mlx, 2)
    conversation(store, mlx, 3)
    m = memoire(mlx, store)
    assert m.liberer_pour_gen(U) == U
    assert store.ecrits == [2]
    assert sorted(e.tokens[0] for e in store._entries) == [1, 3]


def test_un_point_non_ecrivable_est_garde():
    mlx, store = monde(2)
    m = memoire(mlx, store, peut_ecrire=lambda entree: False)  # larger than the disk ceiling
    assert m.liberer_pour_gen(U) == 0
    assert len(store) == 2 and store.ecrits == [] and not m.abaisse()


def test_sans_ecriture_possible_aucun_point_n_est_perdu():
    mlx, store = monde(2)
    store.on_evict = None                                      # --spill-gib 0: eviction would discard the checkpoint
    assert memoire(mlx, store).liberer_pour_gen(U) == 0
    assert len(store) == 2


def test_un_point_continue_par_un_plus_recent_part_sans_ecriture():
    mlx = FauxMLX(active=15 * GO)
    store = FauxStore(mlx)
    store.on_evict = None
    mlx.active += 2 * U
    store.insert([1, 2], [Couche(U)], last_prompt=[1, 2, 3])
    store.insert([1, 2, 3], [Couche(U)], last_prompt=[1, 2, 3, 4])   # the newest contains the state of the first
    assert memoire(mlx, store).liberer_pour_gen(U) == U
    assert [e.tokens for e in store._entries] == [[1, 2, 3]]


def test_les_blocs_systeme_epingles_restent():
    mlx, store = monde(0)
    conversation(store, mlx, 1, pinned=True)
    conversation(store, mlx, 2)
    assert memoire(mlx, store).liberer_pour_gen(10 * U) == U
    assert [e.tokens[0] for e in store._entries] == [1]


def test_la_limite_est_abaissee_puis_restauree():
    mlx, store = monde(3)
    budget = SimpleNamespace(budget=50 * GO)
    m = memoire(mlx, store, prompt_memory=budget)
    rendu = m.liberer_pour_gen(2 * U)
    assert rendu == 2 * U
    assert mlx.limite == 100 * GO - rendu and mlx.cache_limite == 2 * GO  # cache gardé au général (2 Go par défaut, PLI_GEN_CACHE_GO)
    assert budget.budget == 50 * GO - rendu
    assert m.abaisse()
    m.rendre_apres_gen()
    assert (mlx.limite, mlx.cache_limite, budget.budget) == (100 * GO, 8 * GO, 50 * GO) and not m.abaisse()
    m.rendre_apres_gen()                                       # a second return changes nothing
    assert (mlx.limite, mlx.cache_limite, budget.budget) == (100 * GO, 8 * GO, 50 * GO)


def test_deux_liberations_se_cumulent_et_se_rendent_d_un_coup():
    mlx, store = monde(3)
    m = memoire(mlx, store)
    m.liberer_pour_gen(U)
    m.liberer_pour_gen(U)
    assert mlx.limite == 100 * GO - 2 * U
    m.rendre_apres_gen()
    assert mlx.limite == 100 * GO and mlx.cache_limite == 8 * GO


def test_la_limite_ne_passe_pas_sous_la_memoire_utilisee_plus_la_marge():
    mlx, store = monde(1)
    mlx.limite = 20 * GO
    m = memoire(mlx, store, marge_octets=2 * GO)
    m.liberer_pour_gen(U)
    assert mlx.limite >= mlx.active + 2 * GO


def test_moteur_occupe_rien_n_est_touche_et_l_appel_tardif_ne_fait_rien():
    mlx, store = monde(3)
    en_file = []

    def occupe(fn, timeout):
        en_file.append(fn)
        raise TimeoutError

    m = memoire(mlx, store, on_engine=occupe)
    assert m.liberer_pour_gen(U) == 0
    assert en_file[0](None) == 0                               # the engine frees up later: the abandoned call is null
    assert len(store) == 3 and not m.abaisse() and mlx.limite == 100 * GO


def test_sans_magasin_de_points_seul_le_cache_est_rendu():
    mlx = FauxMLX(active=15 * GO, cache=2 * GO)
    assert memoire(mlx, None).liberer_pour_gen(10 * GO) == 2 * GO


def test_attend_la_fin_des_ecritures_avant_de_mesurer():
    mlx, store = monde(2)
    vu = []
    m = memoire(mlx, store, attendre_ecritures=lambda delai: vu.append((delai, len(store))) or True)
    m.liberer_pour_gen(U)
    assert vu and vu[0][1] == 1


# --- superviseur ---------------------------------------------------------------------------------------------

def port_libre() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def image(octets=10 * GO):
    return Famille(nom="image", argv=[sys.executable, "-m", "pli.gen.fils_simule", "--port", "{port}"],
                   port=port_libre(), sante_delai_s=20, memoire_octets=octets, delai_decharge_s=0.3)


@pytest.fixture
def fabrique():
    faits = []

    def faire(fam, **kw):
        kw.setdefault("echange_octets", lambda: 0)
        kw.setdefault("reserve_octets", 6 * GO)
        kw.setdefault("delai_kill_s", 3.0)
        kw.setdefault("delai_remesure_s", 0.0)
        kw.setdefault("env", {"PYTHONPATH": SRC})
        sup = Superviseur({fam.nom: fam}, **kw)
        faits.append(sup)
        return sup

    yield faire
    for sup in faits:
        sup.arreter()


def appel(sup):
    return sup.relayer("image", "POST", "/v1/images/generations", JSON, b"{}")


def test_le_superviseur_libere_avant_de_lancer_puis_rend_a_l_arret(fabrique):
    libre = [12 * GO]                                          # 16 Go requis
    demandes, rendus = [], []

    def liberer(octets):
        demandes.append(octets)
        libre[0] += 5 * GO
        return 5 * GO

    sup = fabrique(image(), memoire_disponible=lambda: libre[0], liberer=liberer, rendre=lambda: rendus.append(1),
                   intervalle_s=0.05)
    assert appel(sup)[0] == 200
    assert demandes == [4 * GO]                                # need - available + reserve
    assert rendus == []                                        # the child is alive: the general model stays lowered
    proc = sup._etats["image"].fils.proc
    deadline = time.monotonic() + 10
    while not rendus and time.monotonic() < deadline:
        time.sleep(0.05)                                       # unloaded after 0.3 s of inactivity
    assert proc.poll() is not None and rendus == [1]


def test_503_si_toujours_insuffisant_apres_liberation_et_le_general_est_rendu(fabrique):
    rendus = []
    sup = fabrique(image(), memoire_disponible=lambda: 12 * GO, liberer=lambda octets: 1 * GO,
                   rendre=lambda: rendus.append(1))
    code, _, corps = appel(sup)
    erreur = json.loads(corps)["error"]
    assert code == 503 and erreur["type"] == "memoire" and erreur["liberes_octets"] == 1 * GO
    assert sup._etats["image"].fils is None and rendus == [1]


def test_503_sans_rien_rendre_quand_la_liberation_ne_rend_rien(fabrique):
    rendus = []
    sup = fabrique(image(), memoire_disponible=lambda: 12 * GO, liberer=lambda octets: 0,
                   rendre=lambda: rendus.append(1))
    assert appel(sup)[0] == 503 and rendus == []


def test_pas_de_liberation_quand_la_memoire_suffit(fabrique):
    appels = []
    sup = fabrique(image(), memoire_disponible=lambda: 64 * GO, liberer=lambda o: appels.append(o) or 0)
    assert appel(sup)[0] == 200 and appels == []


def test_une_liberation_en_echec_donne_503_sans_exception(fabrique):
    def casse(octets):
        raise RuntimeError("engine stopped")

    sup = fabrique(image(), memoire_disponible=lambda: 12 * GO, liberer=casse)
    assert appel(sup)[0] == 503


def test_sans_rappels_le_comportement_est_celui_d_avant(fabrique):
    sup = fabrique(image(), memoire_disponible=lambda: 12 * GO)
    assert appel(sup)[0] == 503 and sup.liberer is None and sup.rendre is None


def test_sans_pli_gen_rien_n_est_branche(monkeypatch):
    from pli.server.http import make_handler

    monkeypatch.delenv("PLI_GEN", raising=False)
    appels = []
    monkeypatch.setattr(memoire_general, "brancher", lambda *a: appels.append(a))
    app = SimpleNamespace(scheduler=SimpleNamespace())
    make_handler(app)
    assert appels == [] and not hasattr(app, "gen")


def test_brancher_relie_les_deux_rappels_au_moteur_du_general():
    mlx, store = monde(2)
    sup = SimpleNamespace()
    vues = []
    app = SimpleNamespace(
        checkpoints=store, spill_writer=None, spill_bytes=10 * U,
        prompt_memory=SimpleNamespace(runtime=mlx, budget=40 * GO, release_freed=lambda: vues.append("tampons")),
        scheduler=SimpleNamespace(on_engine=lambda fn, timeout: fn(None), engine=SimpleNamespace()))
    m = memoire_general.brancher(sup, app)
    assert sup.liberer == m.liberer_pour_gen and sup.rendre == m.rendre_apres_gen
    assert sup.liberer(U) == U and vues and store.ecrits == [1]
    sup.rendre()
    assert mlx.limite == 100 * GO and app.prompt_memory.budget == 40 * GO


# --- PLI 5.6.3 (11/17/2026): cap while a child runs, even when nothing had to be freed


def test_le_plafond_vide_le_cache_et_fige_la_memoire_du_general_puis_se_rend():
    mlx, store = monde(2)                                      # 15 Go de poids + 2 x 4 Go de conversations
    mlx.cache = 3 * GO
    budget = SimpleNamespace(budget=40 * GO)
    m = memoire(mlx, store, prompt_memory=budget, marge_octets=2 * GO)
    assert m.plafonner_pour_gen() is True and m.abaisse()
    utilise = 15 * GO + 2 * U
    assert mlx.cache == 0 and mlx.cache_limite == 2 * GO       # cache vidé, 2 Go laissés au général
    assert mlx.limite == utilise + 2 * GO + 2 * GO             # ce qu'il tient + marge + cache gardé
    assert budget.budget == mlx.limite and len(store) == 2     # aucune conversation évincée
    m.rendre_apres_gen()
    assert (mlx.limite, mlx.cache_limite, budget.budget) == (100 * GO, 8 * GO, 40 * GO) and not m.abaisse()


def test_le_plafond_ne_remonte_jamais_une_limite_plus_basse():
    mlx, store = monde(1)
    mlx.limite = 10 * GO
    m = memoire(mlx, store)
    assert m.plafonner_pour_gen() is True and mlx.limite == 10 * GO
    m.rendre_apres_gen()
    assert mlx.limite == 10 * GO


def test_apres_une_liberation_le_plafond_garde_ses_limites():
    mlx, store = monde(3)
    m = memoire(mlx, store)
    m.liberer_pour_gen(U)
    limite = mlx.limite
    assert m.plafonner_pour_gen() is True and mlx.limite == limite


def test_moteur_occupe_le_plafond_ne_touche_rien():
    mlx, store = monde(2)

    def occupe(fn, timeout):
        raise TimeoutError

    m = memoire(mlx, store, on_engine=occupe)
    assert m.plafonner_pour_gen() is False and not m.abaisse() and mlx.limite == 100 * GO


def test_le_superviseur_plafonne_au_lancement_puis_rend_a_l_arret(fabrique):
    plafonds, rendus, liberations = [], [], []
    sup = fabrique(image(), memoire_disponible=lambda: 64 * GO, liberer=lambda o: liberations.append(o) or 0,
                   rendre=lambda: rendus.append(1), plafonner=lambda: plafonds.append(1) or True, intervalle_s=0.05)
    assert appel(sup)[0] == 200
    assert plafonds == [1] and liberations == [] and rendus == []
    proc = sup._etats["image"].fils.proc
    deadline = time.monotonic() + 10
    while not rendus and time.monotonic() < deadline:
        time.sleep(0.05)
    assert proc.poll() is not None and rendus == [1]


def test_un_plafond_en_echec_n_empeche_pas_le_lancement(fabrique):
    def casse():
        raise RuntimeError("engine stopped")

    sup = fabrique(image(), memoire_disponible=lambda: 64 * GO, plafonner=casse)
    assert appel(sup)[0] == 200


def test_brancher_relie_aussi_le_plafond():
    mlx, store = monde(2)
    sup = SimpleNamespace()
    app = SimpleNamespace(
        checkpoints=store, spill_writer=None, spill_bytes=10 * U,
        prompt_memory=SimpleNamespace(runtime=mlx, budget=40 * GO, release_freed=lambda: None),
        scheduler=SimpleNamespace(on_engine=lambda fn, timeout: fn(None), engine=SimpleNamespace()))
    m = memoire_general.brancher(sup, app)
    assert sup.plafonner == m.plafonner_pour_gen and m.delai_moteur_s == 60.0
    assert sup.plafonner() is True
    sup.rendre()
    assert mlx.limite == 100 * GO and app.prompt_memory.budget == 40 * GO
