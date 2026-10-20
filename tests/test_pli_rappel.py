import gc
import os
import sys
pass
os.environ["PLI_RAPPEL"] = "1"
from tensorfold.pli import rappel


class Base:
    last_match = 0
    def propose(self, context, max_draft):
        return []
    def observe(self, p, a):
        self.vu = (p, a)
    def telemetry(self):
        return {"ok": 1}


def demande(reg, ctx, max_draft=8):
    """Une demande : propose à chaque jeton, puis fin (le proposeur est libéré)."""
    p = rappel.ProposeurRappel(Base(), reg)
    props = [p.propose(ctx[:i], max_draft) for i in range(rappel.CLE, len(ctx) + 1)]
    del p
    gc.collect()
    return props


def test_rien_avant_la_fin_puis_recopie():
    reg = rappel.Registre(10000)
    prompt, reponse = list(range(100, 160)), list(range(500, 540))
    ctx = prompt + reponse
    props = demande(reg, ctx)
    assert all(x == [] for x in props)                     # rien au registre pendant la demande
    assert len(reg.ids) == len(ctx)                          # versée à la fin
    b = rappel.ProposeurRappel(Base(), reg)
    c1 = list(prompt)
    assert b.propose(c1, 8) == reponse[:8] and b.source == "rappel" and b.last_match >= rappel.CLE
    assert b.propose(c1, 8) == reponse[:8]                   # même contexte : même réponse (profondeur puis tour)
    b.observe(8, 8)
    assert reg.acceptes == 8
    c1 += reponse[:10]
    assert b.propose(c1, 5) == reponse[10:15]
    assert b.telemetry() == {"ok": 1}


def test_regle_tete():
    reg = rappel.Registre(10000)
    prompt, reponse = list(range(100, 160)), list(range(500, 540))
    demande(reg, prompt + reponse)
    b = rappel.ProposeurRappel(Base(), reg)
    for _ in range(30):
        b.observe_tete(7)                                    # le brouillon du modèle fait 7 jetons par tour
    reg.attendu = [3.0, 3.0, 3.0]                            # le registre en fait 3
    refus = [b.propose(prompt + reponse[:i], 8) for i in range(0, 7)]
    assert refus[:6] == [[]] * 6                             # écartée, sauf sondage au 8e refus
    b.observe_tete(0)
    assert b.tete < 7


def test_taire_apres_echec():
    reg = rappel.Registre(10000)
    prompt, reponse = list(range(100, 160)), list(range(500, 540))
    demande(reg, prompt + reponse)
    b = rappel.ProposeurRappel(Base(), reg)
    assert b.propose(prompt, 8)
    b.observe(8, 0)
    assert [b.propose(prompt + [1] * i, 8) for i in range(1, 5)] == [[]] * 4


def test_base_d_abord():
    class B2(Base):
        last_match = 9
        def propose(self, context, max_draft):
            return [1, 2, 3]
    reg = rappel.Registre(10000)
    p = rappel.ProposeurRappel(B2(), reg)
    assert p.propose(list(range(50)), 4) == [1, 2, 3] and p.source == "base" and p.last_match == 9
    p.observe(3, 2)
    assert p.base.vu == (3, 2)


def test_elagage_et_dedoublement():
    reg = rappel.Registre(1000)
    for k in range(30):
        demande(reg, [7] * 5 + list(range(k * 100, k * 100 + 80)))
    assert len(reg.ids) <= 1000
    long_prompt = list(range(20000, 20600))
    reg.verser(long_prompt)
    n = len(reg.ids)
    reg.verser(long_prompt + list(range(30000, 30040)))
    assert len(reg.ids) - n < 200                            # le prompt déjà lu n'est pas reversé en entier


def test_eteint():
    del os.environ["PLI_RAPPEL"]
    rappel._REGISTRE = None
    b = Base()
    assert rappel.envelopper(b) is b
