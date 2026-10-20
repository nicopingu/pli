"""PLI 3.6.0 : journal d'événements (pli/evenements.py)."""

import importlib
import json
import os
import tempfile
import types
import unittest


def charger(chemin: str, max_mo: str | None = None):
    os.environ["PLI_EVENEMENTS"] = chemin
    if max_mo is None:
        os.environ.pop("PLI_EVENEMENTS_MAX_MO", None)
    else:
        os.environ["PLI_EVENEMENTS_MAX_MO"] = max_mo
    from tensorfold.pli import evenements
    return importlib.reload(evenements)


class JournalEvenements(unittest.TestCase):
    def tearDown(self) -> None:
        os.environ.pop("PLI_EVENEMENTS", None)
        os.environ.pop("PLI_EVENEMENTS_MAX_MO", None)
        from tensorfold.pli import evenements
        importlib.reload(evenements)

    def test_eteint_n_ecrit_rien(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            ev = charger("")
            self.assertFalse(ev.actif())
            ev.ev("fin", conv="x")
            self.assertEqual(os.listdir(d), [])

    def test_meme_conversation_sur_plusieurs_tours(self) -> None:
        ev = charger("/dev/null")
        tour1 = [{"role": "system", "content": "outil"}, {"role": "user", "content": "trie la liste"}]
        tour2 = tour1 + [{"role": "assistant", "content": "voici"}, {"role": "user", "content": "et en PHP ?"}]
        autre = [{"role": "system", "content": "outil"}, {"role": "user", "content": "autre sujet"}]
        self.assertEqual(ev.conversation(tour1), ev.conversation(tour2))
        self.assertNotEqual(ev.conversation(tour1), ev.conversation(autre))
        self.assertEqual(len(ev.conversation(tour1)), 10)
        # contenu en morceaux (format Anthropic ou images) : même empreinte que le texte assemblé
        morceaux = [{"role": "system", "content": "outil"}, {"role": "user", "content": [{"type": "text", "text": "trie "}, {"type": "text", "text": "la liste"}]}]
        self.assertEqual(ev.conversation(tour1), ev.conversation(morceaux))

    def test_ligne_ecrite_et_rattachement_par_jetons(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            chemin = os.path.join(d, "ev.jsonl")
            ev = charger(chemin)
            a = types.SimpleNamespace(job_id="req-a", prompt_ids=list(range(100)))
            b = types.SimpleNamespace(job_id="req-b", prompt_ids=list(range(50)) + [7] * 50)
            ev.lier(a, "conva")
            ev.lier(b, "convb")
            self.assertEqual(a.pli_conv, "conva")
            self.assertEqual(ev.conv_des_jetons(list(range(90))), "conva")
            self.assertEqual(ev.conv_des_jetons(list(range(100)) + [1, 2, 3]), "conva")   # invite + réponse
            self.assertEqual(ev.conv_des_jetons(list(range(40))), "partagé")              # début commun
            self.assertEqual(ev.conv_des_jetons([9, 9, 9]), "")
            ev.ev("fin", a, sortie=12, tok_s=41.234567, attente=None)
            ev.ev("deversement", conv=ev.conv_des_jetons(list(range(90))), jetons=90)
            lignes = [json.loads(l) for l in open(chemin, encoding="utf-8")]
            self.assertEqual(lignes[0]["a"], "fin")
            self.assertEqual(lignes[0]["req"], "req-a")
            self.assertEqual(lignes[0]["conv"], "conva")
            self.assertEqual(lignes[0]["tok_s"], 41.235)
            self.assertNotIn("attente", lignes[0])
            self.assertEqual(lignes[1]["conv"], "conva")
            self.assertNotIn("req", lignes[1])

    def test_rotation_et_erreur_avalee(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            chemin = os.path.join(d, "ev.jsonl")
            ev = charger(chemin, "1")
            with open(chemin, "w") as f:
                f.write("x" * (1024 * 1024 + 10))
            ev.ev("fin", conv="c")
            self.assertTrue(os.path.exists(chemin + ".1"))
            self.assertEqual(len(open(chemin).read().splitlines()), 1)
            ev = charger(os.path.join(d, "absent", "ev.jsonl"))
            ev.ev("fin", conv="c")          # dossier absent : rien ne sort


if __name__ == "__main__":
    unittest.main()
