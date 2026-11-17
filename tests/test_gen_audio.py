"""PLI 5 audio child process (fils_audio.py) in --simuler mode: no model, no mlx_audio import."""

from __future__ import annotations

import http.client
import io
import json
import socket
import subprocess
import sys
import time
import wave
from pathlib import Path

import pytest

import pli
from pli.gen import fils_audio
from pli.gen.superviseur import GO, Famille, Superviseur

SRC = str(Path(pli.__file__).resolve().parent.parent)
FILS = str(Path(fils_audio.__file__).resolve())
JSON = {"Content-Type": "application/json"}


def port_libre() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def requete(port, methode, chemin, corps=None, en_tetes=None):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=20)
    c.request(methode, chemin, body=corps, headers=en_tetes or {})
    r = c.getresponse()
    donnees = r.read()
    c.close()
    return r.status, r.getheader("Content-Type"), donnees


def multipart(champs, fichiers, frontiere="XfrontiereX"):
    corps = b""
    for nom, val in champs.items():
        corps += (f"--{frontiere}\r\nContent-Disposition: form-data; name=\"{nom}\"\r\n\r\n{val}\r\n").encode()
    for nom, (fichier, donnees) in fichiers.items():
        corps += (f"--{frontiere}\r\nContent-Disposition: form-data; name=\"{nom}\"; filename=\"{fichier}\"\r\n"
                  f"Content-Type: application/octet-stream\r\n\r\n").encode() + donnees + b"\r\n"
    corps += f"--{frontiere}--\r\n".encode()
    return corps, {"Content-Type": f"multipart/form-data; boundary={frontiere}"}


@pytest.fixture
def fils():
    port = port_libre()
    proc = subprocess.Popen([sys.executable, FILS, "--port", str(port), "--simuler"])
    fin = time.monotonic() + 20
    while time.monotonic() < fin:
        try:
            if requete(port, "GET", "/sante")[0] == 200:
                break
        except OSError:
            time.sleep(0.05)
    else:
        proc.kill()
        pytest.fail("the audio child process is not responding")
    yield port
    proc.terminate()
    proc.wait(10)


def test_sante_et_pas_d_import_de_mlx_audio(fils):
    code, _, corps = requete(fils, "GET", "/sante")
    sante = json.loads(corps)
    assert code == 200 and sante["ok"] and sante["simule"]
    assert sante["mlx_audio_importe"] is False
    assert sante["voix"] == ["kokoro-siwis"]                          # public code: Kokoro only
    assert sante["defaut"] == {"*": "kokoro-siwis", "kokoro": "kokoro-siwis"}


def test_speech_renvoie_un_wav_valide(fils):
    code, type_, corps = requete(fils, "POST", "/v1/audio/speech", json.dumps({"input": "Bonjour tout le monde"}), JSON)
    assert code == 200 and type_ == "audio/wav" and corps[:4] == b"RIFF"
    with wave.open(io.BytesIO(corps)) as w:
        assert w.getnchannels() == 1 and w.getsampwidth() == 2
        assert w.getnframes() / w.getframerate() > 0
    # light voice and speed
    d = {"input": "Salut", "model": "kokoro", "voice": "kokoro-siwis", "speed": 1.2}
    assert requete(fils, "POST", "/v1/audio/speech", json.dumps(d), JSON)[0] == 200
    pcm = requete(fils, "POST", "/v1/audio/speech", json.dumps({"input": "Salut", "response_format": "pcm"}), JSON)
    assert pcm[0] == 200 and pcm[1] == "audio/pcm" and len(pcm[2]) > 0


def test_speech_demandes_invalides(fils):
    def post(d):
        return requete(fils, "POST", "/v1/audio/speech", json.dumps(d), JSON)
    code, _, corps = post({"input": "x", "voice": "inconnue"})
    assert code == 400 and "inconnue" in json.loads(corps)["error"]["message"]
    assert post({})[0] == 400
    assert post({"input": "x", "response_format": "mp3"})[0] == 400
    code, _, corps = post({"input": "x", "model": "drama3"})          # no Drama 3 voice by default in the code
    assert code == 400 and "defaut" in json.loads(corps)["error"]["message"]
    assert requete(fils, "POST", "/v1/audio/speech", b"pas du json", JSON)[0] == 400


def test_transcriptions_multipart_binaire(fils):
    binaire = bytes(range(256)) * 40 + b"\r\n--XfrontiereX-presque\r\n\x00\xff"
    corps, en_tetes = multipart({"model": "qwen3-asr", "language": "fr"}, {"file": ("a.wav", binaire)})
    code, type_, donnees = requete(fils, "POST", "/v1/audio/transcriptions", corps, en_tetes)
    assert code == 200 and json.loads(donnees) == {"text": "simulated text"}
    code, _, _ = requete(fils, "POST", "/v1/audio/transcriptions", *multipart({"model": "m"}, {}))
    assert code == 400


def test_analyseur_multipart_octets_intacts():
    binaire = bytes(range(256)) * 3 + b"\r\n"
    corps, en_tetes = multipart({"language": "fr"}, {"file": ("x.mp3", binaire)})
    parties = fils_audio.analyser_multipart(en_tetes["Content-Type"], corps)
    assert [p["nom"] for p in parties] == ["language", "file"]
    assert parties[0]["donnees"] == b"fr" and parties[0]["fichier"] is None
    assert parties[1]["donnees"] == binaire and parties[1]["fichier"] == "x.mp3"
    guillemets = 'multipart/form-data; boundary="XfrontiereX"'
    assert len(fils_audio.analyser_multipart(guillemets, corps)) == 2
    with pytest.raises(fils_audio.ErreurClient):
        fils_audio.analyser_multipart("application/json", corps)


def test_catalogue_invalide(tmp_path):
    f = tmp_path / "v.json"
    f.write_text(json.dumps({"voix": {"x": {"modele": "inconnu"}}}))
    with pytest.raises(ValueError):
        fils_audio.charger_catalogue(str(f))
    f.write_text(json.dumps({"voix": {"x": {"modele": "kokoro", "voice": "ff_siwis"}}}))
    assert list(fils_audio.charger_catalogue(str(f))) == ["x"]


def test_catalogue_defaut_et_alias(tmp_path):
    f = tmp_path / "voix.json"
    voix = {"maria": {"modele": "drama3", "references": [["a.wav", "a.txt"]], "alias": ["ancien-nom"]},
            "kokoro-siwis": {"modele": "kokoro", "voice": "ff_siwis", "lang_code": "f"}}
    f.write_text(json.dumps({"voix": voix, "defaut": {"drama3": "maria", "*": "maria"}}))
    cat = fils_audio.charger_catalogue(str(f))
    assert cat.noms() == ["kokoro-siwis", "maria"]
    assert cat["ancien-nom"] is cat["maria"] and cat["ancien-nom"]["_nom"] == "maria"
    assert cat.voix_par_defaut(None) == "maria" and cat.voix_par_defaut("tts-1") == "maria"
    assert cat.voix_par_defaut("drama3") == "maria" and cat.voix_par_defaut("kokoro") == "kokoro-siwis"
    f.write_text(json.dumps({"voix": voix}))                         # without « defaut »: Kokoro, nothing for Drama 3
    cat = fils_audio.charger_catalogue(str(f))
    assert cat.voix_par_defaut(None) == "kokoro-siwis" and cat.voix_par_defaut("drama3") is None
    for mauvais in ({"drama3": "kokoro-siwis"}, {"drama3": "absente"}, {"inconnu": "maria"}, ["maria"]):
        f.write_text(json.dumps({"voix": voix, "defaut": mauvais}))
        with pytest.raises(ValueError):
            fils_audio.charger_catalogue(str(f))
    f.write_text(json.dumps({"voix": {**voix, "autre": {"modele": "kokoro", "voice": "x", "alias": ["maria"]}}}))
    with pytest.raises(ValueError):                                  # an alias does not take the name of a voice
        fils_audio.charger_catalogue(str(f))


def test_defaut_et_alias_servis_par_le_fils(tmp_path):
    cat = tmp_path / "voix.json"
    cat.write_text(json.dumps({"voix": {"maria": {"modele": "drama3", "references": [["a.wav", "a.txt"]],
                                                  "alias": ["ancien-nom"]}},
                               "defaut": {"drama3": "maria", "*": "maria"}}))
    port, proc = lancer_fils(tmp_path, "--voix", cat)
    try:
        sante = json.loads(requete(port, "GET", "/sante")[2])
        assert sante["voix"] == ["maria"] and sante["defaut"]["*"] == "maria"
        assert list(sante["memoire_go"]) == ["maria"]
        assert parole(port, voix="ancien-nom")[0] == 200            # the old name is still served
        code, _, corps = requete(port, "POST", "/v1/audio/speech", json.dumps({"input": "Bonjour"}), JSON)
        assert code == 200 and corps[:4] == b"RIFF"                  # without « voice »: the catalog default
    finally:
        proc.terminate()
        proc.wait(10)


def test_voix_nommee_sans_defaut_drama3(tmp_path):
    """Shape of a production catalog: a named Drama 3 voice (with its old name), Kokoro by default."""

    cat = tmp_path / "voix.json"
    cat.write_text(json.dumps({"voix": {"voix-a": {"modele": "drama3", "references": [["a.wav", "a.txt"]],
                                                   "alias": ["ancien-a"]},
                                        "kokoro-siwis": {"modele": "kokoro", "voice": "ff_siwis", "lang_code": "f"}},
                               "defaut": {"*": "kokoro-siwis"}}))
    port, proc = lancer_fils(tmp_path, "--voix", cat)
    try:
        sante = json.loads(requete(port, "GET", "/sante")[2])
        assert sante["defaut"] == {"*": "kokoro-siwis", "kokoro": "kokoro-siwis"}
        assert parole(port, voix="ancien-a")[0] == 200                # request naming the old name: served
        code, _, corps = requete(port, "POST", "/v1/audio/speech", json.dumps({"input": "Bonjour"}), JSON)
        assert code == 200 and corps[:4] == b"RIFF"                    # without voice: Kokoro
        code, _, corps = requete(port, "POST", "/v1/audio/speech", json.dumps({"input": "x", "model": "drama3"}), JSON)
        assert code == 400                                             # Drama 3 without voice or default: clear refusal
    finally:
        proc.terminate()
        proc.wait(10)


def test_bout_a_bout_superviseur_vers_fils_audio():
    port = port_libre()
    fam = Famille(nom="audio", argv=[sys.executable, FILS, "--port", "{port}", "--simuler"], port=port,
                  memoire_octets=12 * GO, sante_delai_s=20)
    sup = Superviseur({"audio": fam}, memoire_disponible=lambda: 64 * GO, echange_octets=lambda: 0,
                      reserve_octets=0, delai_kill_s=3.0, env={"PYTHONPATH": SRC})
    try:
        corps = json.dumps({"input": "Bonjour", "voice": "kokoro-siwis"}).encode()
        code, en_tetes, wav = sup.relayer("audio", "POST", "/v1/audio/speech", JSON, corps)
        assert code == 200 and en_tetes["Content-Type"] == "audio/wav" and wav[:4] == b"RIFF"
        mp, mp_tetes = multipart({"language": "fr"}, {"file": ("a.wav", wav)})
        code, _, donnees = sup.relayer("audio", "POST", "/v1/audio/transcriptions", mp_tetes, mp)
        assert code == 200 and json.loads(donnees) == {"text": "simulated text"}
        code, _, donnees = sup.relayer("audio", "POST", "/v1/audio/speech", JSON, b'{"input":"x","voice":"zz"}')
        assert code == 400
    finally:
        sup.arreter()


# ---------------------------------------------------------------- preemption and memory

def lancer_fils(tmp_path, *options, drapeau=None):
    import os
    port = port_libre()
    env = {**os.environ}
    env.pop("PYTHONPATH", None)
    if drapeau is not None:
        env["PLI_GEN_DRAPEAU"] = str(drapeau)
    proc = subprocess.Popen([sys.executable, "-I", FILS, "--port", str(port), "--simuler", *map(str, options)],
                            env=env, stdin=subprocess.DEVNULL)
    fin = time.monotonic() + 20
    while time.monotonic() < fin:
        try:
            if requete(port, "GET", "/sante")[0] == 200:
                return port, proc
        except OSError:
            time.sleep(0.05)
    proc.kill()
    raise AssertionError("audio child process not ready")


def parole(port, texte="Un texte assez long pour produire plusieurs morceaux de parole simulés.", voix="kokoro-siwis"):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    c.request("POST", "/v1/audio/speech", body=json.dumps({"input": texte, "voice": voix}), headers=JSON)
    r = c.getresponse()
    corps = r.read()
    c.close()
    return r.status, r.getheader("X-PLI-Attentes"), corps


def test_preemption_attend_le_drapeau_entre_deux_morceaux(tmp_path):
    import threading
    drapeau = tmp_path / "general.drapeau"
    port, proc = lancer_fils(tmp_path, "--duree-morceau", 0.01, drapeau=drapeau)
    try:
        drapeau.write_text("")
        sortie = {}
        fil = threading.Thread(target=lambda: sortie.update(r=parole(port)))
        fil.start()
        time.sleep(0.5)
        assert fil.is_alive() and "r" not in sortie                  # flag set: no response
        drapeau.unlink()
        fil.join(15)
        code, attentes, corps = sortie["r"]
        assert code == 200 and int(attentes) > 0 and corps[:4] == b"RIFF"
        code, attentes, _ = parole(port)
        assert code == 200 and attentes == "0"                       # chemin libre : aucune attente
    finally:
        proc.terminate()
        proc.wait(10)


def test_refus_503_memoire_et_modele_deja_charge(tmp_path):
    # 5 GB available: Kokoro (3 GB) passes, Drama 3 (24 GB) is refused without loading
    cat = tmp_path / "voix.json"
    cat.write_text(json.dumps({"voix": {"drama-essai": {"modele": "drama3", "references": [["a.wav", "a.txt"]]},
                                        "kokoro-siwis": {"modele": "kokoro", "voice": "ff_siwis", "lang_code": "f"}}}))
    port, proc = lancer_fils(tmp_path, "--voix", cat, "--memoire-disponible-go", 5)
    try:
        assert parole(port)[0] == 200
        code, _, corps = parole(port, voix="drama-essai")
        erreur = json.loads(corps)["error"]
        assert code == 503 and erreur["type"] == "memoire" and "24.0 GB required" in erreur["message"]
        assert parole(port)[0] == 200                                # Kokoro remains served
    finally:
        proc.terminate()
        proc.wait(10)


def test_memoire_go_de_la_voix_remplace_le_defaut(tmp_path):
    cat = tmp_path / "voix.json"
    cat.write_text(json.dumps({"voix": {"petite": {"modele": "drama3", "memoire_go": 4,
                                                   "references": [["a.wav", "a.txt"]]}}}))
    port, proc = lancer_fils(tmp_path, "--voix", cat, "--memoire-disponible-go", 5)
    try:
        sante = json.loads(requete(port, "GET", "/sante")[2])
        assert sante["memoire_go"] == {"petite": 4.0}
        assert parole(port, voix="petite")[0] == 200
    finally:
        proc.terminate()
        proc.wait(10)
    cat.write_text(json.dumps({"voix": {"x": {"modele": "kokoro", "memoire_go": 0}}}))
    with pytest.raises(ValueError):
        fils_audio.charger_catalogue(str(cat))


def test_analyseur_vm_stat_du_fils_audio():
    texte = ("Mach Virtual Memory Statistics: (page size of 16384 bytes)\nPages free:   4494.\nPages inactive:  10.\n"
             "Pages speculative: 1.\nPages purgeable: 5.\nPages active: 99999.\n")
    assert fils_audio.analyser_vm_stat(texte) == (4494 + 10 + 1 + 5) * 16384


def test_preemption_drama3_branchee_sur_les_trames(tmp_path):
    """MoteurMlx._brancher_preemption wraps modele._frames: a wait every FRAMES_ENTRE_ATTENTES frames."""

    class Faux:
        def _frames(self, n):
            yield from range(n)

    moteur = fils_audio.MoteurMlx("x", drapeau=str(tmp_path / "absent"))
    appels = []
    moteur.attendre = lambda: appels.append(1)
    faux = Faux()
    moteur._brancher_preemption(faux)
    assert list(faux._frames(20)) == list(range(20))
    assert len(appels) == 3                                          # frames 0, 8 and 16


def test_chargement_par_morceaux_evalue_tout_par_lots(tmp_path):
    """load_model(lazy=True) puis evaluer_par_morceaux : tous les poids lus, par lots, attente du drapeau entre deux lots."""
    mx = pytest.importorskip("mlx.core")
    nn = pytest.importorskip("mlx.nn")
    import numpy as np
    chemin = str(tmp_path / "poids.safetensors")
    source = {f"c{i}.weight": np.random.default_rng(i).standard_normal((512, 512)).astype(np.float32) for i in range(4)}
    mx.save_safetensors(chemin, {k: mx.array(v) for k, v in source.items()})       # 4 tenseurs de 1 Mio

    class Petit(nn.Module):
        def __init__(self):
            super().__init__()
            for i in range(4):
                setattr(self, f"c{i}", nn.Linear(512, 512, bias=False))

    modele = Petit()
    modele.load_weights(list(mx.load(chemin).items()))                             # paresseux, rien de lu
    appels = []
    n = fils_audio.evaluer_par_morceaux(mx, modele, 2 * 1024**2, entre=lambda: appels.append(1))
    assert n == 2 and len(appels) == 2
    for k, v in source.items():
        assert np.array_equal(np.array(getattr(modele, k.split(".")[0]).weight), v)
    assert fils_audio.evaluer_par_morceaux(mx, modele, 10 * 1024**2) == 1             # un seul lot si le morceau couvre tout


# ---------------------------------------------------------------- un seul fil MLX et annulation (moteur simulé, en processus)

@pytest.fixture
def serveur_simule():
    import threading
    lances = []

    def lancer(**kw):
        moteur = fils_audio.MoteurSimule(**kw)
        port = port_libre()
        serveur = fils_audio.faire_serveur(port, fils_audio.charger_catalogue(None), moteur)
        serveur.daemon_threads = True
        threading.Thread(target=serveur.serve_forever, daemon=True).start()
        lances.append(serveur)
        return port, moteur

    yield lancer
    for s in lances:
        s.shutdown()
        s.server_close()


def test_tts_et_asr_concurrents_sur_le_meme_fil_mlx(serveur_simule):
    import threading
    port, moteur = serveur_simule(duree_morceau=0.05)
    reponses = []

    def tts():
        reponses.append(requete(port, "POST", "/v1/audio/speech", json.dumps({"input": "x" * 100}), JSON)[0])

    def asr():
        corps, en_tetes = multipart({"language": "fr"}, {"file": ("a.wav", b"RIFFxxxx")})
        reponses.append(requete(port, "POST", "/v1/audio/transcriptions", corps, en_tetes)[0])

    fils_http = [threading.Thread(target=f) for f in (tts, asr, tts, asr)]
    [t.start() for t in fils_http]
    [t.join(30) for t in fils_http]
    assert reponses == [200] * 4
    assert len(moteur.fils_utilises) == 4
    assert len(set(moteur.fils_utilises)) == 1                      # all MLX on the same thread identifier
    assert moteur.fils_utilises[0] != threading.get_ident()         # and not that of an HTTP thread
    assert moteur.max_en_cours == 1                                 # never two jobs at the same time


def test_annulation_au_milieu_d_une_synthese_puis_le_fils_resservit(serveur_simule):
    import threading
    port, moteur = serveur_simule(duree_morceau=0.1)
    res = {}

    def longue():
        res["r"] = requete(port, "POST", "/v1/audio/speech", json.dumps({"input": "x" * 400}), JSON)   # 20 morceaux, 2 s

    t = threading.Thread(target=longue)
    t0 = time.monotonic()
    t.start()
    time.sleep(0.4)
    code, _, corps = requete(port, "POST", "/annuler")
    assert code == 200 and json.loads(corps) == {"annulee": True}
    t.join(10)
    code, _, corps = res["r"]
    assert code == 499 and json.loads(corps)["error"]["type"] == "annulee"
    assert time.monotonic() - t0 < 1.5                              # stopped well before 2 s
    assert requete(port, "POST", "/v1/audio/speech", json.dumps({"input": "bonjour"}), JSON)[0] == 200
    assert json.loads(requete(port, "POST", "/annuler")[2]) == {"annulee": False}      # rien en cours


def test_annulation_pendant_l_attente_du_drapeau(tmp_path, serveur_simule):
    import threading
    drapeau = tmp_path / "general.drapeau"
    drapeau.write_text("")
    port, moteur = serveur_simule(drapeau=str(drapeau))
    res = {}
    t = threading.Thread(target=lambda: res.update(r=requete(port, "POST", "/v1/audio/speech", json.dumps({"input": "x"}), JSON)))
    t0 = time.monotonic()
    t.start()
    time.sleep(0.3)
    requete(port, "POST", "/annuler")
    t.join(10)
    assert res["r"][0] == 499 and time.monotonic() - t0 < 5          # without cancellation: up to 600 s of waiting
    drapeau.unlink()
    assert requete(port, "POST", "/v1/audio/speech", json.dumps({"input": "x"}), JSON)[0] == 200
